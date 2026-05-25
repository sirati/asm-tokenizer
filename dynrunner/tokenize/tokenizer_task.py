"""TokenizerTask: migrate binary files into per-binary CSV token streams.

The dynamic_runner framework dispatches one task type — the tokenizer
worker module — through one phase. The worker internally cycles through
three angr passes (the old TokenizerPhase enum) and a final
tokenization writeout; that internal phasing is invisible to the
framework now (the new design has no `get_stages`).
"""

from __future__ import annotations

import logging
import math
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

from dynamic_runner import _native
from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId

from dynrunner.binary_selection import (
    BinaryIdentifier,
    SelectionFilters,
    TaskInfo,
    add_asm_selection_arguments,
    compile_selection_filters,
    is_excluded_subfolder,
    process_selection_arguments,
)
from dynrunner.tokenize.identifier import TokenizerIdentifier
from tokenizer.arch_translation import arch_to_platform
from tokenizer.binary_discovery import BinaryHandle, walk_dataset
from tokenizer.output_filename import format_output_csv_filename
from tokenizer.variant_info import VariantInfo


_PHASE_ID = "tokenize"
_TYPE_ID = "tokenizer"

_logger = logging.getLogger(__name__)

# Ghidra workspace artifacts that pyghidra writes adjacent to each
# binary it touches (a sibling ``<binary-stem>_ghidra/`` directory
# populated with project metadata). When TokenizerTask runs on a corpus
# that retained those sidecars from a prior run, discovery would walk
# them and try to tokenize the metadata as if it were ELF — angr then
# fails with CLECompatibilityError on the Ghidra archive format.
_GHIDRA_WORKSPACE_DIR_SUFFIX = "_ghidra"
_GHIDRA_SIDECAR_EXTENSIONS: frozenset[str] = frozenset(
    {".gpr", ".rep", ".lock", ".lock~", ".bak~", ".prp"}
)


def _is_ghidra_workspace_artifact(path: Path) -> bool:
    """True if `path` is inside a ``*_ghidra/`` workspace directory or
    has a Ghidra sidecar extension. Used to filter discovery so prior
    Ghidra runs do not pollute the input set.
    """
    if any(seg.endswith(_GHIDRA_WORKSPACE_DIR_SUFFIX) for seg in path.parts[:-1]):
        return True
    name = path.name
    return any(name.endswith(ext) for ext in _GHIDRA_SIDECAR_EXTENSIONS)


class TokenizerPhase(str, Enum):
    """Worker-internal progress labels.

    The framework no longer cares about these — the new task protocol
    has only one phase per type. The worker still emits these strings
    via `PhaseUpdateResponse` so the runner's logs/progress bar can show
    which sub-stage of tokenization is running.
    """

    ANGR_1 = "angr-1"
    ANGR_2 = "angr-2"
    TOKENIZATION = "tokenization"


class TokenizerTask:
    """Task definition for binary tokenization."""

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        return (
            PhaseSpec(
                phase_id=_PHASE_ID,
                types=(
                    TaskTypeSpec(
                        type_id=_TYPE_ID,
                        worker_module="tokenizer",
                        # Old design: per-stage timeouts (None, None,
                        # 10.0s). The new design has one timeout per
                        # type; pick the tightest meaningful value (the
                        # final csv-writing keepalive).
                        timeout_seconds=10.0,
                    ),
                ),
            ),
        )

    # ── Item discovery ─────────────────────────────────────────────────

    def discover_items(
        self, source_dir: Path, args: Namespace
    ) -> Iterable[TaskInfo]:
        """Walk the source corpus, build TaskInfos with VariantInfo payload, sort.

        ``walk_dataset`` from ``tokenizer.binary_discovery`` does the
        walk; it handles both legacy 4-axis filenames and the
        sidecar-JSON folder format with per-directory dispatch,
        returning ``(handle, variant)`` pairs where ``handle.path`` is
        the binary file (and ``handle.variant_dir`` carries the
        folder in sidecar mode).

        Always operates on a real local filesystem from the discoverer's
        POV. Under ``--multi-computer slurm`` with
        ``--source-already-staged``, the framework runs discover_items
        on a promoted setup-secondary against its bind-mounted
        ``/app/src-network`` (a local path from that secondary's POV);
        no SSH walk involved.

        Selection filters (``--platform``, ``--compiler``,
        ``--compiler-versions``, ``--opt``, ``--name-regex``,
        ``--exclude-subfolder``) are applied uniformly post-walk.
        """
        config = process_selection_arguments(args)
        filters: SelectionFilters = compile_selection_filters(config)

        pairs = self._iter_local_pairs(source_dir, filters)
        # `source_root` is the absolute prefix the discovery walks
        # produced their `BinaryHandle.path` / `.tarball` against. The
        # framework treats `TaskInfo.path` as the wire identifier and
        # forwards it to the worker as `task.relative_path`; emitting
        # absolute paths here makes the wire identifier non-portable
        # across primary/secondary FS-views, so we strip the prefix
        # at the TaskInfo boundary.
        sorted_items = list(self._sort_and_tag_pairs(pairs, source_dir))

        if getattr(args, "skip_existing", False):
            output_root = getattr(args, "resolved_output_root", None)
            if output_root:
                completed = _collect_existing_output_filenames(output_root)
                before = len(sorted_items)
                sorted_items = [
                    it
                    for it in sorted_items
                    if self.get_output_filename_pattern(_TYPE_ID, it)
                    not in completed
                ]
                skipped = before - len(sorted_items)
                _logger.info(
                    "skip-existing: %d candidates → %d remaining "
                    "(%d skipped via %d existing outputs at %s)",
                    before,
                    len(sorted_items),
                    skipped,
                    len(completed),
                    output_root,
                )

        return sorted_items

    def _iter_local_pairs(
        self,
        source_dir: Path,
        filters: SelectionFilters,
    ) -> Iterable[tuple[BinaryHandle, VariantInfo, int]]:
        """Walk ``source_dir`` locally via ``walk_dataset`` and apply filters."""
        for handle, variant in walk_dataset(source_dir):
            if _is_ghidra_workspace_artifact(handle.path):
                continue
            if not _variant_passes_filters(variant, filters):
                continue
            if not _path_passes_subfolder_filter(handle.path, source_dir, filters):
                continue
            try:
                # ``BinaryHandle.binary_size()`` returns the uncompressed
                # size of the binary content regardless of transport
                # (legacy filesystem stat / sidecar tarball-header sum).
                # The duality lives inside the handle so this loop
                # doesn't branch on layout.
                size = handle.binary_size()
            except OSError:
                continue
            yield handle, variant, size

    @staticmethod
    def _sort_and_tag_pairs(
        pairs: Iterable[tuple[BinaryHandle, VariantInfo, int]],
        source_root: Path,
    ) -> Iterable[TaskInfo]:
        """Group by ``variant.pkg``; sort within group by size DESC; order
        groups by group-average size DESC; emit ``TaskInfo`` instances
        tagged with this task's phase + type.

        Carries the ``VariantInfo`` (full identity incl. ``variant_id``
        and ``extra_metadata``) plus the optional sidecar tarball path on
        ``TaskInfo.payload`` so the worker (S3-tok) can reconstruct
        ``VariantInfo`` and locate the archive without re-parsing the
        filename / sidecar JSON. ``TaskInfo.identifier`` keeps the locked
        5-field ``BinaryIdentifier`` shape for framework FFI compatibility
        (per the master plan's "VariantInfo wraps BinaryIdentifier"
        decision).

        affinity_id stays ``None`` — the tokenizer has no cache-locality
        classes worth exploiting.
        """
        materialised = list(pairs)
        groups: dict[str, list[tuple[BinaryHandle, VariantInfo, int]]] = defaultdict(list)
        for triple in materialised:
            _, variant, _ = triple
            groups[variant.pkg].append(triple)

        group_averages: list[
            tuple[str, float, list[tuple[BinaryHandle, VariantInfo, int]]]
        ] = []
        for pkg, group in groups.items():
            avg_size = sum(size for _, _, size in group) / len(group)
            group.sort(key=lambda t: t[2], reverse=True)
            group_averages.append((pkg, avg_size, group))
        group_averages.sort(key=lambda x: x[1], reverse=True)

        for _, _, group in group_averages:
            for handle, variant, size in group:
                # `TaskInfo.path` is the binary file the framework
                # uploads / stages / hashes for this task. Sidecar
                # mode points at `<variant_dir>/<pkg>`; legacy mode
                # points at the canonical-format binary on disk. The
                # JSON sidecar (in sidecar mode) is metadata-only and
                # already fully decoded into `variant` at discovery
                # time, so it doesn't need to travel across the wire.
                #
                # Emit RELATIVE-to-source-root paths so the wire
                # identifier is portable across primary/secondary
                # filesystem views. The framework forwards
                # `TaskInfo.path` as `task.relative_path` to workers,
                # which then resolve it via `_SOURCE_DIR / rel`.
                # Absolute primary-side paths break that semantics
                # in SLURM dispatch (the secondary's source mount is
                # at a different absolute location).
                try:
                    wire_path = handle.path.relative_to(source_root)
                except ValueError:
                    wire_path = handle.path
                # `TokenizerIdentifier.identifier_key()` is the canonical
                # per-task identity string the framework consumes via
                # `TaskInfo.task_id`. The framework's memprofile sampler
                # gates on a non-None `task_id` (output filenames key on
                # it); without it the sampler short-circuits with a
                # debug log. Keep the dataclass as the single owner of
                # the canonical string format so we don't duplicate it
                # at the construction site.
                identifier = TokenizerIdentifier(
                    binary_name=variant.pkg,
                    platform=variant.arch,
                    compiler=variant.compiler,
                    version=variant.compiler_version,
                    opt_level=variant.opt,
                )
                yield TaskInfo(
                    path=wire_path,
                    size=size,
                    identifier=BinaryIdentifier(
                        binary_name=identifier.binary_name,
                        platform=identifier.platform,
                        compiler=identifier.compiler,
                        version=identifier.version,
                        opt_level=identifier.opt_level,
                    ),
                    phase_id=_PHASE_ID,
                    type_id=_TYPE_ID,
                    task_id=identifier.identifier_key(),
                    payload=_build_payload(variant),
                )

    # ── Per-type plumbing ──────────────────────────────────────────────

    def estimate_memory(self, item: TaskInfo) -> int:
        """Power-law estimator: RAM_MiB = 430.870 * size_MiB^1.051 + 260.15.

        R² = 0.9866, RMSE = 203.66 MiB. Adds the RMSE so we
        rather over-estimate than under-estimate.
        """
        mb = item.size / 1024 / 1024
        a = 430.870
        b = 1.051
        c = 260.15
        rmse = 203.66
        ram_mb = a * (mb**b) + c + rmse
        return math.ceil(ram_mb * 1024 * 1024)

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        # Asm-binary corpus-shape filter flags (--platform, --compiler,
        # --compiler-versions, --opt, --file-format, --debugs,
        # --exclude-subfolder, --name-regex, --version-regex, --opt-regex)
        # used to live in `dynamic_runner._shared.selection_args`. After
        # framework commit 6c65bb7 they're consumer-owned; we vendor them
        # under `dynrunner/binary_selection/` and register them per-task
        # here so the `discover_items` body can consume them via
        # `process_selection_arguments(args)`.
        add_asm_selection_arguments(parser)
        self.add_private_task_arguments(parser)

    def add_private_task_arguments(self, parser: ArgumentParser) -> None:
        """Task-private argparse registrations (no asm-selection flags).

        The composite ``FullPipelineTask`` calls this directly so it can
        register the asm-selection flag block exactly once across all
        three child tasks; the standalone entry point still funnels
        through ``add_task_arguments`` (which composes both).
        """

    def build_worker_command_args(
        self,
        type_id: TypeId,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        cmd_args = ["--platform", "auto"]
        if hasattr(args, "simulate_errors") and args.simulate_errors is not None:
            cmd_args.extend(["--simulate-errors", str(args.simulate_errors)])
        return cmd_args

    def get_output_filename_pattern(
        self, type_id: TypeId, item: TaskInfo
    ) -> str:
        """Output filename for ``item``.

        Composes the canonical-format
        ``<arch>-<compiler>-<compiler_version>-<opt>_<pkg>`` from the
        VariantInfo carried in ``item.payload`` (sidecar mode) or
        recovered from the legacy filename (legacy mode), then appends
        ``_output.csv`` — with a ``__<variant_id:08x>`` suffix on the
        binary-name slot when ``variant_id != 0`` to disambiguate
        same-canonical-4 sidecar variants. Delegates the format string
        to ``tokenizer.output_filename`` so the worker's CSV writeout
        and the build_memmap phase's pairing walk use the same single
        source of truth.

        Using ``item.path.name`` here would mis-name sidecar tasks (the
        JSON sidecar's filename, e.g.
        ``clang10_armv7l-hf_Oz_15f3f338.json``, does NOT match the
        canonical regex consumed by build_memmap and can't carry the
        ``pkg`` slot the pairing walk relies on); reconstructing from
        the canonical fields keeps both formats on a single naming
        scheme.
        """
        variant = _payload_variant(item)
        return format_output_csv_filename(
            variant.arch,
            variant.compiler,
            variant.compiler_version,
            variant.opt,
            variant.pkg,
            variant.variant_id,
        )

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def on_run_start(
        self, source_dir: Path, output_dir: Path, args: Namespace
    ) -> None:
        pass

    def on_run_end(self, success: bool) -> None:
        pass

    def on_phase_start(self, phase_id: str) -> None:
        pass

    def on_phase_end(self, phase_id: str, completed: int, failed: int) -> None:
        pass


_PAYLOAD_VARIANT_KEY = "variant"


def _variant_to_payload_dict(variant: VariantInfo) -> dict[str, Any]:
    """Serialise a ``VariantInfo`` to the JSON-safe dict the worker
    receives via ``TaskInfo.payload``.

    Mirrors the dataclass fields verbatim — including
    ``extra_metadata`` so the opaque pass-through carries through to
    the worker without enumeration. ``__hash__``/``__eq__`` excluded
    fields aren't excluded here: identity is one concern, transport is
    another.
    """
    return {
        "arch": variant.arch,
        "compiler": variant.compiler,
        "compiler_version": variant.compiler_version,
        "opt": variant.opt,
        "pkg": variant.pkg,
        "variant_id": variant.variant_id,
        "extra_metadata": variant.extra_metadata,
    }


def _build_payload(variant: VariantInfo) -> dict[str, Any]:
    """Build the ``TaskInfo.payload`` dict for one variant.

    The framework FFI serialises the dict to JSON on the wire (see
    ``_native.TaskInfo.payload_json``); the worker decodes back to a
    dict and reconstructs ``VariantInfo`` from
    ``payload[_PAYLOAD_VARIANT_KEY]``.

    Tarball location is no longer carried in the payload — for
    sidecar tasks ``TaskInfo.path`` IS the tarball (so the framework
    uploads/stages the right file), and the worker resolves it via
    ``source_dir / task.relative_path``. The sidecar/legacy fork on
    the worker side is driven by ``variant.variant_id != 0``.
    """
    return {
        _PAYLOAD_VARIANT_KEY: _variant_to_payload_dict(variant),
    }


def _payload_variant(item: TaskInfo) -> VariantInfo:
    """Reconstruct the ``VariantInfo`` carried on ``item.payload``.

    Mirrors the encode-side ``_variant_to_payload_dict`` field order
    verbatim. Falls back to ``VariantInfo.from_legacy_filename`` when
    the payload is missing or unshaped — keeps
    ``get_output_filename_pattern`` working for any TaskInfo a framework
    consumer constructs without going through ``_sort_and_tag_pairs``
    (e.g. the gateway-SSH path which still has the legacy filename on
    ``item.path``).
    """
    payload = item.payload or {}
    variant_dict = payload.get(_PAYLOAD_VARIANT_KEY)
    if variant_dict is None:
        return VariantInfo.from_legacy_filename(item.path)
    return VariantInfo(
        arch=variant_dict["arch"],
        compiler=variant_dict["compiler"],
        compiler_version=variant_dict["compiler_version"],
        opt=variant_dict["opt"],
        pkg=variant_dict["pkg"],
        variant_id=int(variant_dict.get("variant_id", 0)),
        extra_metadata=variant_dict.get("extra_metadata", {}) or {},
    )


def _variant_passes_filters(
    variant: VariantInfo, filters: SelectionFilters
) -> bool:
    """Apply the corpus-shape allowlists (platforms, compiler,
    compiler_versions, normalized_opt_levels) to a parsed VariantInfo.

    Mirrors the gating semantics ``match_filename`` applies on the
    legacy ``(parsed-tuple, filters)`` boundary, but operates on the
    already-parsed VariantInfo so both legacy and sidecar pathways
    share the same gate without re-parsing the source filename.
    """
    if filters.platforms is not None:
        # Sidecar archs like `armv7l-hf` / `x86_64` aren't in the
        # canonical Platform allowlist (`x64`, `arm32`, ...). Translate
        # to the canonical form before comparing so the dispatcher's
        # default Platform-allowlist accepts both layouts uniformly.
        # Legacy variants already carry canonical arch names verbatim
        # from `parse_binary_filename`, so the translator is a no-op
        # for them — but we still try-translate first to avoid two
        # comparison branches.
        try:
            canonical_arch = arch_to_platform(variant.arch)
        except ValueError:
            canonical_arch = variant.arch
        if (
            variant.arch not in filters.platforms
            and canonical_arch not in filters.platforms
        ):
            return False
    if filters.compiler and variant.compiler != filters.compiler:
        return False
    if (
        filters.compiler_versions
        and variant.compiler_version not in filters.compiler_versions
    ):
        return False
    if (
        filters.normalized_opt_levels
        and variant.opt not in filters.normalized_opt_levels
    ):
        return False
    # Name-regex post-parse check. The local pathway (`walk_dataset` +
    # `VariantInfo.from_legacy_filename`) doesn't run filenames through
    # `match_filename`, so `binary_format`'s embedded name-regex never
    # gets applied there. `filters.name_pattern` is the same regex
    # exposed as a standalone `re.Pattern`; checking it against
    # `variant.pkg` (the parsed binary-name field) closes the gap.
    # Without this, `--name-regex minigzipsh` on the local pathway
    # admits ~13× more items than the SLURM-pathway sibling (BUGS.md #3).
    if filters.name_pattern is not None and not filters.name_pattern.search(variant.pkg):
        return False
    return True


def _path_passes_subfolder_filter(
    path: Path, source_dir: Path, filters: SelectionFilters
) -> bool:
    """Apply the ``--exclude-subfolder`` substring filter to a file's
    parent-relative path.

    ``walk_dataset`` does not prune subdirs (it has no knowledge of
    consumer-side selection state), so any exclude rule must filter
    the emitted pairs after the walk. Files at the root itself
    (rel_path == '.') are always retained, matching the existing
    ``is_excluded_subfolder`` contract.
    """
    try:
        rel_dir = path.parent.relative_to(source_dir)
    except ValueError:
        # Path lives outside source_dir — defensive; walk_dataset only
        # yields paths under the source root, but this keeps the
        # contract robust if the caller passes a non-canonicalised
        # source_dir.
        return True
    rel_dir_str = str(rel_dir)
    return not is_excluded_subfolder(rel_dir_str, filters)


class _OutputFilenameCollector:
    """Single-purpose visitor for `_native.find_items`: walks an output
    tree and records every basename in `self.filenames`.

    Used by `TokenizerTask.discover_items` when `--skip-existing` is set
    so the task can drop already-completed binaries from the source
    candidate list. Stays at the task level (not the framework) so the
    "what counts as already done" decision lives next to
    `get_output_filename_pattern`, the matching key.

    Doesn't call `f.mark(...)` — we don't need `find_items` to return
    `TaskInfo` records, just the side-effect set of basenames.
    """

    def __init__(self) -> None:
        self.filenames: set[str] = set()

    def visit(
        self,
        parent_payload: object,
        subfolders: list,
        files: list,
    ) -> None:
        for folder in subfolders:
            folder.enter(True)
        for f in files:
            self.filenames.add(f.name)


def _collect_existing_output_filenames(output_root: str) -> set[str]:
    """Walk `output_root` via `_native.find_items` and return the set of
    file basenames present.

    A non-existent `output_root` (first run; framework hasn't created the
    directory yet) yields an empty set rather than an error — the
    skip-existing filter degrades to no-op on a fresh deployment instead
    of failing the whole dispatch.

    Post the 2026-05-13 native-task-discovery refactor (`6f583fc`),
    `_native.find_items` accepts only `(task_definition, root)` and
    always operates on a local filesystem; the gateway-SSH walk path
    is gone framework-side.
    """
    collector = _OutputFilenameCollector()
    try:
        _native.find_items(collector, output_root)
    except OSError:
        # Path doesn't exist or unreadable — first-run case; treat as
        # "no completions yet".
        return set()
    return collector.filenames
