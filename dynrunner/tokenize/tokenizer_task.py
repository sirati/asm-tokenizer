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
    match_filename,
    process_selection_arguments,
)
from tokenizer.arch_translation import arch_to_platform
from tokenizer.binary_discovery import BinaryHandle, walk_dataset
from tokenizer.output_filename import format_output_csv_filename
from tokenizer.variant_info import VariantInfo


_PHASE_ID = "tokenize"
_TYPE_ID = "tokenizer"

_logger = logging.getLogger(__name__)


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

        Two transport pathways converge on the same
        ``(BinaryHandle, VariantInfo)`` pair stream:

        * Local (default): ``walk_dataset`` from
          ``tokenizer.binary_discovery`` handles both legacy 4-axis
          filenames and the new sidecar-JSON format with per-directory
          dispatch, returning ``(handle, variant)`` pairs with
          ``handle.tarball`` set in sidecar mode.
        * Gateway / SSH (``args.source_already_staged`` set, SLURM
          mode): the Rust ``_native.find_items`` walker still drives
          remote discovery via the existing ``--gateway`` connection;
          its ``BinaryIdentifier`` outputs are adapted to
          ``VariantInfo`` via ``VariantInfo.from_legacy_filename`` for
          symmetry with the local pathway. Sidecar over SSH is not yet
          supported (TODO: extend ``walk_dataset`` to take a path-walk
          backend so the gateway transport can carry both formats).

        Selection filters (``--platform``, ``--compiler``,
        ``--compiler-versions``, ``--opt``, ``--exclude-subfolder``) are
        applied uniformly on the converged pair stream so both formats
        respect the same allowlists.
        """
        config = process_selection_arguments(args)
        filters: SelectionFilters = compile_selection_filters(config)

        gateway_url = (
            getattr(args, "gateway", None)
            if getattr(args, "source_already_staged", None)
            else None
        )

        pairs = self._iter_filtered_pairs(args, config, filters)
        sorted_items = list(self._sort_and_tag_pairs(pairs))

        if getattr(args, "skip_existing", False):
            output_root = getattr(args, "resolved_output_root", None)
            if output_root:
                completed = _collect_existing_output_filenames(
                    output_root, gateway_url
                )
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

    def _iter_filtered_pairs(
        self,
        args: Namespace,
        config,
        filters: SelectionFilters,
    ) -> Iterable[tuple[BinaryHandle, VariantInfo, int]]:
        """Yield ``(handle, variant, size)`` triples respecting selection filters.

        Dispatches between the local ``walk_dataset`` pathway and the
        gateway ``_native.find_items`` pathway based on whether
        ``--source-already-staged`` is set; both converge on the same
        triple shape so downstream sort + emit logic stays uniform.
        """
        if getattr(args, "source_already_staged", None):
            yield from self._iter_gateway_pairs(args, filters)
        else:
            yield from self._iter_local_pairs(config.source_dir, filters)

    def _iter_local_pairs(
        self,
        source_dir: Path,
        filters: SelectionFilters,
    ) -> Iterable[tuple[BinaryHandle, VariantInfo, int]]:
        """Walk ``source_dir`` locally via ``walk_dataset`` and apply filters."""
        for handle, variant in walk_dataset(source_dir):
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

    def _iter_gateway_pairs(
        self,
        args: Namespace,
        filters: SelectionFilters,
    ) -> Iterable[tuple[BinaryHandle, VariantInfo, int]]:
        """Walk a gateway-side path via ``_native.find_items`` over SSH.

        Only legacy 4-axis filenames are handled here — the Rust walker
        + ``visit()`` only mark files matching that format. Sidecar-over-
        SSH is a known gap (see ``discover_items`` docstring).
        """
        root = args.source_already_staged
        gateway_url = getattr(args, "gateway", None)
        self._filters: SelectionFilters | None = filters
        try:
            items = _native.find_items(self, root, gateway_url=gateway_url)
        finally:
            self._filters = None

        root_path = Path(root)
        for item in items:
            absolute_path = root_path / str(item.path)
            try:
                variant = VariantInfo.from_legacy_filename(absolute_path)
            except ValueError:
                continue
            handle = BinaryHandle(path=absolute_path, tarball=None)
            yield handle, variant, item.size

    def visit(
        self,
        parent_payload: str | None,
        subfolders: list,
        files: list,
    ) -> None:
        """Per-directory policy callback driven by `_native.find_items`.

        Used only on the gateway-SSH pathway (``_iter_gateway_pairs``);
        the local pathway uses ``walk_dataset`` directly. Per-file
        matching + per-subfolder exclude both delegate to the framework's
        ``match_filename`` / ``is_excluded_subfolder`` helpers so this
        stays in lockstep with the standard
        ``platform-compiler-version-opt-binary`` filename format.
        """
        filters = self._filters
        if filters is None:
            return

        current_rel = parent_payload or ""

        for folder in subfolders:
            child_rel = (
                f"{current_rel}/{folder.name}" if current_rel else folder.name
            )
            if is_excluded_subfolder(child_rel, filters):
                folder.enter(False)
            else:
                folder.enter(True, payload=child_rel)

        for f in files:
            identifier = match_filename(f.name, filters)
            if identifier is not None:
                f.mark(True, payload=identifier)

    @staticmethod
    def _sort_and_tag_pairs(
        pairs: Iterable[tuple[BinaryHandle, VariantInfo, int]],
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
                # `TaskInfo.path` is the file the framework uploads /
                # stages / hashes for this task. For sidecar tasks
                # that's the .tar.zst (the archive carries the binaries
                # to tokenize); the JSON sidecar is metadata-only and
                # already fully decoded into `variant` at discovery
                # time, so it doesn't need to travel across the wire.
                # Legacy tasks emit the binary file directly.
                wire_path = handle.tarball if handle.tarball is not None else handle.path
                yield TaskInfo(
                    path=wire_path,
                    size=size,
                    identifier=BinaryIdentifier(
                        binary_name=variant.pkg,
                        platform=variant.arch,
                        compiler=variant.compiler,
                        version=variant.compiler_version,
                        opt_level=variant.opt,
                    ),
                    phase_id=_PHASE_ID,
                    type_id=_TYPE_ID,
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


def _collect_existing_output_filenames(
    output_root: str, gateway_url: str | None
) -> set[str]:
    """Walk `output_root` via `_native.find_items` and return the set of
    file basenames present.

    A non-existent `output_root` (first run; framework hasn't created the
    directory yet) yields an empty set rather than an error — the
    skip-existing filter degrades to no-op on a fresh deployment instead
    of failing the whole dispatch.
    """
    collector = _OutputFilenameCollector()
    try:
        _native.find_items(collector, output_root, gateway_url=gateway_url)
    except OSError:
        # Path doesn't exist or unreadable — first-run case; treat as
        # "no completions yet".
        return set()
    return collector.filenames
