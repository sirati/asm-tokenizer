"""TaskDefinition for the per-binary-group memmap builder.

The library function `tokenizer.memmap_builder.builder.build_memmap_files`
consumes a list of `BinaryVersionInfo` (one per (compiler, version, opt)
triple) for a single `binary_name` group. The wire protocol's `task:`
form (FR-3, dynamic_runner) carries a per-task `payload` JSON value,
so the starting instance does ALL discovery + pairing + grouping inside
`discover_items` and emits one TaskInfo per group with the per-version
pairing data inline in `TaskInfo.payload`. The worker reads
`command.payload`, reconstructs `BinaryVersionInfo`, and calls
`build_memmap_files`. No manifest file on disk; works identically
under local dispatch and under SLURM `--source-already-staged`.

`uses_file_based_items = False` tells the framework that
`TaskInfo.path` isn't a real filesystem path — it's just an opaque
identifier (the `binary_name`). Primary-side staging
(content-hashing, StageFile, src_network resolution) is therefore
skipped for these items; the wire's `local_path` carries the
binary_name verbatim and the worker uses the inline payload.
"""

from __future__ import annotations

import dataclasses
import logging
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from dynamic_runner import _native
from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId

from tokenizer.arch_translation import arch_to_platform
from tokenizer.variant_info import split_variant_id_suffix

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


_logger = logging.getLogger(__name__)

_PHASE_ID = "memmap"
_TYPE_ID = "memmap"

# Filename suffixes that the tokenize phase + vocab unifier emit.
# Appended to the user-supplied --file-format. The framework's
# format-string DSL maps 1-char field shorthands `p` (platform), `c`
# (compiler), `name` etc.; the backslashes are the DSL's escape so
# literal `p`/`c` aren't gobbled as field placeholders. See
# `dynamic_runner._shared.binary_info.process_escaping`.
# Pairing is by `BinaryIdentifier` equality after parsing.
_CSV_SUFFIX = "_out\\put.\\csv"
_MAPPING_SUFFIX = "_out\\put.ma\\p\\ping.b64\\c"
# Per-version metadata sidecar emitted by the tokenize worker beside
# each `_output.csv`. Forward-compat: legacy outputs from before the
# sidecar emitter shipped have no `_meta.json`, in which case the
# corresponding payload entry's `meta_path` is null. Unlike the CSV
# and mapping suffixes, none of the literal characters here collide
# with field-name shorthands (no `p`/`c`/`opt`/`name` substring), so
# no `\\` escapes are needed.
_META_SUFFIX = "_meta.json"

# Variant-id suffix on the binary_name slot of sidecar-emitted
# filenames: `<binary>__<8hex>` (e.g. `helloworld__15f3f338`) is
# peeled off by ``tokenizer.variant_info.split_variant_id_suffix``
# — the canonical owner of the regex. The greedy
# `binary_name = ".+"` regex in the file-format parser swallows the
# `__<8hex>` into `binary_name`; the helper strips it so the pairing
# key collapses to the same group across legacy and sidecar variants
# while still distinguishing different variants of the same binary.


def _csv_format(base_format: str) -> str:
    return base_format + _CSV_SUFFIX


def _mapping_format(base_format: str) -> str:
    return base_format + _MAPPING_SUFFIX


def _meta_format(base_format: str) -> str:
    return base_format + _META_SUFFIX


class _FormatVisitor:
    """Visitor for `_native.find_items`: marks every file matching the
    pre-compiled `SelectionFilters` with its parsed `BinaryIdentifier`
    as the per-file payload. One instance is used per pass (CSV pass,
    mapping pass); each pass has its own filters compiled from a
    different file_format.
    """

    def __init__(self, filters: SelectionFilters) -> None:
        self._filters = filters

    def visit(
        self,
        parent_payload: str | None,
        subfolders: list,
        files: list,
    ) -> None:
        current_rel = parent_payload or ""
        for folder in subfolders:
            child_rel = (
                f"{current_rel}/{folder.name}" if current_rel else folder.name
            )
            if is_excluded_subfolder(child_rel, self._filters):
                folder.enter(False)
            else:
                folder.enter(True, payload=child_rel)
        for f in files:
            identifier = match_filename(f.name, self._filters)
            if identifier is not None:
                f.mark(True, payload=identifier)


class _OutputFilenameCollector:
    """Visitor that records every file's basename. Used by the
    task-side `--skip-existing` filter to enumerate already-completed
    outputs at `args.resolved_output_root` (gateway-side in SLURM
    pre-staged mode, local otherwise)."""

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


def _walk_with_filters(
    root: str, filters: SelectionFilters
) -> list:
    """Run `_native.find_items` against `root` with a `_FormatVisitor`
    parameterised by `filters`. Returns the list of marked PyTaskInfo
    objects (relative_path under `root`, parsed `BinaryIdentifier` as
    `.identifier`).
    """
    visitor = _FormatVisitor(filters)
    return _native.find_items(visitor, root)


def _collect_existing_output_filenames(
    output_root: str,
) -> set[str]:
    """Walk `output_root` via `_native.find_items` and return the set of
    file basenames present. A non-existent or unreadable `output_root`
    yields an empty set rather than failing the dispatch — first-run
    deployments don't have the directory yet.
    """
    collector = _OutputFilenameCollector()
    try:
        _native.find_items(collector, output_root)
    except OSError:
        return set()
    return collector.filenames


def _config_with_format(config, file_format: str):
    """Return a copy of the parsed `SelectionConfig` with `file_format`
    overridden — needed because the CSV pass and the mapping pass share
    every other field but differ on the filename format string.
    """
    return dataclasses.replace(config, file_format=file_format)


class MemmapBuilderTask:
    """Memmap-builder task: one phase, one type, one item per binary_name group."""

    # Items represent binary_name groups (the worker iterates a list of
    # `BinaryVersionInfo` per group); `TaskInfo.path` is an opaque
    # identifier (the binary_name itself), not a real filesystem path.
    # Setting `uses_file_based_items = False` tells the framework to
    # skip its file-based staging machinery (content-hashing, StageFile
    # transfer, src_network resolve) for these items — see FR-2 in the
    # dynamic_runner protocol.
    uses_file_based_items: bool = False

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        return (
            PhaseSpec(
                phase_id=_PHASE_ID,
                types=(
                    TaskTypeSpec(
                        type_id=_TYPE_ID,
                        worker_module="dynrunner.build_memmap.worker",
                        # Memmap building scans large CSVs and writes
                        # multiple bin/csv outputs; per-group runtime is
                        # dominated by I/O on the largest version. 5 min
                        # is a generous keepalive ceiling — the worker
                        # has no inner-loop keepalive yet.
                        timeout_seconds=300.0,
                    ),
                ),
            ),
        )

    # ── Item discovery ─────────────────────────────────────────────────

    def discover_items(
        self, source_dir: Path, args: Namespace
    ) -> Iterable[TaskInfo]:
        """Discover CSV outputs + mapping files via two `find_items`
        passes, pair by `BinaryIdentifier`, group by `binary_name`,
        emit one TaskInfo per group with the per-version pairing data
        inline in `TaskInfo.payload`. Walks resolve locally against
        the discoverer-side `source_dir` (either submitter-side for
        local mode, or the promoted setup-secondary's bind-mount under
        `--multi-computer slurm`). Payload paths are relative to the
        walk roots; the worker resolves them against its own
        source_dir at run time.
        """
        config = process_selection_arguments(args)

        # Always operates on a real local filesystem from the discoverer's
        # POV. Under --multi-computer slurm with --source-already-staged,
        # the framework runs discover_items on a promoted setup-secondary
        # against its bind-mounted local path (passed here as source_dir);
        # no SSH walk involved. Mirrors TokenizerTask.discover_items.
        # `--vocab-source` (if set) overrides the mapping-pass root;
        # otherwise it tracks the source root.
        source_root = str(source_dir)
        vocab_root = (
            str(Path(args.vocab_source).resolve())
            if getattr(args, "vocab_source", None)
            else str(source_dir)
        )

        csv_filters = compile_selection_filters(
            _config_with_format(config, _csv_format(config.file_format))
        )
        mapping_filters = compile_selection_filters(
            _config_with_format(config, _mapping_format(config.file_format))
        )
        meta_filters = compile_selection_filters(
            _config_with_format(config, _meta_format(config.file_format))
        )

        csv_items = _walk_with_filters(source_root, csv_filters)
        mapping_items = _walk_with_filters(vocab_root, mapping_filters)
        # Meta sidecars live next to the CSVs (tokenize phase emits both
        # together), so the meta walk shares the CSV walk root. A meta
        # file's absence is benign: legacy build_memmap output predates
        # the sidecar writer, so each entry's `meta_path` defaults to
        # `None` when no sibling `_meta.json` is found.
        meta_items = _walk_with_filters(source_root, meta_filters)

        # Pair CSV ↔ mapping ↔ meta by explicit tuple key.
        # `_native.find_items` returns `PyTaskInfo` objects whose
        # `.identifier` is a Rust pyclass (`PyBinaryIdentifier`). The
        # Rust pyclass doesn't implement `__eq__`/`__hash__` semantically,
        # so a dict keyed on the identifier object compares by Python
        # default (pointer identity) — pairing fails. Tuple keys force
        # value equality on the relevant fields.
        #
        # The 5th field `variant_id` distinguishes sidecar variants
        # whose other 4 axes collide (e.g. two builds of the same
        # `<arch,compiler,version,opt>` differing only in flag_set).
        # Legacy filenames without a `__<8hex>` suffix yield
        # `variant_id=0`, so legacy and sidecar items co-exist in one
        # `binary_name` group and pair correctly within their
        # canonical-4 + variant_id partition.
        def _key(item) -> tuple[str, str, str, str, str, int]:
            ident = item.identifier
            stripped_name, variant_id = split_variant_id_suffix(ident.binary_name)
            return (
                stripped_name,
                arch_to_platform(ident.platform),
                ident.compiler,
                ident.version,
                ident.opt_level,
                variant_id,
            )

        mapping_by_key = {_key(m): m for m in mapping_items}
        meta_by_key = {_key(m): m for m in meta_items}
        matched_csv_items = []
        unmatched_csv = []
        for csv_item in csv_items:
            if _key(csv_item) in mapping_by_key:
                matched_csv_items.append(csv_item)
            else:
                unmatched_csv.append(csv_item)
        if unmatched_csv:
            _logger.warning(
                "%d CSV file(s) have no matching mapping file:", len(unmatched_csv)
            )
            for csv_item in unmatched_csv:
                _logger.warning("  %s", _key(csv_item))

        # Group by stripped binary_name (the variant_id is part of the
        # pairing key, NOT the group key — same binary built with
        # different metadata still belongs to one memmap group).
        groups: dict[str, list] = defaultdict(list)
        for csv_item in matched_csv_items:
            stripped_name, _vid = split_variant_id_suffix(
                csv_item.identifier.binary_name
            )
            groups[stripped_name].append(csv_item)

        group_items: list[TaskInfo] = []
        for binary_name, group in groups.items():
            entries = []
            total_size = 0
            for csv_item in group:
                key = _key(csv_item)
                map_item = mapping_by_key[key]
                meta_item = meta_by_key.get(key)
                total_size += csv_item.size
                # ``filename`` is the variant's stable on-disk identity.
                # Two sidecar layouts coexist on LMU NFS:
                #
                #   * **legacy tarball-derived**:
                #     ``<bin>/<variant_dir>/<csv>`` — parent is the
                #     variant folder (e.g. ``clang10_aarch64_O0_90c970e8``,
                #     or pre-folder-migration ``clang10…tar.zst``).
                #   * **asm-dataset-nix folder-native**:
                #     ``<bin>/<variant_dir>/<bin>/<csv>`` — parent is
                #     the *binary subfolder* (e.g. ``hello``),
                #     grandparent is the variant folder.
                #
                # Pre-2026-05-17 the code unconditionally used
                # ``parent.name`` and collided every folder-native
                # variant of the same binary onto ``filename=<bin>``,
                # losing the variant identity in the emitted
                # ``_variants.csv``. The variant_id hex (8 chars,
                # parsed from the canonical-4 ``__<8hex>`` filename
                # suffix) is part of every variant folder name by
                # asm-dataset-nix convention, so the layout check is:
                # if the parent's name contains the variant-id hex
                # suffix, parent IS the variant folder; otherwise the
                # variant folder is the grandparent.
                #
                # Legacy 4-axis Dataset-1 variants (``variant_id == 0``,
                # no per-variant folder) keep the original
                # ``<binary_basename>`` shape.
                csv_path_obj = Path(csv_item.path)
                variant_id = key[5]
                if variant_id != 0:
                    variant_id_hex = f"{variant_id:08x}"
                    parent_name = csv_path_obj.parent.name
                    if variant_id_hex in parent_name:
                        variant_filename = parent_name
                    else:
                        variant_filename = csv_path_obj.parent.parent.name
                else:
                    name = csv_path_obj.name
                    variant_filename = (
                        name[: -len("_output.csv")]
                        if name.endswith("_output.csv")
                        else csv_path_obj.stem
                    )
                entries.append(
                    {
                        # Per-version paths are kept *relative* to the
                        # walk roots so the worker resolves them against
                        # its own `source_dir` / `vocab_dir` at run
                        # time. In SLURM pre-staged mode the worker
                        # sees the bind-mounted in-container root
                        # (`/app/src-network`) and joins these relatives
                        # against it; locally the worker sees the
                        # primary's filesystem root. Emitting absolutes
                        # here would inline the primary's gateway path
                        # into the payload, which the container can't
                        # resolve through its bind-mount.
                        "csv_path": str(csv_item.path),
                        "mapping_path": str(map_item.path),
                        "filename": variant_filename,
                        # `meta_path` is forward-compat scaffolding for
                        # the per-variant metadata sidecar that the
                        # tokenize worker emits (`_meta.json`). Null
                        # when absent (legacy outputs predating the
                        # sidecar). Worker consumption is downstream:
                        # `meta_path` reconstructs `VariantInfo` so the
                        # builder library can persist `_versions.json`.
                        "meta_path": (
                            str(meta_item.path) if meta_item is not None else None
                        ),
                        "arch": arch_to_platform(csv_item.identifier.platform),
                        "compiler": csv_item.identifier.compiler,
                        "compilerversion": csv_item.identifier.version,
                        "opt": csv_item.identifier.opt_level,
                        # `variant_id` is the integer suffix peeled off
                        # the parsed binary_name (see
                        # `split_variant_id_suffix`). 0 for legacy
                        # filenames, otherwise the 8-hex suffix decoded
                        # as base-16. Disambiguates same-`pkg` variants
                        # that share the canonical-4 axes.
                        "variant_id": key[5],
                    }
                )

            representative = group[0]
            group_items.append(
                TaskInfo(
                    # `path` is the binary_name — not a file. The
                    # framework treats it opaquely because of
                    # uses_file_based_items=False; on the wire it
                    # becomes the worker's `command.relative_path`.
                    path=Path(binary_name),
                    size=total_size,
                    identifier=BinaryIdentifier(
                        binary_name=binary_name,
                        platform=arch_to_platform(representative.identifier.platform),
                        compiler=representative.identifier.compiler,
                        version=representative.identifier.version,
                        opt_level=representative.identifier.opt_level,
                    ),
                    phase_id=_PHASE_ID,
                    type_id=_TYPE_ID,
                    affinity_id=None,
                    payload={
                        "binary_name": binary_name,
                        "versions": entries,
                    },
                )
            )

        # Largest groups first (rough wallclock heuristic — total CSV
        # size is the dominant work driver).
        group_items.sort(key=lambda ti: ti.size, reverse=True)

        # Task-side --skip-existing: walk args.resolved_output_root for
        # already-produced index files; drop matching groups.
        if getattr(args, "skip_existing", False):
            output_root = getattr(args, "resolved_output_root", None)
            if output_root:
                completed = _collect_existing_output_filenames(output_root)
                before = len(group_items)
                group_items = [
                    ti
                    for ti in group_items
                    if self.get_output_filename_pattern(_TYPE_ID, ti)
                    not in completed
                ]
                _logger.info(
                    "skip-existing: %d candidates → %d remaining "
                    "(%d skipped via %d existing outputs at %s)",
                    before,
                    len(group_items),
                    before - len(group_items),
                    len(completed),
                    output_root,
                )

        return group_items

    # ── Per-type plumbing ──────────────────────────────────────────────

    def estimate_memory(self, item: TaskInfo) -> int:
        """Estimate per-group RAM. ``item.size`` is the sum of CSV sizes
        in this group; the builder's working set is dominated by
        ``lockstep_function_match`` buffering + numpy mapping arrays.
        We have no fitted model yet, so use 4× size + 512 MiB floor +
        256 MiB constant overhead.
        """
        return max(4 * item.size, 512 * 1024 * 1024) + 256 * 1024 * 1024

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        # Vendored asm-binary corpus selection flags (--platform etc.) —
        # see TokenizerTask.add_task_arguments for rationale.
        add_asm_selection_arguments(parser)
        self.add_private_task_arguments(parser)

    def add_private_task_arguments(self, parser: ArgumentParser) -> None:
        """Task-private argparse registrations (no asm-selection flags).

        See ``TokenizerTask.add_private_task_arguments`` for the
        composite-pipeline rationale.
        """
        parser.add_argument(
            "--vocab-source",
            type=str,
            default=None,
            help=(
                "Source directory for vocabulary and mapping files. "
                "If not specified, uses the same as --source / "
                "--source-already-staged."
            ),
        )

    def build_worker_command_args(
        self,
        type_id: TypeId,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        # Worker resolves payload `csv_path` against `--source` and
        # `mapping_path` against `--vocab-source` (which the worker
        # defaults to `--source` when unset). For SLURM
        # `--source-already-staged` mode the user-supplied
        # `args.vocab_source` is a local primary-side path that is
        # *not* meaningful inside the container, so we don't forward
        # it; the worker's vocab_dir falls back to source_dir which IS
        # the bind-mount root. Only forward `--vocab-source` for plain
        # local dispatch where the path is meaningful at worker time.
        cmd: list[str] = []
        if getattr(args, "vocab_source", None) and not getattr(
            args, "source_already_staged", None
        ):
            cmd.extend(["--vocab-source", str(args.vocab_source)])
        return cmd

    def get_output_filename_pattern(
        self, type_id: TypeId, item: TaskInfo
    ) -> str:
        # `build_memmap_files` writes <binary_name>_index.bin (and a
        # few siblings — _data.bin, _meta.json, etc.). Pick the index
        # file as the canonical done-marker for skip-existing checks.
        return f"{item.binary_name}_index.bin"

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
