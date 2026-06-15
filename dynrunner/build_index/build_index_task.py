"""TaskDefinition for the per-binary index-build phase (phase 4).

Single concern: expose the two existing per-binary index generators —
``tokenizer.aligned_data.realized_lengths`` (the realized-length
sidecars) and ``tokenizer.aligned_data.sorted_index`` (the sorted-index
``.idx`` files) — as TWO dynrunner task types inside ONE phase, with the
sorted-index task declared DEPENDENT on its same-binary realized-length
task so the framework gates dispatch per binary.

This module ORCHESTRATES; it owns NO generation logic. Discovery scans
the memmap directory (``discover_binaries``, the same scan both standalone
CLIs use) and emits, per binary, one ``realized_lengths`` TaskInfo and one
``sorted_index`` TaskInfo. The sorted-index item carries
``task_depends_on=(<rlen task_id>,)`` so the framework's per-task
dependency machinery (``TaskInfo.task_depends_on`` →
``TaskState::Blocked``) holds the index build until its realized-length
sibling has terminated. The two workers each call exactly one library
entry point:

  * ``realized_lengths`` worker → ``generate_realized_lengths``
  * ``sorted_index`` worker     → ``write_sorted_index_files``

``uses_file_based_items = False``: ``TaskInfo.path`` is the binary_name,
an opaque identifier — not a real filesystem path. The framework's
file-based staging (content-hashing, StageFile transfer, src_network
resolve) is therefore skipped for these items, exactly like
``MemmapBuilderTask``; the worker re-derives every path from its own
``--source`` (the memmap dir) plus the binary_name on the wire.

Dependency boundary (the design-first sentence):

  *Given the memmap directory + a binary name, emit a realized-length
  task and a sorted-index task that DEPENDS on it — owning neither
  generator's internals, only the (b)→(a) edge between them.*

The per-(reduction, depth) selection (``--mode`` / ``--depth``) and the
gating/duplicate knobs are per-run config, identical for every binary in
the dispatch, so they ride the worker argv via
``build_worker_command_args`` (NOT duplicated into N payloads) — the same
precedent ``MemmapBuilderTask`` set for ``--unified-vocab``.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from pathlib import Path

from dynamic_runner import TaskDep
from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId

from tools.batch_smoke._discovery import discover_binaries, filter_binaries

from dynrunner.binary_selection import BinaryIdentifier, TaskInfo


_logger = logging.getLogger(__name__)

# Phase + the two type ids. Both types live in ONE phase so the
# sorted-index task can name its realized-length sibling with a bare
# (intra-phase) ``task_depends_on`` entry — no cross-phase qualifier
# needed (see ``TaskDep.phase_id`` default-empty == enclosing phase).
PHASE_ID = "index"
REALIZED_LENGTHS_TYPE = "realized_lengths"
SORTED_INDEX_TYPE = "sorted_index"

REALIZED_LENGTHS_WORKER = "dynrunner.build_index.realized_lengths_worker"
SORTED_INDEX_WORKER = "dynrunner.build_index.sorted_index_worker"

# Per-binary task-id grammar. The sorted-index task references the
# realized-length task by id, so both ids are derived from the same
# binary_name with a distinct per-type prefix to keep the
# ``(phase_id, task_id)`` identity unique within the phase (the
# framework rejects collisions at pool composition).
_RLEN_TASK_PREFIX = "rlen:"
_SIDX_TASK_PREFIX = "sidx:"


def _rlen_task_id(binary_name: str) -> str:
    return f"{_RLEN_TASK_PREFIX}{binary_name}"


def _sidx_task_id(binary_name: str) -> str:
    return f"{_SIDX_TASK_PREFIX}{binary_name}"


def _opaque_identifier(binary_name: str) -> BinaryIdentifier:
    """Build the opaque ``BinaryIdentifier`` for a binary-name item.

    Index items are keyed only by binary_name; the canonical-4 axes are
    not meaningful at this layer (the memmap dir is already
    variant-merged per binary). The binary_name fills the name slot; the
    axis slots are empty sentinels. Mirrors ``MemmapBuilderTask``'s
    group-level identifier shape minus the representative-axis copy
    (there is no representative variant to read here).
    """
    return BinaryIdentifier(
        binary_name=binary_name,
        platform="",
        compiler="",
        version="",
        opt_level="",
    )


class BuildIndexTask:
    """Phase-4 index build: one phase, two types, one (b)→(a) edge per binary."""

    # See module docstring: ``TaskInfo.path`` is the binary_name, not a
    # real path. Skip the framework's file-based staging for these items.
    uses_file_based_items: bool = False

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        """One phase carrying both index types.

        ``may_be_empty=True`` declares the phase legitimately may drain
        with zero items: under the composite pipeline the index items
        are spawned per-binary as their memmap predecessor completes
        (no eager discovery), so the phase can activate before any item
        exists. Standalone ``--task build-index`` discovery DOES return
        items, so this opt-out is harmless there.
        """
        return (
            PhaseSpec(
                phase_id=PHASE_ID,
                types=(
                    TaskTypeSpec(
                        type_id=REALIZED_LENGTHS_TYPE,
                        worker_module=REALIZED_LENGTHS_WORKER,
                        # Realized-length compute memmaps the arm's
                        # _data.bin and runs one dedup-aware pass; I/O
                        # bound on the largest arm. No inner keepalive
                        # yet, so a generous ceiling.
                        timeout_seconds=300.0,
                    ),
                    TaskTypeSpec(
                        type_id=SORTED_INDEX_TYPE,
                        worker_module=SORTED_INDEX_WORKER,
                        # One shared catalog pre-pass + walk-free length
                        # compute across every (mode, depth); body lengths
                        # come from the realized-length sidecar (the (a)
                        # dependency), so no _data.bin scan at build time.
                        timeout_seconds=300.0,
                    ),
                ),
                may_be_empty=True,
            ),
        )

    # ── Item discovery ─────────────────────────────────────────────────

    def discover_items(
        self, source_dir: Path, args: Namespace
    ) -> Iterable[TaskInfo]:
        """Emit one rlen + one sorted-index item per discovered binary.

        Scans the memmap directory (``source_dir``) with the shared
        ``discover_binaries`` (a binary is recognised by its
        ``<name>_index.bin`` matched-arm sidecar — the memmap builder's
        output). ``--only`` / ``--max-binaries`` narrow the set exactly
        as the two standalone CLIs do. The sorted-index item declares
        ``task_depends_on=(<rlen task_id>,)`` so its dispatch is gated
        on the realized-length sibling per binary.
        """
        discovered = discover_binaries(Path(source_dir))
        selected = filter_binaries(
            discovered,
            only=getattr(args, "only", None),
            max_binaries=getattr(args, "max_binaries", None),
        )
        items: list[TaskInfo] = []
        for binary_name in selected:
            items.extend(self.items_for_binary(binary_name))
        _logger.info(
            "build-index: discovered %d binaries → %d items (%d binaries × 2 types)",
            len(selected),
            len(items),
            len(selected),
        )
        return items

    def items_for_binary(self, binary_name: str) -> list[TaskInfo]:
        """The two phase-4 items for a single binary: rlen + sorted-index.

        Public so the composite pipeline can spawn a single binary's
        index work the moment that binary's memmap build completes
        (per-binary phase-3→4 overlap), reusing the exact same item shape
        as standalone discovery. The sorted-index item depends on the
        realized-length item (same phase, bare ``TaskDep``).
        """
        identifier = _opaque_identifier(binary_name)
        rlen_id = _rlen_task_id(binary_name)
        sidx_id = _sidx_task_id(binary_name)
        rlen_item = TaskInfo(
            path=Path(binary_name),
            size=0,
            identifier=identifier,
            phase_id=PHASE_ID,
            type_id=REALIZED_LENGTHS_TYPE,
            affinity_id=None,
            task_id=rlen_id,
            payload={"binary_name": binary_name},
        )
        sidx_item = TaskInfo(
            path=Path(binary_name),
            size=0,
            identifier=identifier,
            phase_id=PHASE_ID,
            type_id=SORTED_INDEX_TYPE,
            affinity_id=None,
            task_id=sidx_id,
            # (b) consumes (a): the realized-length sidecars must exist
            # before the sorted-index build reads them. Bare TaskDep ⇒
            # same-phase prerequisite; the framework holds this item in
            # ``Blocked`` until ``rlen_id`` terminates (success OR
            # permanent failure — barrier-on-completion, matching the
            # pipeline's per-version skip-on-missing-input contract).
            task_depends_on=(TaskDep(task_id=rlen_id),),
            payload={"binary_name": binary_name},
        )
        return [rlen_item, sidx_item]

    # ── Per-type plumbing ──────────────────────────────────────────────

    def estimate_memory(self, item: TaskInfo) -> int:
        """Per-item RAM. Both passes memmap the binary's ``_data.bin``
        read-only (paged in bounded chunks) and hold only small per-
        variant columns; no fitted model, so a flat 512 MiB floor +
        256 MiB overhead — the same conservative shape MemmapBuilderTask
        uses without the CSV-size multiplier (these items carry size=0,
        the work driver is on-disk and not known at discovery time).
        """
        return 512 * 1024 * 1024 + 256 * 1024 * 1024

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        self.add_private_task_arguments(parser)

    def add_private_task_arguments(self, parser: ArgumentParser) -> None:
        """Task-private argparse registrations.

        Split out (no asm-selection flags) so the composite pipeline can
        register each child's private flags individually after
        registering the shared selection block once — mirrors
        ``MemmapBuilderTask.add_private_task_arguments``.

        ``--only`` / ``--max-binaries`` mirror both standalone CLIs.
        ``--mode`` / ``--depth`` / the gating + duplicate knobs are the
        sorted-index build's per-run config, forwarded verbatim to the
        sorted-index worker. The realized-length worker ignores them.
        """
        parser.add_argument(
            "--only",
            type=str,
            default=None,
            help=(
                "Comma-separated allow-list of binary names. Applied "
                "before --max-binaries."
            ),
        )
        parser.add_argument(
            "--max-binaries",
            type=int,
            default=None,
            help="Cap on number of binaries to process (after --only).",
        )
        parser.add_argument(
            "--mode",
            action="append",
            required=True,
            metavar="MODE",
            help=(
                "Sorted-index reduction mode (repeatable). 'max' or "
                "'p<NN>' (1<=NN<=99). Required: the sorted-index worker "
                "declares it required, so the dispatcher enforces it at "
                "parse time — fail-loud here rather than letting an "
                "omitted flag yield an incomplete worker argv (and zero "
                ".idx). Forwarded verbatim to its worker."
            ),
        )
        parser.add_argument(
            "--depth",
            action="append",
            type=int,
            required=True,
            metavar="DEPTH",
            help=(
                "Sorted-index splice depth (repeatable). Required: see "
                "--mode — the dispatcher enforces the worker's "
                "requirement at parse time. Forwarded verbatim to its "
                "worker."
            ),
        )
        parser.add_argument(
            "--min-variants",
            type=int,
            default=0,
            metavar="N",
            help="Sorted-index top-level minimum-variant emission gate.",
        )
        parser.add_argument(
            "--min-variants-unique",
            type=int,
            default=0,
            metavar="M",
            help="Sorted-index unique-variant minimum (composes with --min-variants).",
        )
        parser.add_argument(
            "--adjust-for-duplicates",
            action="store_true",
            help="Sorted-index: collapse data-pointer-duplicate variants before reduction.",
        )

    def build_worker_command_args(
        self,
        type_id: TypeId,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        """Per-type worker argv.

        The realized-length worker needs no per-run config beyond the
        framework's ``--source`` / ``--output`` (it reads/writes the
        memmap dir). The sorted-index worker needs the
        ``--mode`` / ``--depth`` selection + gating/duplicate knobs;
        they are the same for every binary in the dispatch, so they
        belong on argv (not duplicated into each payload). Forwarded
        verbatim — the worker's argparse owns text→typed conversion.
        """
        if type_id == REALIZED_LENGTHS_TYPE:
            return []
        cmd: list[str] = []
        for mode in getattr(args, "mode", None) or ():
            cmd.extend(["--mode", str(mode)])
        for depth in getattr(args, "depth", None) or ():
            cmd.extend(["--depth", str(depth)])
        if getattr(args, "min_variants", 0):
            cmd.extend(["--min-variants", str(args.min_variants)])
        if getattr(args, "min_variants_unique", 0):
            cmd.extend(["--min-variants-unique", str(args.min_variants_unique)])
        if getattr(args, "adjust_for_duplicates", False):
            cmd.append("--adjust-for-duplicates")
        return cmd

    def get_output_filename_pattern(
        self, type_id: TypeId, item: TaskInfo
    ) -> str:
        """Canonical done-marker per type.

        Realized-length: the matched-arm body ``<binary>_lengths.bin``.
        Sorted-index: there is one ``.idx`` per (mode, depth); the
        framework's ``--skip-existing`` machinery keys on a single
        filename, so this is a representative marker only — the
        sorted-index build is cheap to re-run and idempotent, so a
        coarse marker is acceptable. We return the matched-arm rlen
        marker for the rlen type and the binary-name stem suffix for
        the sorted-index type (informational; not load-bearing for
        per-(mode,depth) skip, which the worker does not implement).
        """
        if type_id == REALIZED_LENGTHS_TYPE:
            return f"{item.binary_name}_lengths.bin"
        return f"{item.binary_name}_sorted.idx"

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
