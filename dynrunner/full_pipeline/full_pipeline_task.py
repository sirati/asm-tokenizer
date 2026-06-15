"""Composite TaskDefinition: the three-phase tokenize→unify→memmap chain.

Single concern: expose the existing ``TokenizerTask`` /
``VocabUnifierTask`` / ``MemmapBuilderTask`` under one
:class:`~dynamic_runner.task_protocol.TaskDefinition` so the framework
manages the entire chain on a persistent secondary mesh (one sbatch
wave for the full pipeline, not three).

The class is pure composition. Every protocol method delegates to one
of the three children, picked by ``phase_id`` / ``type_id``. No child
logic is duplicated here; if a child changes its memory estimator,
output-filename pattern, or worker-argv builder, this composite
inherits the change automatically through the dispatch table.

Cross-phase routing — "phase 2/3 read the tokenize output; phase 3
writes into a memmap/ subdir" — is centralised in
``phase_routing.route_for_phase``. The composite consults the route
at three boundary points:

* ``discover_items`` for phase 1 (the framework's upfront discovery).
* ``on_phase_end`` for phases 2 and 3, where the composite runs the
  child's ``discover_items`` against the *output* of the just-
  finished phase and injects the items via
  ``primary_handle.spawn_tasks(...)``. Lazy discovery is required:
  those children walk the output tree, which doesn't exist at run
  start.
* ``build_worker_command_args`` to override the worker's ``--source``
  and ``--output`` argv so workers (which still parse standalone-mode
  flags) write to the per-phase destination.

The lazy ``on_phase_end`` + ``spawn_tasks`` path depends on the
framework firing ``on_phase_end`` on the pool-owning coordinator and
exposing a ``PrimaryHandle`` whose ``spawn_tasks`` reaches that pool.
The distributed manager's documented contract supports this (task
protocol docstring; ``run_distributed`` passes a live handle to the
modern ``on_run_start`` signature) — see the parallel framework-side
work for the post-promotion firing wiring.
"""

from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from dataclasses import replace as _replace_dataclass
from pathlib import Path
from typing import Any, Optional

from dynamic_runner.task_protocol import PhaseSpec, TaskTypeSpec, TypeId

from dynrunner.binary_selection import TaskInfo, add_asm_selection_arguments
from dynrunner.build_index.build_index_task import BuildIndexTask
from dynrunner.build_memmap.memmap_builder_task import MemmapBuilderTask
from dynrunner.tokenize.tokenizer_task import TokenizerTask
from dynrunner.unify_vocab.vocab_unifier_task import VocabUnifierTask

from .phase_routing import (
    BUILD_INDEX_PHASE,
    BUILD_MEMMAP_PHASE,
    TOKENIZE_PHASE,
    UNIFY_VOCAB_PHASE,
    PhaseRoute,
    phase_for_type,
    route_for_phase,
)


_logger = logging.getLogger(__name__)


def _route_args(args: Namespace, route: PhaseRoute) -> Namespace:
    """Clone ``args`` with ``source`` + ``resolved_output_root`` overridden
    to the per-phase route.

    Each child's ``discover_items`` reads
    ``args.source`` (via ``process_selection_arguments``) and the
    ``args.resolved_output_root`` set by the framework before dispatch;
    the positional ``source_dir`` parameter is informational. The lazy-
    discovery path needs to make both fields agree with the routed
    location BEFORE the child reads them, mirroring what the legacy
    ``_chain_for_phase`` rewrite did at the CLI level. Cloning the
    namespace keeps the override scoped to the discovery call — the
    original ``args`` the framework holds is untouched.
    """
    routed = Namespace(**vars(args))
    routed.source = str(route.source_dir)
    routed.resolved_output_root = str(route.output_dir)
    # `_route_args` is only used to materialise the DERIVED downstream
    # phases (unify-vocab, build-memmap) in `_spawn_phase_items`. Those
    # are aggregate artifacts of the WHOLE tokenize output, not per-
    # binary-idempotent work — so they must always regenerate after a
    # fresh tokenize phase. Inheriting the user's `--skip-existing`
    # (meant for the per-binary tokenize step) would make them short-
    # circuit on their own done-marker (`unified_vocab.csv` /
    # already-built memmaps) and silently leave newly-tokenized binaries
    # out of the vocab + reference a stale vocab from the old memmaps.
    # Phase 1 (tokenize) still honours skip-existing via the direct
    # `discover_items` path; only the routed phase-2/3 clone clears it.
    routed.skip_existing = False
    return routed


# The phases chained via the lazy ``on_phase_end`` → ``spawn_tasks``
# discovery path: each one's items are walked off the PRIOR phase's
# on-disk output, so they materialise only after that prior phase
# drains. Phase 4 (``index``) is deliberately NOT in this tuple — it is
# spawned PER BINARY off each memmap task's completion (see
# ``task_completed_listener``), not in bulk at the end of phase 3, so
# index work for an already-built binary overlaps phase 3's remaining
# builds.
_PHASE_ORDER: tuple[str, ...] = (
    TOKENIZE_PHASE,
    UNIFY_VOCAB_PHASE,
    BUILD_MEMMAP_PHASE,
)


class FullPipelineTask:
    """Four-phase composite: tokenize → unify-vocab → build-memmap →
    index (realized-lengths + sorted-index).

    Instantiates each child task once and dispatches every protocol
    method to the right one. The framework drives phases 1-3 via
    ``PhaseSpec.depends_on``; phase 2 and 3 items materialise lazily
    through ``primary_handle.spawn_tasks`` from inside the composite's
    ``on_phase_end`` hook so children whose discovery walks the output
    tree see their inputs on disk.

    Phase 4 (``index``) carries NO ``depends_on`` the memmap phase — that
    phase-level barrier is exactly what we must avoid (requirement: no
    all-of-phase-3 barrier). Instead the composite registers a
    ``task_completed_listener``; when an individual memmap task
    completes, the listener spawns THAT binary's two index items
    (realized-length + sorted-index, the latter depending on the former
    per binary). So binary X's index build starts the moment X's memmap
    build finishes, while other binaries' memmap builds continue. The
    ``index`` PhaseSpec is ``may_be_empty=True`` because its items arrive
    via late per-binary spawn, never via run-start discovery.

    Why a per-task listener and not ``PhaseSpec.barrier=False``: in
    dynamic_runner 0.4.0 the ``barrier`` field is documented "reserved
    for future pipelined work and is not used today", and the PyO3
    ``get_phases`` extractor reads only ``depends_on`` / ``may_be_empty``
    / ``types`` off each PhaseSpec — ``barrier`` is never carried across
    the boundary, so it cannot relax the gate. Per-binary overlap is
    therefore expressed through the per-task completion path that the
    framework DOES wire (``task_completed_listener`` +
    ``PrimaryHandle.spawn_tasks``), with the per-binary (b)→(a) edge
    carried on ``TaskInfo.task_depends_on``.
    """

    # `TaskInfo.path` for unify_vocab and build_memmap items is an
    # opaque sentinel, not a real filesystem path. The composite must
    # opt out of the framework's file-based staging machinery for those
    # phases — the underlying VocabUnifierTask / MemmapBuilderTask both
    # set this. Tokenize items are real files; the framework's per-item
    # staging rule still applies to them because the discovery returns
    # real Path objects (the flag's effect is per-item via the wire
    # protocol, not a hard gate on the task class).
    uses_file_based_items: bool = False

    def __init__(self) -> None:
        self._tokenize = TokenizerTask()
        self._unify_vocab = VocabUnifierTask()
        self._memmap = MemmapBuilderTask()
        self._index = BuildIndexTask()
        self._child_by_phase = {
            TOKENIZE_PHASE: self._tokenize,
            UNIFY_VOCAB_PHASE: self._unify_vocab,
            BUILD_MEMMAP_PHASE: self._memmap,
            BUILD_INDEX_PHASE: self._index,
        }
        # Captured from on_run_start. The composite needs all of them to
        # drive lazy phase 2 + 3 discovery from on_phase_end and the
        # per-binary phase-4 spawn from the completion listener.
        self._primary_handle: Optional[Any] = None
        self._user_source: Optional[Path] = None
        self._user_output: Optional[Path] = None
        self._args: Optional[Namespace] = None
        # Task-ids of the memmap items the composite spawned for phase 3.
        # The completion listener distinguishes "a memmap task finished"
        # (→ spawn that binary's index work) from any other terminal by
        # set membership — no string-shape inference on the task_id.
        self._memmap_task_ids: set[str] = set()
        # Binaries whose index items have already been spawned, so a
        # duplicate completion signal (retry, failover replay) never
        # double-spawns. spawn_tasks itself dedups by content hash, but
        # this keeps the composite's own logging honest.
        self._index_spawned_binaries: set[str] = set()

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        """Build the four phase specs.

        Phases 1-3 are a linear ``depends_on`` chain over
        ``_PHASE_ORDER``. We replace each child's ``PhaseSpec.depends_on``
        to wire the chain without touching the child's own definition
        (the child still declares no deps because it's authoritative for
        standalone single-phase dispatch).

        Phase 4 (``index``) is appended with NO ``depends_on``: it must
        not barrier behind full-phase-3 drain (requirement: no
        all-of-phase-3 barrier). Its items arrive per binary via the
        completion listener, so the child's own ``may_be_empty=True`` is
        preserved verbatim.
        """
        phases: list[PhaseSpec] = []
        prior_phase: Optional[str] = None
        for phase_id in _PHASE_ORDER:
            child_phases = self._child_by_phase[phase_id].get_phases()
            assert len(child_phases) == 1, (
                f"FullPipelineTask expects each chained child to declare one "
                f"phase; {phase_id} declared {len(child_phases)}"
            )
            child_spec = child_phases[0]
            depends_on = (prior_phase,) if prior_phase is not None else ()
            phases.append(
                _replace_dataclass(child_spec, depends_on=depends_on)
            )
            prior_phase = phase_id

        # Phase 4: independent of the chain barrier (no depends_on). The
        # child declares one phase with its two types and may_be_empty;
        # we take it verbatim so the per-binary spawn path drives it.
        index_phases = self._index.get_phases()
        assert len(index_phases) == 1, (
            f"FullPipelineTask expects the index child to declare one phase; "
            f"declared {len(index_phases)}"
        )
        phases.append(index_phases[0])
        return tuple(phases)

    # ── Item discovery ─────────────────────────────────────────────────

    def discover_items(
        self, source_dir: Path, args: Namespace
    ) -> Iterable[TaskInfo]:
        """Return phase-1 items only.

        Phases 2 and 3 walk the OUTPUT tree which doesn't yet exist at
        run start; we defer their discovery to ``on_phase_end`` and
        inject via ``primary_handle.spawn_tasks``. Each child's
        existing ``discover_items`` is the single authoritative source
        of its own items — the composite never reaches inside a
        child's payload schema.
        """
        return self._tokenize.discover_items(source_dir, args)

    # ── Per-type plumbing ──────────────────────────────────────────────

    def estimate_memory(self, item: TaskInfo) -> int:
        return self._child_for_type(item.type_id).estimate_memory(item)

    def add_task_arguments(self, parser: ArgumentParser) -> None:
        """Register the asm-selection flag block once + each child's
        private flags.

        Each child task's standalone ``add_task_arguments`` calls
        ``add_asm_selection_arguments`` itself; calling all three here
        would re-register every selection flag and trip an argparse
        conflict. The split into ``add_private_task_arguments`` lets
        the composite register the shared block once and the per-child
        private flags individually.
        """
        add_asm_selection_arguments(parser)
        for child in (self._tokenize, self._unify_vocab, self._memmap, self._index):
            child.add_private_task_arguments(parser)

    def build_worker_command_args(
        self,
        type_id: TypeId,
        args: Namespace,
        source_dir: Path,
        output_dir: Path,
        skip_existing: bool,
    ) -> list[str]:
        """Delegate to the child + append per-phase ``--source``/``--output``
        overrides so workers see their phase-specific roots.

        The framework already emits ``--source <source_dir> --output
        <output_dir>`` at the front of the worker argv. argparse's
        last-occurrence semantics means appending another pair at the
        end overrides the framework's default cleanly without any
        worker-side change.

        The framework calls this method BEFORE ``on_run_start``
        (worker argv templates are built during manager construction),
        so the per-phase route is derived from the ``source_dir`` /
        ``output_dir`` parameters supplied here — they're the
        user-supplied roots in their per-call form.
        """
        phase_id = phase_for_type(type_id)
        child = self._child_by_phase[phase_id]
        child_args = child.build_worker_command_args(
            type_id, args, source_dir, output_dir, skip_existing
        )
        # The framework passes ``source_dir`` / ``output_dir`` as raw
        # strings on this call site (PyO3 boundary); coerce to Path so
        # the routing helpers can do path arithmetic.
        route = route_for_phase(phase_id, Path(source_dir), Path(output_dir))
        override = [
            "--source",
            str(route.source_dir),
            "--output",
            str(route.output_dir),
        ]
        return [*child_args, *override]

    def get_output_filename_pattern(
        self, type_id: TypeId, item: TaskInfo | None
    ) -> str:
        return self._child_for_type(type_id).get_output_filename_pattern(
            type_id, item
        )

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def on_run_start(
        self,
        source_dir: Path,
        output_dir: Path,
        args: Namespace,
        primary_handle: Optional[Any] = None,
    ) -> None:
        """Capture roots + ``primary_handle`` for lazy phase 2/3
        discovery, then forward to every child.

        ``primary_handle`` is the framework's modern kwarg supplying
        the runtime control surface for the pool-owning coordinator;
        we hold it for ``on_phase_end`` to call ``spawn_tasks``. The
        legacy positional-only signature on every child task means
        each child's forward stays kwarg-free.
        """
        self._user_source = Path(source_dir)
        self._user_output = Path(output_dir)
        self._args = args
        self._primary_handle = primary_handle
        for child in (self._tokenize, self._unify_vocab, self._memmap, self._index):
            child.on_run_start(source_dir, output_dir, args)

    def on_run_end(self, success: bool) -> None:
        for child in (self._tokenize, self._unify_vocab, self._memmap, self._index):
            child.on_run_end(success)

    def on_phase_start(self, phase_id: str) -> None:
        self._child_by_phase[phase_id].on_phase_start(phase_id)

    def on_phase_end(
        self, phase_id: str, completed: int, failed: int
    ) -> None:
        """Forward to the just-finished child, then trigger lazy
        discovery + spawn for the *next* phase.

        Fires AFTER the just-finished phase's items have all
        terminated; the framework guarantees the dep-graph schedule
        won't dispatch the next phase until this method returns. We
        run the next child's ``discover_items`` against the routed
        source root and ``spawn_tasks`` the result into the live
        cluster.
        """
        self._child_by_phase[phase_id].on_phase_end(phase_id, completed, failed)

        next_phase = self._next_phase(phase_id)
        if next_phase is None:
            return
        self._spawn_phase_items(next_phase)

    def task_completed_listener(
        self,
        task_id: Optional[str],
        success: bool,
        error_kind: Optional[str],
        last_error: Optional[str],
    ) -> None:
        """Per-binary phase-3 → phase-4 hand-off.

        Fires once per terminal task transition (success or failure),
        off the CRDT apply path. When the terminal is a MEMMAP task (by
        membership in ``self._memmap_task_ids`` — the composite recorded
        them when it spawned phase 3, so no task_id string-shape
        inference is needed), the listener spawns THAT binary's two index
        items via ``primary_handle.spawn_tasks``. The result: binary X's
        index build starts the moment X's memmap build terminates, with
        no wait for the rest of phase 3.

        Spawn on COMPLETION, not only on success: a memmap task that
        failed for one binary still gets its index work attempted; the
        index workers surface their own missing-input miss as
        NonRecoverable, matching the pipeline's barrier-on-completion
        (not barrier-on-success) contract. The memmap binary_name IS the
        task_id (``MemmapBuilderTask`` sets ``task_id=binary_name``), so
        the binary name is the task_id verbatim.
        """
        if task_id is None or task_id not in self._memmap_task_ids:
            return
        binary_name = task_id
        if binary_name in self._index_spawned_binaries:
            return
        self._index_spawned_binaries.add(binary_name)

        if self._primary_handle is None:
            _logger.error(
                "FullPipelineTask: index spawn for binary %s skipped — "
                "primary_handle is None in task_completed_listener. "
                "Per-binary phase-4 hand-off requires the framework to "
                "fire on_run_start with a live PrimaryHandle on the "
                "pool-owning coordinator.",
                binary_name,
            )
            return

        items = self._index.items_for_binary(binary_name)
        _logger.info(
            "FullPipelineTask: memmap %s done → spawning %d index item(s).",
            binary_name,
            len(items),
        )
        errors = self._primary_handle.spawn_tasks(items)
        if errors:
            for idx, err in errors:
                _logger.warning(
                    "FullPipelineTask: spawn_tasks rejected index item for "
                    "binary %s (index %d): %r",
                    binary_name,
                    idx,
                    err,
                )

    # ── Internals ──────────────────────────────────────────────────────

    def _child_for_type(self, type_id: str):
        return self._child_by_phase[phase_for_type(type_id)]

    @staticmethod
    def _next_phase(phase_id: str) -> Optional[str]:
        idx = _PHASE_ORDER.index(phase_id)
        if idx + 1 >= len(_PHASE_ORDER):
            return None
        return _PHASE_ORDER[idx + 1]

    def _spawn_phase_items(self, phase_id: str) -> None:
        """Discover ``phase_id`` items against its routed source and
        inject them into the running cluster via ``primary_handle.spawn_tasks``.

        Logs and aborts when ``primary_handle`` is ``None`` — the
        modern signature is the contract the composite depends on,
        and the absence indicates a framework-side gap that needs
        addressing before this composite can run end-to-end. The
        ``_logger.error`` makes the gap visible at runtime so the
        downstream phases' empty output isn't silently passed off as
        success.
        """
        if self._primary_handle is None:
            _logger.error(
                "FullPipelineTask: phase %s discovery skipped — "
                "primary_handle is None at on_phase_end. Lazy phase "
                "chaining requires the framework to fire on_phase_end "
                "on the pool-owning coordinator with a live "
                "PrimaryHandle; downstream phases will not run.",
                phase_id,
            )
            return

        assert self._user_source is not None and self._user_output is not None, (
            "FullPipelineTask: lazy phase discovery requires on_run_start to "
            "have captured the user-supplied source/output roots."
        )
        child = self._child_by_phase[phase_id]
        route = route_for_phase(phase_id, self._user_source, self._user_output)
        # Each child's ``discover_items`` reads ``args.source`` (via
        # ``process_selection_arguments``) and ``args.resolved_output_root``
        # rather than the positional ``source_dir`` parameter — the same
        # behaviour the standalone CLI relies on. Build a routed
        # Namespace clone for the child so its in-walk semantics match
        # the routing the framework's worker argv carries.
        routed_args = _route_args(self._args, route)
        items = list(child.discover_items(route.source_dir, routed_args))
        _logger.info(
            "FullPipelineTask: phase %s discovered %d item(s); injecting via "
            "spawn_tasks.",
            phase_id,
            len(items),
        )
        # Record the memmap items' task-ids so the completion listener can
        # recognise a memmap-task terminal by set membership (not by
        # parsing the task_id string) and spawn that binary's index work.
        if phase_id == BUILD_MEMMAP_PHASE:
            self._memmap_task_ids.update(
                item.task_id for item in items if item.task_id
            )
        if not items:
            return
        errors = self._primary_handle.spawn_tasks(items)
        if errors:
            for idx, err in errors:
                _logger.warning(
                    "FullPipelineTask: spawn_tasks rejected phase %s item "
                    "index %d: %r",
                    phase_id,
                    idx,
                    err,
                )
