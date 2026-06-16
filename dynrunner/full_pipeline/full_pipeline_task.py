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
from dynrunner.build_memmap._publish_scope import memmap_dir_for
from dynrunner.build_memmap.memmap_builder_task import MemmapBuilderTask
from dynrunner.tokenize.tokenizer_task import TokenizerTask
from dynrunner.unify_vocab.vocab_unifier_task import VocabUnifierTask

from .phase_routing import (
    BUILD_INDEX_PHASE,
    BUILD_MEMMAP_PHASE,
    TOKENIZE_PHASE,
    UNIFY_VOCAB_PHASE,
    PhaseRoute,
    decorate_index_items_with_memmap_dep,
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


# The phase chain. Each phase's downstream items materialise off the
# PRIOR phase's on-disk output, so the chain is wired with a linear
# ``depends_on`` edge in ``get_phases``. Phase 4 (``index``) IS in this
# tuple so it flows through the same chain-wiring loop — but its edge is
# RELAXED to a pipelined one (``barrier=False``) so a binary's index
# build can start the moment THAT binary's memmap build finishes, while
# the rest of phase 3 is still draining. The per-binary overlap is
# realised by co-spawning each binary's index items in the SAME
# ``spawn_tasks`` batch as its memmap item (see ``_spawn_phase_items``),
# each index item carrying a cross-phase ``task_depends_on`` edge to its
# memmap predecessor.
_PHASE_ORDER: tuple[str, ...] = (
    TOKENIZE_PHASE,
    UNIFY_VOCAB_PHASE,
    BUILD_MEMMAP_PHASE,
    BUILD_INDEX_PHASE,
)

# The subset of phases whose downstream edge is a strict whole-of-
# upstream barrier (``barrier=True``, the framework default). The
# pipelined-edge phase (``index``) is excluded so its ``PhaseSpec`` is
# decorated with ``barrier=False`` in ``get_phases``.
_PIPELINED_EDGE_PHASES: frozenset[str] = frozenset({BUILD_INDEX_PHASE})

# The phases whose items materialise via lazy ``on_phase_end`` →
# ``spawn_tasks`` discovery off the prior phase's on-disk output, in
# chain order. ``index`` is NOT here: it is not discovered off disk but
# CO-SPAWNED per binary alongside its memmap predecessor inside the
# ``memmap`` batch (``_spawn_phase_items(memmap)``), so the cross-phase
# ``rlen → memmap`` dependency validates against the same batch's
# known-set. ``on_phase_end`` therefore advances only across this tuple.
_LAZY_DISCOVERY_ORDER: tuple[str, ...] = (
    TOKENIZE_PHASE,
    UNIFY_VOCAB_PHASE,
    BUILD_MEMMAP_PHASE,
)


class FullPipelineTask:
    """Four-phase composite: tokenize → unify-vocab → build-memmap →
    index (realized-lengths + sorted-index).

    Cited against dynamic-runner trunk @ a19ac75b (#540 + #584 + #588 +
    #590): ``PhaseSpec.barrier`` IS carried across the PyO3 boundary and
    enforced at both the scheduler phase-state gate and the runtime
    ``spawn_tasks`` interlock; #590 folds ``task_id`` + ``type_id`` into
    the wire-canonical task hash so a binary's rlen + sidx items hash
    distinctly (no dedup-drop).

    Instantiates each child task once and dispatches every protocol
    method to the right one. The framework drives phases 1-3 via
    ``PhaseSpec.depends_on``; phase 2 and 3 items materialise lazily
    through ``primary_handle.spawn_tasks`` from inside the composite's
    ``on_phase_end`` hook so children whose discovery walks the output
    tree see their inputs on disk.

    Phase 4 (``index``) declares ``depends_on=("memmap",)`` but RELAXES
    that edge to a pipelined one with ``barrier=False`` — the framework
    starts the phase ``Active`` from run start, and each index item
    dispatches as soon as its own per-task ``task_depends_on`` resolves,
    not when all of phase 3 drains. The per-binary overlap is realised by
    CO-SPAWNING each binary's two index items (realized-length +
    sorted-index) in the SAME ``spawn_tasks`` batch as that binary's
    memmap item, from inside ``_spawn_phase_items(memmap)`` (fired off
    ``on_phase_end(unify_vocab)``). Each rlen item carries a CROSS-PHASE
    ``task_depends_on=(TaskDep(task_id=binary_name, phase_id="memmap"),)``
    edge to its memmap predecessor (memmap ``task_id == binary_name``);
    the existing intra-phase sidx→rlen edge stays. So per binary X:
    memmap(X) → rlen(X) → sidx(X), gated per-task even under
    ``barrier=False``, with X's index build starting the moment X's
    memmap build finishes while other binaries' memmap builds continue.

    Co-spawning in the SAME batch is what validates the cross-phase
    dependency: the batch's own known-set contains the memmap item the
    rlen item names, so the framework's spawn-time validator sees the
    predecessor and never raises ``UnknownDependency``. It also keeps the
    index phase non-empty from injection time, avoiding the empty-drain
    edge — ``may_be_empty=True`` is preserved as the honest declaration
    that the phase has no RUN-START-discovered items.

    The cross-phase-dep decoration lives HERE in the composite (the
    pipeline owns the cross-phase knowledge); the standalone
    ``BuildIndexTask`` child stays dep-free across phases so it remains
    usable for standalone ``--task build-index`` dispatch.
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
        # drive lazy phase 2 + 3 discovery from on_phase_end (phase 3's
        # batch also co-spawns the phase-4 index items).
        self._primary_handle: Optional[Any] = None
        self._user_source: Optional[Path] = None
        self._user_output: Optional[Path] = None
        self._args: Optional[Namespace] = None

    # ── Topology ───────────────────────────────────────────────────────

    def get_phases(self) -> tuple[PhaseSpec, ...]:
        """Build the four phase specs as one chain over ``_PHASE_ORDER``.

        Every phase gets a linear ``depends_on`` edge to its predecessor
        by replacing the child's ``PhaseSpec.depends_on`` — the child
        itself still declares no deps because it's authoritative for
        standalone single-phase dispatch. Phase 4 (``index``) is wired
        through the SAME loop (it is in ``_PHASE_ORDER``), so it inherits
        ``depends_on=("memmap",)``; its edge is then RELAXED to a
        pipelined one with ``barrier=False`` (membership in
        ``_PIPELINED_EDGE_PHASES``). The framework starts a
        ``barrier=False`` phase ``Active`` from run start and gates each
        item on its own ``task_depends_on`` rather than on full upstream
        drain, which is what lets a binary's index build overlap the rest
        of phase 3. The child's ``may_be_empty=True`` is preserved (the
        index items arrive via the phase-3 co-spawn batch, not run-start
        discovery).
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
            barrier = phase_id not in _PIPELINED_EDGE_PHASES
            phases.append(
                _replace_dataclass(
                    child_spec, depends_on=depends_on, barrier=barrier
                )
            )
            prior_phase = phase_id
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

    # ── Internals ──────────────────────────────────────────────────────

    def _child_for_type(self, type_id: str):
        return self._child_by_phase[phase_for_type(type_id)]

    @staticmethod
    def _next_phase(phase_id: str) -> Optional[str]:
        """The next lazily-discovered phase after ``phase_id``, or None.

        Ranges over ``_LAZY_DISCOVERY_ORDER`` (tokenize → unify_vocab →
        memmap), NOT ``_PHASE_ORDER``: the ``index`` phase is co-spawned
        inside the memmap batch rather than discovered off disk, so
        ``on_phase_end(memmap)`` has no lazy next phase to discover.
        """
        if phase_id not in _LAZY_DISCOVERY_ORDER:
            return None
        idx = _LAZY_DISCOVERY_ORDER.index(phase_id)
        if idx + 1 >= len(_LAZY_DISCOVERY_ORDER):
            return None
        return _LAZY_DISCOVERY_ORDER[idx + 1]

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
        # Co-spawn each binary's phase-4 index items in the SAME batch as
        # the memmap items. Doing it here (not as a separate
        # ``_spawn_phase_items(index)`` call) is what makes the cross-
        # phase ``rlen → memmap`` dependency validate: the memmap
        # predecessor the rlen item names is present in this very batch's
        # known-set, so the framework's spawn-time dep validator never
        # raises ``UnknownDependency``. It also keeps the ``barrier=False``
        # index phase non-empty from injection time, so it does not hit
        # the empty-drain edge before any item arrives.
        index_items = (
            self._build_index_items_for_memmap(items)
            if phase_id == BUILD_MEMMAP_PHASE
            else []
        )
        batch = [*items, *index_items]
        _logger.info(
            "FullPipelineTask: phase %s discovered %d item(s)%s; injecting via "
            "spawn_tasks.",
            phase_id,
            len(items),
            f" + {len(index_items)} co-spawned index item(s)" if index_items else "",
        )
        if not batch:
            return
        errors = self._primary_handle.spawn_tasks(batch)
        if errors:
            for idx, err in errors:
                _logger.warning(
                    "FullPipelineTask: spawn_tasks rejected phase %s batch "
                    "item index %d: %r",
                    phase_id,
                    idx,
                    err,
                )

    def _build_index_items_for_memmap(
        self, memmap_items: list[TaskInfo]
    ) -> list[TaskInfo]:
        """Build the decorated phase-4 index items for every memmap item.

        For each memmap item (``task_id == binary_name``) the index child
        emits that binary's rlen + sorted-index items; the composite then
        decorates them with the cross-phase ``rlen → memmap`` edge. The
        child stays cross-phase-dep-free (standalone-pure); the edge is
        added only on this composite boundary.

        Each binary's ``memmap_dir`` is resolved through the shared
        ``memmap_dir_for`` (the inverse of the memmap worker's
        per-binary publish scope) against the memmap phase's output root,
        so the index worker reads the sidecars from the same
        ``build_memmap/<name>/`` subdir the memmap phase published them to
        (nested under container deployment, flat standalone).
        """
        assert self._user_source is not None and self._user_output is not None, (
            "FullPipelineTask: index co-spawn requires on_run_start to have "
            "captured the user-supplied source/output roots."
        )
        memmap_root = route_for_phase(
            BUILD_MEMMAP_PHASE, self._user_source, self._user_output
        ).output_dir
        index_items: list[TaskInfo] = []
        for memmap_item in memmap_items:
            binary_name = memmap_item.task_id
            if not binary_name:
                continue
            memmap_dir = memmap_dir_for(memmap_root, binary_name)
            binary_index_items = self._index.items_for_binary(
                binary_name, memmap_dir
            )
            index_items.extend(
                decorate_index_items_with_memmap_dep(
                    binary_index_items, binary_name
                )
            )
        return index_items
