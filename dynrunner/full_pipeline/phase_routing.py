"""Per-phase source/output routing for the composite pipeline.

Single concern: given the user-supplied ``--source`` / ``--output``
roots, compute the per-phase ``(source_dir, output_dir)`` pair each
child task expects. This mirrors the consumer-side ``_chain_for_phase``
semantics that ``dynrunner.__main__`` used to apply between three
independent dispatches, but expressed as a pure data mapping consumed
inside one composite TaskDefinition.

Why a separate module: the routing rule ("phase 2/3 read the tokenize
output; phase 3 writes into a ``memmap/`` subdir") is the single piece
of cross-phase knowledge the composite needs. Keeping it isolated lets
the rest of ``full_pipeline`` stay agnostic of which subtree each child
walks — every protocol method delegates to the child paired with a
``PhaseRoute`` returned from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TOKENIZE_PHASE = "tokenize"
UNIFY_VOCAB_PHASE = "unify_vocab"
BUILD_MEMMAP_PHASE = "memmap"

TOKENIZE_TYPE = "tokenizer"
UNIFY_VOCAB_TYPE = "unify_vocab"
BUILD_MEMMAP_TYPE = "memmap"


@dataclass(frozen=True)
class PhaseRoute:
    """Where a phase's child task should read from and write to.

    ``source_dir`` and ``output_dir`` are the per-phase analogues of
    the framework's per-run ``--source`` / ``--output``. The composite
    feeds ``source_dir`` to the child's ``discover_items`` and emits
    overriding ``--source``/``--output`` argv via
    ``build_worker_command_args`` so the workers (which still parse
    those flags as standalone-mode argv) write to the right place.

    Last-occurrence semantics on argparse: when the composite appends
    a second ``--source``/``--output`` to the worker argv after the
    framework's own pair, Python's argparse keeps the LAST value.
    Confirmed against ``argparse.ArgumentParser`` default behaviour
    for non-append store actions.
    """

    source_dir: Path
    output_dir: Path


def _route_tokenize(user_source: Path, user_output: Path) -> PhaseRoute:
    return PhaseRoute(source_dir=user_source, output_dir=user_output)


def _route_unify_vocab(user_source: Path, user_output: Path) -> PhaseRoute:
    # Phase 2 reads the per-binary CSVs the tokenize phase wrote.
    # Mappings land in the same tree so build_memmap pairs csv↔mapping
    # by relative subdir.
    return PhaseRoute(source_dir=user_output, output_dir=user_output)


def _route_build_memmap(user_source: Path, user_output: Path) -> PhaseRoute:
    # Phase 3 reads the same tree the tokenize+unify phases wrote, but
    # publishes flat-named memmap artefacts into a sibling ``memmap/``
    # subdir to keep them separate from the per-binary CSV layout.
    return PhaseRoute(source_dir=user_output, output_dir=user_output / "memmap")


_PHASE_ROUTES = {
    TOKENIZE_PHASE: _route_tokenize,
    UNIFY_VOCAB_PHASE: _route_unify_vocab,
    BUILD_MEMMAP_PHASE: _route_build_memmap,
}

_TYPE_TO_PHASE = {
    TOKENIZE_TYPE: TOKENIZE_PHASE,
    UNIFY_VOCAB_TYPE: UNIFY_VOCAB_PHASE,
    BUILD_MEMMAP_TYPE: BUILD_MEMMAP_PHASE,
}


def route_for_phase(
    phase_id: str, user_source: Path, user_output: Path
) -> PhaseRoute:
    """Return the per-phase ``(source, output)`` pair for ``phase_id``.

    Raises ``KeyError`` for unknown ``phase_id``; the caller (the
    composite TaskDefinition) constructs every phase from a fixed
    list, so an unknown id is a contract bug, not a runtime concern.
    """
    return _PHASE_ROUTES[phase_id](user_source, user_output)


def phase_for_type(type_id: str) -> str:
    """Reverse-lookup: which phase owns ``type_id``.

    The framework's per-type protocol methods
    (``build_worker_command_args``, ``estimate_memory``,
    ``get_output_filename_pattern``) only get the ``type_id``; we
    need the phase to look up its route.
    """
    return _TYPE_TO_PHASE[type_id]
