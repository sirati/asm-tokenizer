"""Length-bucketed batch helper + session-agnostic decode core (plan D7).

Two single-concern functions:

* :func:`decode_pointer_batch` -- the session-agnostic core. Given a
  mapping of ALREADY-OPEN sessions plus a flat list of
  :class:`MultiBinarySectionPointer` rows, it groups by binary, runs
  the one-collector-per-batch_load decode, and concatenates the
  per-binary results. It does NOT open or close sessions.
* :func:`open_length_bucketed_batch` -- the session-lifecycle wrapper.
  It samples ``batch_size`` pointers via a
  :class:`MultiBinarySortedIndexSampler` (optionally length-banded),
  opens one :class:`BinarySession` per sampled binary via
  ``session_factory``, and delegates the decode to
  :func:`decode_pointer_batch`.

One-collector-per-batch_load contract:

The core drives a SINGLE :class:`BucketedRunLengthCollector` across
every per-binary Stage 1 walk in the batch_load. Per-binary
``batch_decode`` calls run in the deferred dispatch shape
(``collector`` provided): each call returns a
:class:`PendingBatchDecode` whose Stage 1 has been staged but not
flushed. After every binary's Stage 1 has been staged, the core
flushes the collector ONCE -- that single 2D pow2-bucketed
``run_lengths`` dispatch amortises across every call_target row in
the whole batch_load. Each pending decode is then finalised, running
its own Stages 2-4 against the now-finalised :class:`Stage1Batch`.

Every per-binary session must stay open through both the staging phase
AND the finalise phase (because Stages 2-4 read numpy views that may be
memmap-backed by the session). :func:`decode_pointer_batch` therefore
reads sessions from the caller-supplied mapping but never manages their
lifetime; :func:`open_length_bucketed_batch` opens them under a
:class:`contextlib.ExitStack` that spans the whole core call.

Binary ordering: per-binary results are concatenated in alphabetical
``binary_name`` order (the same order
:attr:`MultiBinarySortedIndexSampler.binary_names` exposes). This
determines the ``binary_id_per_row`` numbering and is stable across
runs.

Imports from ``loader/batch_decode`` here are the typed handoff
classes plus the public :func:`batch_decode` entry; we do NOT touch
any internal stage helpers.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import (
    Callable,
    ContextManager,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._entry import (
    PendingBatchDecode,
    batch_decode,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.session import BinarySession

from .._types import MultiBinaryBatchDecodeResult, MultiBinarySectionPointer
from ._concat import _concat_results
from ._sample import MultiBinarySortedIndexSampler


__all__ = [
    "decode_pointer_batch",
    "open_length_bucketed_batch",
]


def decode_pointer_batch(
    sessions: Mapping[str, BinarySession],
    pointers: List[MultiBinarySectionPointer],
    *,
    context_len: int,
    num_variants_per_section: int,
    max_depth: int,
    rng: np.random.Generator,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    inlined_equivalent_call_targets_only: bool = True,
    include_fid_sidecar: bool = False,
) -> MultiBinaryBatchDecodeResult:
    """Session-agnostic core: decode a flat pointer batch + concat (plan D7).

    Groups ``pointers`` by ``binary_name``, iterates the binaries in
    ``sorted(name)`` order, runs :func:`batch_decode` over each group on
    the matching ALREADY-OPEN session looked up in ``sessions``, drives a
    single shared :class:`BucketedRunLengthCollector` with ONE flush for
    the whole batch, finalises every pending decode, and concatenates the
    per-binary results via :func:`_concat_results`.

    This function owns NO session lifetime: it neither opens nor closes
    sessions. Every binary referenced by a pointer MUST have an open
    session in ``sessions`` that stays open for the duration of the call
    (the finalise phase reads session-backed numpy views).

    Binary ordering: ``sorted(name)`` matches the alphabetical order
    :attr:`MultiBinarySortedIndexSampler.binary_names` exposes, so the
    resulting ``binary_id_per_row`` numbering is stable across runs.

    Raises
    ------
    ValueError
        When ``pointers`` is empty (no work to decode).
    ValueError
        When a pointer names a binary absent from ``sessions`` (a
        missing session is a hard caller error, not a skip).
    """
    if not pointers:
        raise ValueError(
            "decode_pointer_batch: empty pointer batch",
        )

    # Group section pointers by binary_name. Only binaries that received
    # a pointer appear here, so no empty groups are ever decoded.
    per_binary_pointers: Dict[str, List[SectionPointerSpec]] = {}
    for ptr in pointers:
        per_binary_pointers.setdefault(ptr.binary_name, []).append(
            ptr.section_pointer,
        )

    # Iterate per-binary work in alphabetical order so the concat input
    # list is canonical and the resulting binary_id_per_row numbering is
    # stable.
    #
    # One collector spans every per-binary Stage 1 walk; one flush
    # amortises every call_target row's ``run_lengths`` across the whole
    # batch_load. Caller-owned sessions stay open through both the
    # staging phase AND the post-flush finalise phase.
    collector = BucketedRunLengthCollector()
    pending_decodes: List[Tuple[str, PendingBatchDecode]] = []
    for binary_name in sorted(per_binary_pointers):
        section_pointers = per_binary_pointers[binary_name]
        if binary_name not in sessions:
            raise ValueError(
                "decode_pointer_batch: no open session for binary "
                f"{binary_name!r}",
            )
        session = sessions[binary_name]
        pending = batch_decode(
            session,
            section_pointers,
            num_variants_per_section=num_variants_per_section,
            context_len=context_len,
            max_depth=max_depth,
            variant_padding=variant_padding,
            inlined_equivalent_call_targets_only=(
                inlined_equivalent_call_targets_only
            ),
            include_fid_sidecar=include_fid_sidecar,
            keep_intermediate=False,
            rng=rng,
            collector=collector,
        )
        pending_decodes.append((binary_name, pending))

    # ONE flush -- one pow2-bucketed 2D run_lengths dispatch per bucket
    # across every binary's call_target rows.
    runlen_results = collector.flush()

    per_binary_results: List[Tuple[str, BatchDecodeResult]] = [
        (binary_name, pending.finalise(runlen_results))
        for binary_name, pending in pending_decodes
    ]

    return _concat_results(per_binary_results)


def open_length_bucketed_batch(
    session_factory: Callable[[str], ContextManager[BinarySession]],
    sampler: MultiBinarySortedIndexSampler,
    target_length: int,
    batch_size: int,
    *,
    context_len: int,
    num_variants_per_section: int,
    max_depth: int,
    rng: np.random.Generator,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    inlined_equivalent_call_targets_only: bool = True,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
    band: Optional[Tuple[int, int]] = None,
) -> MultiBinaryBatchDecodeResult:
    """Length-bucketed batch helper (plan D7).

    Samples ``batch_size`` section pointers via ``sampler`` at
    ``target_length`` (or, when ``band=(lo, hi)`` is given, from the
    length band ``[lo, hi]`` inclusive), opens one session per sampled
    binary via ``session_factory``, and delegates the decode +
    concatenation to :func:`decode_pointer_batch`.

    Binary ordering: per-binary results are concatenated in
    alphabetical ``binary_name`` order (the same order
    :attr:`MultiBinarySortedIndexSampler.binary_names` exposes). This
    determines the ``binary_id_per_row`` numbering and is stable
    across runs.

    Parameters
    ----------
    band:
        When ``None`` (default), sampling targets the exact
        ``target_length`` bucket. When ``(lo, hi)``, eligible sections
        are those whose index key falls in ``[lo, hi]`` inclusive
        (length-band sampling); ``target_length`` is then ignored by the
        sampler. This is the motivating case for sampling near a target
        length whose exact bucket is empty but whose neighbourhood is
        populated.

    Raises
    ------
    ValueError
        When the sampler returns 0 pointers (empty pool at
        ``target_length`` -- or across the whole ``band`` -- over every
        binary). The caller is expected to handle this -- either skip
        the training step or widen the band / pick a different
        ``target_length``.
    ValueError
        When ``keep_intermediate=True``. The cross-binary
        :func:`_concat_results` boundary inherently drops per-binary
        :class:`Stage3Batch` intermediates (each is single-binary-
        scoped and not stitchable), so the helper rejects this flag
        rather than silently producing inconsistent state.
    """
    if keep_intermediate:
        raise ValueError(
            "open_length_bucketed_batch: keep_intermediate=True is not "
            "supported -- per-binary Stage3Batch intermediates cannot "
            "cross the concat boundary",
        )

    pointers = sampler.sample_section_pointers(
        target_length, batch_size, rng, band=band,
    )
    if not pointers:
        if band is not None:
            raise ValueError(
                "open_length_bucketed_batch: empty sampler pool in band "
                f"{band}",
            )
        raise ValueError(
            "open_length_bucketed_batch: empty sampler pool at "
            f"target_length={target_length}",
        )

    # Open one session per sampled binary (binaries with zero samples
    # are never opened). The ExitStack spans the whole
    # :func:`decode_pointer_batch` call because its finalise phase reads
    # numpy views backed by session-owned memmaps. Iterate
    # ``binary_names`` (alphabetical) so the open order is canonical;
    # the core re-derives the same order from the pointers themselves.
    sampled_binaries = {ptr.binary_name for ptr in pointers}
    with ExitStack() as session_stack:
        sessions: Dict[str, BinarySession] = {
            binary_name: session_stack.enter_context(
                session_factory(binary_name)
            )
            for binary_name in sampler.binary_names
            if binary_name in sampled_binaries
        }
        return decode_pointer_batch(
            sessions,
            pointers,
            context_len=context_len,
            num_variants_per_section=num_variants_per_section,
            max_depth=max_depth,
            rng=rng,
            variant_padding=variant_padding,
            inlined_equivalent_call_targets_only=(
                inlined_equivalent_call_targets_only
            ),
            include_fid_sidecar=include_fid_sidecar,
        )
