"""Top-level ``batch_decode`` entry point -- wires the four stages into the
end-to-end public API.

Composition: :func:`walk_sections` (stage 1) -> :func:`predict_lengths`
(stage 2) -> :func:`build_bulk_bytes` (stage 3) -> :func:`assemble_batch`
(stage 4). Each stage owns ONE concern; this module owns the linear
threading + the public-API surface (default values, RNG defaulting).

Default values match the plan's D5 + D6:

* ``variant_padding=VariantPadding.PAD_NULL`` -- short sections pad with
  all-null-content rows (recommended default).
* ``inlined_equivalent_call_targets_only=True`` -- the walk includes
  only inlining-equivalent (``is_matched``) call targets; the non-
  inlined-equivalent mode is unsupported (the walkers assert on False).
* ``include_fid_sidecar=False``, ``keep_intermediate=False`` -- minimal
  output by default.

Collector lifetime (orchestrator amortisation):

The end-to-end pipeline runs Stage 1 entirely on a
:class:`BucketedRunLengthCollector` (per :mod:`._section_walk`). Two
dispatch shapes on the single :func:`batch_decode` entry, picked via
the ``collector`` kwarg:

* ``collector=None`` (default): owns a fresh collector, runs Stage 1,
  flushes, finalises, then runs Stages 2-4. Returns the
  :class:`BatchDecodeResult` directly.
* ``collector`` provided: stages Stage 1 on the caller-owned
  collector and returns a :class:`PendingBatchDecode`. The caller
  flushes the collector once (potentially across MANY pending
  decodes) and finalises each pending decode by calling
  :meth:`PendingBatchDecode.finalise` with the flush result.

The Stages 2-4 do not touch the collector at all -- they consume the
finalised :class:`Stage1Batch`. The lift is therefore strictly a
Stage-1 amortisation: Stages 2-4 still run per pending decode, but
the ``run_lengths`` dispatches inside Stage 1 are pooled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union, overload

import numpy as np

from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)

from ._assemble import assemble_batch
from ._bulk_bytes import build_bulk_bytes
from ._length_predict import predict_lengths
from ._section_walk import (
    PendingStage1Batch,
    finalise_pending_stage1,
    walk_sections,
)
from ._types import (
    BatchDecodeResult,
    SectionPointerSpec,
    Stage1Batch,
    VariantPadding,
)

if TYPE_CHECKING:
    from ..session import BinarySession


__all__ = ["PendingBatchDecode", "batch_decode"]


@dataclass(frozen=True)
class PendingBatchDecode:
    """End-to-end batch-decode result BEFORE the shared collector has
    been flushed.

    Carries the :class:`PendingStage1Batch` (which holds the un-
    finalised Stage 1 walk) and the Stage 2-4 configuration the
    finaliser needs to reproduce the synchronous-path pipeline. Once
    the orchestrator flushes the shared collector and gets
    ``runlen_results``, it calls :meth:`finalise` to materialise the
    :class:`BatchDecodeResult`.

    Lazy view: the pending stage-1 reference is shared with the
    collector; the rest is plain config scalars.
    """

    stage1_pending: PendingStage1Batch
    context_len: int
    include_fid_sidecar: bool
    keep_intermediate: bool
    emit_block_n_insns_runlength: bool

    def finalise(
        self, runlen_results: dict[int, np.ndarray]
    ) -> BatchDecodeResult:
        """Materialise the :class:`BatchDecodeResult` once the shared
        collector has been flushed.

        Runs the Stage-1 finalisation
        (:func:`finalise_pending_stage1`) then the synchronous Stage
        2 -> 3 -> 4 chain via :func:`_batch_decode_post_stage1`. Pure
        on its inputs modulo the embedded numpy work.
        """
        stage1 = finalise_pending_stage1(self.stage1_pending, runlen_results)
        return _batch_decode_post_stage1(
            stage1,
            context_len=self.context_len,
            include_fid_sidecar=self.include_fid_sidecar,
            keep_intermediate=self.keep_intermediate,
            emit_block_n_insns_runlength=self.emit_block_n_insns_runlength,
        )


@overload
def batch_decode(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: VariantPadding = ...,
    inlined_equivalent_call_targets_only: bool = ...,
    include_fid_sidecar: bool = ...,
    keep_intermediate: bool = ...,
    emit_block_n_insns_runlength: bool = ...,
    rng: "Optional[np.random.Generator]" = ...,
    collector: None = ...,
) -> BatchDecodeResult: ...
@overload
def batch_decode(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: VariantPadding = ...,
    inlined_equivalent_call_targets_only: bool = ...,
    include_fid_sidecar: bool = ...,
    keep_intermediate: bool = ...,
    emit_block_n_insns_runlength: bool = ...,
    rng: "Optional[np.random.Generator]" = ...,
    collector: BucketedRunLengthCollector,
) -> PendingBatchDecode: ...
def batch_decode(
    session: "BinarySession",
    section_pointers: List[SectionPointerSpec],
    *,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    inlined_equivalent_call_targets_only: bool = True,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
    emit_block_n_insns_runlength: bool = False,
    rng: "Optional[np.random.Generator]" = None,
    collector: Optional[BucketedRunLengthCollector] = None,
) -> Union[BatchDecodeResult, PendingBatchDecode]:
    """End-to-end batch decode: stage 1 -> 2 -> 3 -> 4.

    See ``batch_decode_plan.md`` for the full pipeline design; this
    function is purely the linear composition of the four single-concern
    stage entry points.

    Parameters
    ----------
    session:
        :class:`BinarySession` whose per-arm loaders supply the
        section + function bodies. The session is the boundary across
        which raw-data access happens; all four stages read THROUGH it.
    section_pointers:
        List of :class:`SectionPointerSpec` -- one ``(arm, idx)`` per
        section to include in the batch.
    num_variants_per_section:
        Variant-sampling count per section. Combined with
        ``variant_padding`` to compute the batch's ``batch_size`` at
        stage 1.
    context_len:
        Per-row token budget (column count of the output tensor).
    max_depth:
        Stage-1 splice-tree DFS depth cap (inlining-depth bound).
    variant_padding:
        :class:`VariantPadding` policy for the variant axis (plan D6 +
        ALG-10).
    inlined_equivalent_call_targets_only:
        When True (the default), stage 1's callee walk includes only
        call targets whose ``is_matched`` flag is True (i.e. only
        inlining-equivalent callees). Passing False is unsupported --
        the stage-1 walkers assert on it.
    include_fid_sidecar:
        When True, stage 4 builds the optional
        ``(fid_sidecar, fid_row_offsets)`` pair (plan D5).
    keep_intermediate:
        When True, the finalised :class:`Stage3Batch` is carried on the
        result's :attr:`BatchDecodeResult.intermediate` field.
    emit_block_n_insns_runlength:
        When True, stage 4 builds the optional metatoken-runlength
        sidecars ``(block_runlength, block_runlength_row_offsets,
        insn_runlength, insn_runlength_row_offsets)`` on the result.
        Computed via :func:`._runlengths.compute_metatoken_runlengths`
        (the canonical FTL accountant); enables consumers like the
        inspector to derive metatoken-level block / instruction
        boundaries without reaching into :attr:`BatchDecodeResult.intermediate`.
    rng:
        :class:`numpy.random.Generator` for stage 1's variant sampling.
        Defaults to a fresh :func:`numpy.random.default_rng` -- i.e.
        non-reproducible. Pass an explicit generator for deterministic
        sampling.
    collector:
        Optional caller-owned :class:`BucketedRunLengthCollector`.
        When ``None`` (default): the function runs Stages 1-4
        end-to-end and returns a :class:`BatchDecodeResult`. When
        provided: Stage 1 stages onto the supplied collector but the
        function returns a :class:`PendingBatchDecode` whose
        :meth:`finalise` (called with the orchestrator's flush
        result) yields the :class:`BatchDecodeResult`. The orchestrator
        is responsible for flushing once it has every pending decode.
    """
    # Variant-prefix assembly REQUIRES the unified vocab; fail LOUD on a
    # vocab-less (length/graph-only) session rather than silently dropping
    # the prefix from every decoded row. Guarded on ``section_pointers``
    # being non-empty: an empty batch assembles zero rows, so there is no
    # prefix to drop and no vocab needed (and the no-op caller may legitimately
    # pass no session). The length/graph paths never reach batch_decode (they
    # go through ``_bulk_expand_lengths``), so this is the narrowest seam that
    # is prefix-consuming-ONLY and still holds the session/vocab. See
    # ``BinarySession.require_vocab_manager``.
    if section_pointers:
        session.require_vocab_manager()

    if rng is None:
        rng = np.random.default_rng()

    # ONE decision point: own-collector flush-now OR caller-owned defer.
    if collector is None:
        stage1 = walk_sections(
            session,
            section_pointers,
            num_variants_per_section=num_variants_per_section,
            max_depth=max_depth,
            variant_padding=variant_padding,
            inlined_equivalent_call_targets_only=(
                inlined_equivalent_call_targets_only
            ),
            rng=rng,
        )
        return _batch_decode_post_stage1(
            stage1,
            context_len=context_len,
            include_fid_sidecar=include_fid_sidecar,
            keep_intermediate=keep_intermediate,
            emit_block_n_insns_runlength=emit_block_n_insns_runlength,
        )

    stage1_pending = walk_sections(
        session,
        section_pointers,
        num_variants_per_section=num_variants_per_section,
        max_depth=max_depth,
        variant_padding=variant_padding,
        inlined_equivalent_call_targets_only=(
            inlined_equivalent_call_targets_only
        ),
        rng=rng,
        collector=collector,
    )
    return PendingBatchDecode(
        stage1_pending=stage1_pending,
        context_len=context_len,
        include_fid_sidecar=include_fid_sidecar,
        keep_intermediate=keep_intermediate,
        emit_block_n_insns_runlength=emit_block_n_insns_runlength,
    )


def _batch_decode_post_stage1(
    stage1: Stage1Batch,
    *,
    context_len: int,
    include_fid_sidecar: bool,
    keep_intermediate: bool,
    emit_block_n_insns_runlength: bool,
) -> BatchDecodeResult:
    """Run Stages 2 -> 3 -> 4 on a finalised :class:`Stage1Batch`.

    Pure delegation: stage 2 (length predict) -> stage 3 (bulk bytes)
    -> stage 4 (assemble). Stage 1 is the only stage that touches a
    run-length collector; once it's finalised the rest of the pipeline
    is collector-agnostic.
    """
    stage2 = predict_lengths(stage1, context_len=context_len)
    stage3 = build_bulk_bytes(stage2)
    return assemble_batch(
        stage3,
        context_len=context_len,
        include_fid_sidecar=include_fid_sidecar,
        keep_intermediate=keep_intermediate,
        emit_block_n_insns_runlength=emit_block_n_insns_runlength,
    )
