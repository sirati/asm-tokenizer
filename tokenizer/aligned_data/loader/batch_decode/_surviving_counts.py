"""Stage 2c — surviving identity + number-chunk count prediction.

Single concern: given one call-target's post-promotion + post-strip +
post-shift ``expanded_token_ids`` and the per-call-target
``partial_cut_length`` produced by 2b, count how many positions in the
surviving prefix fall in the IDENTITY band and how many fall in the
NUMBER band. These two integers feed Stage 2's per-call-target
``surviving_identity_count`` / ``surviving_number_chunk_count`` fields
and the per-variant aggregation.

Why one position == one number chunk
------------------------------------
Stage 2a (`_expand_tokens.py`) already promoted every multi-chunk number
source — VC2 carriers spread across the source's runlen, F128 finite
sources painted with a second-chunk carrier — so the post-promotion
stream has *one consecutive number-token position per produced chunk*.
That is the D2 + Stage 2 step 5 invariant the plan relies on:

    "Each token = exactly 1 sidecar entry (promotion already expanded
     multi-chunk sources to consecutive tokens — D2's rule)."

So a simple band-mask cardinality over the slice yields the exact
chunk count; we never have to re-derive chunk counts here.

Band layout (post `-256` shift, plan D5 + vocab layout)
-------------------------------------------------------
The canonical vocab anchors live on
:class:`tokenizer.token_manager.VocabularyManager`:

* ``_V2_RESERVED_DIGIT_COUNT`` (= 256) -- wire-stream digit boundary;
  the post-strip shift subtracts this value.
* ``_V2_NUMBER_BLOCK_START``  (= 257) -- first NUMBER block id;
  becomes id ``1`` post-shift.
* ``_V2_IDENTITY_BLOCK_START`` (= 264) -- first IDENTITY block id;
  becomes id ``8`` post-shift (also the first id *outside* the NUMBER
  block).
* ``_V2_EAGER_BLOCK_END`` (= 272) -- first instruction-rep slot;
  becomes id ``16`` post-shift (first id *outside* the IDENTITY block).

NUMBER band post-shift: ``[1, 8)`` -- ``VC2=1, F16=2, BF16=3, F32=4,
F64=5, F80=6, F128=7``.

IDENTITY band post-shift: ``[8, 16)`` -- ``BLOCK_V2=8, LOCAL_FUNC=9,
PLT_FUNC=10, EXT_FUNC=11, STRING_PTR=12, JUMP_TABLE=13,
RO_DATA_PTR=14, RW_DATA_PTR=15``.

Post-shift id ``0`` is the reserved [null-content] slot (plan D5) and
must NOT contribute to either band -- the lower bound of the NUMBER
band is therefore strictly ``>= 1``.

The constants below are derived from the
:class:`VocabularyManager` anchors at import time so that a future
canonical-block extension only requires updating the source-of-truth
constants; this module rebinds automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.token_manager import VocabularyManager

from ._family_band_reduction import family_band_reduction


__all__ = ["SurvivingCounts", "count_surviving", "count_surviving_batched"]


# ---------------------------------------------------------------------------
# Band constants -- derived from the VocabularyManager source of truth.
#
# Every constant below MUST stay aligned with the canonical vocab layout.
# Reading them from the anchor class avoids drift if the canonical band
# boundaries ever move (e.g. a new metadata-variant block is inserted
# between NUMBER and IDENTITY).
# ---------------------------------------------------------------------------
_NUMBER_BAND_LO_SHIFTED = (
    VocabularyManager._V2_NUMBER_BLOCK_START
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)  # = 257 - 256 = 1
_NUMBER_BAND_HI_SHIFTED = (
    VocabularyManager._V2_IDENTITY_BLOCK_START
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)  # = 264 - 256 = 8 (exclusive)
_IDENTITY_BAND_LO_SHIFTED = _NUMBER_BAND_HI_SHIFTED  # = 8
_IDENTITY_BAND_HI_SHIFTED = (
    VocabularyManager._V2_EAGER_BLOCK_END
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)  # = 272 - 256 = 16 (exclusive)


@dataclass(frozen=True)
class SurvivingCounts:
    """2c's output for one call_target (already cut by 2b).

    Consumed by 2d to set :attr:`Stage2CallTarget.surviving_identity_count`
    and :attr:`Stage2CallTarget.surviving_number_chunk_count`, and by the
    :class:`Stage2Variant` aggregation step.

    Note: ``surviving_token_count`` is upstream (2b's ``CutoffResult``
    provides it via ``partial_cut_length`` or full length); this struct
    only carries the two band-mask cardinalities.
    """

    surviving_identity_count: int
    surviving_number_chunk_count: int


def count_surviving(
    expanded_token_ids: np.ndarray,
    partial_cut_length: int,
) -> SurvivingCounts:
    """Count identity-token + number-chunk-token positions in
    ``expanded_token_ids[:partial_cut_length]``.

    The number-chunk count equals exactly the count of number-token
    positions in the slice (per plan D2 + Stage 2 step 5: multi-chunk
    sources have already been promoted to multiple consecutive
    number-token positions by 2a, so one position == one chunk).

    Parameters
    ----------
    expanded_token_ids:
        Post-promotion, post-strip, post-shift ``u16`` token stream for
        ONE call_target. The caller (Stage 2d) supplies the array
        directly from ``Stage2CallTarget.expanded_token_ids``.
    partial_cut_length:
        Number of leading positions to include. ``0`` skips the call
        target entirely (returns ``(0, 0)``); a value greater than
        ``len(expanded_token_ids)`` is *clamped* to the array length
        (numpy slicing already does this -- callers may pass the array
        length for fully-included call_targets without bothering to
        re-clip).

    Returns
    -------
    :class:`SurvivingCounts`
        Two-field tuple-like with the IDENTITY-band cardinality and the
        NUMBER-band cardinality.

    Notes
    -----
    See :doc:`batch_decode_plan` ``## Stages -- algorithm sketch``
    Stage 2 step 5 for the masking formulation.
    """

    # Fast-path: a zero-length surviving prefix contributes nothing.
    # np.uint16's vectorised compare would also produce zero here, but
    # the early return avoids materialising any masks.
    if partial_cut_length <= 0:
        return SurvivingCounts(
            surviving_identity_count=0,
            surviving_number_chunk_count=0,
        )

    # numpy's slice semantics already clamp ``stop`` to the array length,
    # so we don't need to ``min(partial_cut_length, len(...))`` ourselves
    # -- ``expanded_token_ids[:partial_cut_length]`` is the surviving
    # prefix per plan Stage 2 step 5.
    surviving = expanded_token_ids[:partial_cut_length]

    # Two vectorised band masks. We mask the SAME slice twice (rather
    # than masking once and then partitioning) because the two bands are
    # disjoint and contiguous -- the mask compares fuse to two
    # comparisons each in numpy's hot path, and the cardinalities are
    # what the caller wants. The bands include the lower bound and
    # exclude the upper bound (left-closed, right-open) per the plan's
    # vocab layout.
    identity_mask = (surviving >= _IDENTITY_BAND_LO_SHIFTED) & (
        surviving < _IDENTITY_BAND_HI_SHIFTED
    )
    number_mask = (surviving >= _NUMBER_BAND_LO_SHIFTED) & (
        surviving < _NUMBER_BAND_HI_SHIFTED
    )

    return SurvivingCounts(
        surviving_identity_count=int(identity_mask.sum()),
        surviving_number_chunk_count=int(number_mask.sum()),
    )


def count_surviving_batched(
    expanded_flat: np.ndarray,
    node_offsets: np.ndarray,
    surviving_token_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched twin of :func:`count_surviving` over a flat node stream.

    Computes, for EVERY node in one segmented band-mask reduction, the
    two band cardinalities :func:`count_surviving` returns per node --
    over the node's surviving prefix
    ``expanded_flat[node_offsets[e] : node_offsets[e] +
    min(surviving_token_counts[e], node_len[e])]``.

    Parameters
    ----------
    expanded_flat:
        The flat post-promotion ``u16`` stream for the whole batch; node
        ``e`` owns ``expanded_flat[node_offsets[e] : node_offsets[e + 1]]``.
    node_offsets:
        ``int64[n_nodes + 1]`` CSR jump table into ``expanded_flat``.
    surviving_token_counts:
        ``int64[n_nodes]`` -- node ``e``'s surviving prefix length (the
        per-node ``partial_cut_length``). Clamped to the node length, as
        numpy slicing clamps ``stop`` in the scalar path.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(surviving_identity_count, surviving_number_chunk_count)``, each
        ``int64[n_nodes]`` -- the per-node IDENTITY-band and NUMBER-band
        cardinalities. Identical, element-for-element, to looping
        :func:`count_surviving` over the nodes.
    """
    node_offsets = np.asarray(node_offsets, dtype=np.int64)
    n_nodes = node_offsets.shape[0] - 1
    if n_nodes <= 0 or expanded_flat.shape[0] == 0:
        zeros = np.zeros(max(n_nodes, 0), dtype=np.int64)
        return zeros, zeros.copy()

    # Thin adapter over the shared segmented band-reduction kernel: the
    # IDENTITY-band / NUMBER-band cardinalities are the first two columns of
    # the single pass. The preamble (node_id / offset_in_node / surviving
    # clip) and both band masks live once in the kernel -- never re-derived
    # here.
    result = family_band_reduction(
        expanded_flat, node_offsets, surviving_token_counts
    )
    return (
        result.surviving_identity_count,
        result.surviving_number_chunk_count,
    )
