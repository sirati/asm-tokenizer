"""Shared seam for the per-node surviving-prefix band reduction.

Single concern (the design-first sentence): *given a flat post-shift
``expanded`` u16 stream, its per-node CSR ``node_offsets``, and the per-node
``surviving`` prefix length, run the ONE segmented band-reduction pass the
three vector-decode call sites share* — ``count_surviving_batched``
(:mod:`._surviving_counts`), ``_build_instream_columns``
(:mod:`...vector_batch._scatter._remap_inputs`), and
``build_number_chunk_columns``
(:mod:`...vector_batch._scatter._number_chunk_columns`) — *and return every
column the three slice out of it.*

Why this module exists: those three call sites each recomputed the IDENTICAL
preamble (``node_id``/``offset_in_node``/``within = offset_in_node <
surviving[node_id]``) over the same arrays. The preamble + the per-band
selection now live ONCE, in the GIL-released
:func:`dedup_hashmap.build_family_band_reduction_kernel`; this module owns the
single kernel-call seam (the band constants + the slot LUTs the kernel needs
as params) and hands each adapter a typed :class:`FamilyBandReduction` to
slice. No caller learns the kernel internals; no caller re-derives the
preamble.

Module boundary: the kernel owns the segmented pass; this module owns the
band/LUT param assembly + the typed result; each adapter owns only its own
slice/gather. The band constants and slot LUTs are derived from the canonical
:class:`VocabularyManager` anchors + the shared
``_FUNC_SHIFTED_TO_SLOT`` / ``_COUNTER_SHIFTED_TO_SLOT`` maps, so a vocab- or
slot-layout change reshapes both Python params and the kernel's behaviour
together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dedup_hashmap import build_family_band_reduction_kernel

from tokenizer.token_manager import VocabularyManager

from ._dedup_walk._flat_extract import (
    _COUNTER_SHIFTED_TO_SLOT,
    _FUNC_SHIFTED_TO_SLOT,
)


__all__ = ["FamilyBandReduction", "family_band_reduction"]


# ---------------------------------------------------------------------------
# Band constants -- derived from the canonical VocabularyManager anchors, the
# same source of truth :mod:`._surviving_counts` reads. Post-shift bands:
# NUMBER [1, 8), IDENTITY [8, 16).
# ---------------------------------------------------------------------------
_NUMBER_BAND_LO = (
    VocabularyManager._V2_NUMBER_BLOCK_START
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)
_NUMBER_BAND_HI = (
    VocabularyManager._V2_IDENTITY_BLOCK_START
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)
_IDENTITY_BAND_LO = _NUMBER_BAND_HI
_IDENTITY_BAND_HI = (
    VocabularyManager._V2_EAGER_BLOCK_END
    - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)


def _build_slot_lut(shifted_to_slot: dict[int, int]) -> np.ndarray:
    """Dense ``int64`` LUT indexed by ``shifted_id - IDENTITY_BAND_LO``.

    The kernel maps each in-stream IDENTITY-band id to a slot by indexing
    this array at ``id - IDENTITY_BAND_LO``; absent ids carry the ``-1``
    sentinel (the same default the numpy ``np.full(..., -1)`` columns use).
    Built off the shared ``_*_SHIFTED_TO_SLOT`` maps so a slot-layout change
    reshapes the LUT without touching the kernel.
    """
    band = _IDENTITY_BAND_HI - _IDENTITY_BAND_LO
    lut = np.full(band, -1, dtype=np.int64)
    for shifted, slot in shifted_to_slot.items():
        lut[shifted - _IDENTITY_BAND_LO] = slot
    return lut


_FUNC_SLOT_LUT = _build_slot_lut(_FUNC_SHIFTED_TO_SLOT)
_COUNTER_SLOT_LUT = _build_slot_lut(_COUNTER_SHIFTED_TO_SLOT)


@dataclass(frozen=True)
class FamilyBandReduction:
    """The seven columns the single segmented pass produces.

    Each of the three thin adapters slices out only its portion:

    * ``count_surviving_batched`` -> ``(surviving_identity_count,
      surviving_number_chunk_count)``.
    * ``_build_instream_columns`` -> ``(instream_off, instream_func_slot,
      instream_counter_slot)``.
    * ``build_number_chunk_columns`` -> ``(number_out_block,
      number_ct_ordinal)`` (then gathers slice starts + builds the variant
      CSR in Python).
    """

    surviving_identity_count: np.ndarray  # i64[n_nodes]
    surviving_number_chunk_count: np.ndarray  # i64[n_nodes]
    instream_off: np.ndarray  # i64[n_nodes + 1]
    instream_func_slot: np.ndarray  # i64[n_instream]
    instream_counter_slot: np.ndarray  # i64[n_instream]
    number_out_block: np.ndarray  # i64[n_number]
    number_ct_ordinal: np.ndarray  # i64[n_number]


def family_band_reduction(
    expanded: np.ndarray,
    node_offsets: np.ndarray,
    surviving: np.ndarray,
) -> FamilyBandReduction:
    """Run the single GIL-free segmented band-reduction over the prefix.

    Parameters
    ----------
    expanded:
        The flat post-shift ``u16`` stream; node ``e`` owns
        ``expanded[node_offsets[e] : node_offsets[e + 1]]``.
    node_offsets:
        ``int64[n_nodes + 1]`` CSR jump table into ``expanded``.
    surviving:
        ``int64[n_nodes]`` -- node ``e``'s surviving prefix length, clamped
        to the node length (an over-long value includes the whole node, as
        numpy slicing clamps ``stop``).

    Returns
    -------
    FamilyBandReduction
        The seven per-node / per-position columns the three call sites slice.
    """
    (
        surviving_identity_count,
        surviving_number_chunk_count,
        instream_off,
        instream_func_slot,
        instream_counter_slot,
        number_out_block,
        number_ct_ordinal,
    ) = build_family_band_reduction_kernel(
        np.ascontiguousarray(expanded, dtype=np.uint16).reshape(-1),
        np.ascontiguousarray(node_offsets, dtype=np.int64).reshape(-1),
        np.ascontiguousarray(surviving, dtype=np.int64).reshape(-1),
        np.uint16(_NUMBER_BAND_LO),
        np.uint16(_NUMBER_BAND_HI),
        np.uint16(_IDENTITY_BAND_LO),
        np.uint16(_IDENTITY_BAND_HI),
        _FUNC_SLOT_LUT,
        _COUNTER_SLOT_LUT,
    )
    return FamilyBandReduction(
        surviving_identity_count=surviving_identity_count,
        surviving_number_chunk_count=surviving_number_chunk_count,
        instream_off=instream_off,
        instream_func_slot=instream_func_slot,
        instream_counter_slot=instream_counter_slot,
        number_out_block=number_out_block,
        number_ct_ordinal=number_ct_ordinal,
    )
