"""Synthetic body-free fixtures for the geometry prepass tests.

Single concern: hand-build a :class:`ColumnarSections` catalog with a
real multi-level call graph + multiple variants per section, the
parallel RLG3 geometry axes (as a directly-constructed
:class:`RealizedGeometryReader`), and a ``_variants.bin`` byte buffer
carrying each variant's prefix size header -- WITHOUT a session, a
``sections.bin`` parse, or any ``_data.bin``. Hand-computable ground
truth so the prepass's emission order / straddler / reservations can be
asserted exactly.

Graph (section : variants -> direct calls by call_target slot):

* sec 0 "root" : 2 variants
    - v0 -> slot0=sec1, slot1=sec2
    - v1 -> slot0=sec1            (v1 does NOT call sec2)
* sec 1 "A"    : 1 variant  -> (leaf)
* sec 2 "B"    : 1 variant  -> slot0=sec3
* sec 3 "C"    : 1 variant  -> (leaf)

Every variant's per-call entry resolves its callee with its OWN J = 0
(usable), so the adjacency takes the primary-J branch (no fallback).

The columnwise-ALL exclusion fires on a callee reached by EVERY mask-row
variant. Over the FULL 2-variant set, sec1 is reached by BOTH -> excluded
+ pruned; sec2 is reached ONLY by v0 -> included by v0, then sec2 (the
only survivor) descends and includes sec3. So the full-set inclusion of
the root is ``{sec2-node, sec3-node}``. A SAMPLED subset of just ``{v0}``
is single-variant, so FLAG-A excludes everything (emits root only); the
remembered-excluded pool is then exactly ``{sec2-node, sec3-node}``. A
sampled subset of ``{v0, v1}`` reproduces the full set (v0 emits sec2 ->
sec3; v1 emits nothing). Both cases are asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.realized_lengths._geometry_reader import (
    RealizedGeometryReader,
)


__all__ = ["SyntheticCorpus", "build_synthetic_corpus"]


@dataclass(frozen=True)
class SyntheticCorpus:
    cols: ColumnarSections
    section_offsets: np.ndarray
    geometry: RealizedGeometryReader
    variants_u8: np.ndarray
    #: per-NODE body length (excludes self-token), for hand-computation.
    body_len: np.ndarray
    id_count: np.ndarray
    value_count: np.ndarray
    #: per-NODE variant-prefix width (n_axis), for hand-computation.
    prefix_len: np.ndarray
    #: per-NODE raw prefix payload ids (``record[1:]``), for value-level
    #: byte-identity assertions on the prefix reader.
    prefix_token_ids: list
    #: var_offsets, exposed for node-index arithmetic in tests.
    var_offsets: np.ndarray


def build_synthetic_corpus(
    *, ct_type: np.ndarray | None = None
) -> SyntheticCorpus:
    """Build the hand-computable synthetic catalog + sidecars.

    ``ct_type`` overrides the per-call_target :class:`CallTargetType`
    array (3 slots, in catalog order: root->sec1, root->sec2, sec2->sec3);
    default is all ``LOCAL``. A PLT slot lets a test prove the edge type
    of a re-inlined / pruned callee is carried VERBATIM (not defaulted to
    LOCAL). EXTERN is gated out of the splice, so only LOCAL / PLT are
    meaningful here.
    """
    # --- section / variant shape ----------------------------------------
    n_variants = np.array([2, 1, 1, 1], dtype=np.int64)  # root, A, B, C
    var_offsets = _csr(n_variants)  # nodes: root=0,1 A=2 B=3 C=4
    total_vars = int(var_offsets[-1])  # 5 nodes

    # Section byte offsets (arbitrary, distinct; used only as the
    # adjacency offset->idx key + the function_section_ptr targets below).
    section_offsets = np.array([0x10, 0x20, 0x30, 0x40], dtype=np.int64)

    # --- call_target tables ---------------------------------------------
    # sec0 has 2 call_targets (->sec1, ->sec2); sec2 has 1 (->sec3);
    # sec1, sec3 have none.
    n_call_targets = np.array([2, 0, 1, 0], dtype=np.int64)
    ct_offsets = _csr(n_call_targets)
    ct_function_section_ptr = np.array(
        [section_offsets[1], section_offsets[2], section_offsets[3]],
        dtype=np.uint32,
    )
    if ct_type is None:
        ct_type = np.array([int(CallTargetType.LOCAL)] * 3, dtype=np.uint8)
    else:
        ct_type = np.asarray(ct_type, dtype=np.uint8).reshape(-1)
        if ct_type.size != 3:
            raise ValueError(
                f"ct_type override must have 3 slots; got {ct_type.size}"
            )

    # --- per-call entries (pce) -----------------------------------------
    # Each variant calls its called slots with its OWN J=0.
    #   root v0: slot0(J0), slot1(J0)   -> calls sec1 AND sec2
    #   root v1: slot0(J0)              -> calls sec1 only
    #   A    v0: none
    #   B    v0: slot0(J0)              -> calls sec3
    #   C    v0: none
    var_n_calls = np.array([2, 1, 0, 1, 0], dtype=np.int64)
    pce_offsets = _csr(var_n_calls)
    pce_called_idx = np.array([0, 1, 0, 0], dtype=np.uint16)
    pce_section_variant_index = np.zeros(4, dtype=np.uint16)  # all J=0

    # --- section-header fields (unused by the prepass but required) -----
    function_name_ptr = np.arange(4, dtype=np.uint32)
    is_duplicated = np.zeros(4, dtype=bool)

    # --- variant prefix widths (n_axis) + _variants.bin -----------------
    # Distinct prefix widths so the layout's per-row prefix offset is
    # exercised. var_ref_offset points at each node's record in the
    # synthetic _variants.bin. Each node's payload carries DISTINCT non-zero
    # raw ids (>= the reserved-digit floor) so a value-level reader that
    # drops the last id (the historical off-by-one) is caught by WHICH ids
    # are gathered -- not just by the width count. Per node, the ids are
    # ``[1000 + node*10 + j for j in range(width)]``.
    prefix_len = np.array([1, 1, 2, 0, 3], dtype=np.int64)  # per NODE
    prefix_token_ids = [
        [1000 + node * 10 + j for j in range(int(width))]
        for node, width in enumerate(prefix_len.tolist())
    ]
    variants_u8, var_ref_offset = _build_variants_bin(
        prefix_len, token_ids=prefix_token_ids
    )

    var_data_offset_shifted = np.zeros(total_vars, dtype=np.uint32)

    cols = ColumnarSections(
        function_name_ptr=function_name_ptr,
        is_duplicated=is_duplicated,
        n_call_targets=n_call_targets,
        n_variants=n_variants,
        ct_offsets=ct_offsets,
        ct_function_name_ptr=np.zeros(3, dtype=np.uint32),
        ct_function_section_ptr=ct_function_section_ptr,
        ct_type=ct_type,
        ct_is_matched=np.ones(3, dtype=bool),
        var_offsets=var_offsets,
        var_ref_offset=var_ref_offset,
        var_data_offset_shifted=var_data_offset_shifted,
        var_n_calls=var_n_calls,
        pce_offsets=pce_offsets,
        pce_called_idx=pce_called_idx,
        pce_section_variant_index=pce_section_variant_index,
    )

    # --- RLG3 geometry axes (per NODE, section-major) -------------------
    # Body lengths chosen so a straddler fires at a small L. own = body+1.
    #   node: root_v0=5 root_v1=7 A=3 B=9 C=6
    body_len = np.array([5, 7, 3, 9, 6], dtype=np.uint32)
    id_count = np.array([2, 1, 0, 3, 2], dtype=np.uint32)
    value_count = np.array([1, 0, 2, 1, 1], dtype=np.uint32)
    csr = var_offsets.astype(np.uint32)  # n_sections + 1, terminator == 5
    geometry = RealizedGeometryReader(
        body_lengths=body_len,
        id_counts=id_count,
        value_counts=value_count,
        csr=csr,
    )

    return SyntheticCorpus(
        cols=cols,
        section_offsets=section_offsets,
        geometry=geometry,
        variants_u8=variants_u8,
        body_len=body_len.astype(np.int64),
        id_count=id_count.astype(np.int64),
        value_count=value_count.astype(np.int64),
        prefix_len=prefix_len,
        prefix_token_ids=prefix_token_ids,
        var_offsets=var_offsets,
    )


def _csr(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def _build_variants_bin(prefix_len: np.ndarray, *, token_ids: list | None = None):
    """A ``_variants.bin`` whose record at ``var_ref_offset[node]`` has a
    leading u16 ``n_tokens == prefix_len[node]``.

    Records are ``[n_tokens, *ids]`` u16 exactly as production writes them
    (:func:`tokenizer.variant_tokens.encoder.encode_record`): the leading
    u16 ``n_tokens`` is the COUNT of payload ids that follow, so the array
    has length ``1 + n_tokens`` and ``tokens[1:]`` (what
    ``get_variant_by_ref`` returns) has ``n_tokens`` ids. The prefix WIDTH
    the prepass derives is therefore ``n_tokens`` itself.

    ``token_ids`` optionally supplies the per-node payload id list (length
    ``prefix_len[node]``); when given, the bytes match what
    ``variant_prefix_values`` must gather, so a byte-identity test can
    assert the actual prefix ids (not just the width). Returns
    ``(variants_u8, var_ref_offset)``.
    """
    records = []
    offsets = []
    cursor = 0
    for node, n_axis in enumerate(prefix_len.tolist()):
        n_tokens = int(n_axis)
        rec = np.empty(1 + n_tokens, dtype=np.uint16)
        rec[0] = n_tokens
        if token_ids is None:
            rec[1:] = 0  # token ids irrelevant to the prefix WIDTH read
        else:
            ids = token_ids[node]
            assert len(ids) == n_tokens, (
                f"_build_variants_bin: token_ids[{node}] has {len(ids)} "
                f"ids, expected n_tokens={n_tokens}"
            )
            rec[1:] = np.asarray(ids, dtype=np.uint16)
        offsets.append(cursor)
        records.append(rec)
        cursor += rec.nbytes
    buf = np.concatenate(records).view(np.uint8) if records else np.zeros(
        0, dtype=np.uint8
    )
    return buf, np.array(offsets, dtype=np.uint32)


class RaisingData:
    """A ``_data.bin`` sentinel that raises on ANY access -- proves the
    prepass never reads bodies. Mimics enough of an ndarray-like handle
    that a stray ``.view`` / index / ``.size`` all blow up.
    """

    def __getattr__(self, name):  # pragma: no cover -- defensive
        raise AssertionError(
            f"prepass touched _data.bin via attribute {name!r} -- it must "
            "be body-free"
        )

    def __getitem__(self, _idx):  # pragma: no cover -- defensive
        raise AssertionError(
            "prepass indexed _data.bin -- it must be body-free"
        )
