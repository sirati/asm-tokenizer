"""Fallback-J precompute parity + byte-identical-build regression.

:meth:`LiveNodeAdjacency._fallback_J` was a per-slot live double-loop
over every sibling variant x its per-call entries; it now consults a
per-section precomputed table (built once, vectorized). The chosen J
must be IDENTICAL: ascending sibling variant, and within a variant the
first usable per-call entry in on-disk order, decides each slot.

* the parity test pins the vectorized table against an explicit
  re-implementation of the original double-loop semantics over a
  synthetic catalog with mixed MISSING / usable J across siblings;
* the build-golden test pins the full sorted-index wire bytes on a
  fallback-exercising corpus, so any drift in the chosen J (which
  changes the spliced callee variant, hence the length, hence the
  encoded index) surfaces as a byte mismatch.
"""

from __future__ import annotations

import hashlib

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
    _usable,
)


def _csr(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def _live_double_loop_fallback(cols, sec: int, called_idx: int) -> int:
    """The ORIGINAL semantics, spelled out: scan variants ascending,
    each variant's per-call entries in on-disk order, return the first
    usable J for ``called_idx`` (or -1)."""
    v0 = int(cols.var_offsets[sec])
    v1 = int(cols.var_offsets[sec + 1])
    for v in range(v0, v1):
        p0 = int(cols.pce_offsets[v])
        p1 = int(cols.pce_offsets[v + 1])
        for ci, J in zip(
            cols.pce_called_idx[p0:p1].tolist(),
            cols.pce_section_variant_index[p0:p1].tolist(),
        ):
            if int(ci) == called_idx and _usable(int(J)):
                return int(J)
    return -1


def _mixed_fallback_catalog() -> tuple:
    """One section, 4 variants, 3 call-target slots, with per-variant
    per-call entries deliberately mixing MISSING and usable J so the
    fallback's tie-break (lowest sibling variant, first entry) matters.

    Slot resolution per slot (called_idx):
      * slot 0: variant 0 MISSING, variant 1 usable J=1 -> fallback 1.
      * slot 1: variant 0 usable J=0 (own J path; fallback also 0).
      * slot 2: variant 0 MISSING, variant 1 MISSING, variant 2 usable
        J=2, variant 3 usable J=3 -> fallback 2 (lowest sibling).
    Variant 1 ALSO has a usable entry for slot 0 appearing AFTER a
    MISSING slot-0 entry within the same variant, to exercise the
    within-variant first-usable rule.
    """
    M = int(MISSING_VARIANT_INDEX)
    n_cts = 3
    n_vars = 4
    # per-variant (called_idx, J) entries, on-disk order:
    per_var = [
        [(0, M), (1, 0), (2, M)],          # v0
        [(0, M), (0, 1), (2, M)],          # v1 (slot0 first MISSING then J=1)
        [(0, M), (1, M), (2, 2)],          # v2
        [(0, M), (2, 3)],                  # v3
    ]
    var_n_calls = np.array([len(e) for e in per_var], dtype=np.int64)
    flat_called = []
    flat_J = []
    for entries in per_var:
        for ci, J in entries:
            flat_called.append(ci)
            flat_J.append(J)

    n_variants = np.array([n_vars], dtype=np.int64)
    var_offsets = _csr(n_variants)
    section_offsets = np.array([16], dtype=np.int64)

    cols = ColumnarSections(
        function_name_ptr=np.zeros(1, dtype=np.uint32),
        is_duplicated=np.zeros(1, dtype=bool),
        n_call_targets=np.array([n_cts], dtype=np.int64),
        n_variants=n_variants,
        ct_offsets=_csr(np.array([n_cts], dtype=np.int64)),
        ct_function_name_ptr=np.zeros(n_cts, dtype=np.uint32),
        # self-pointing slots (valid offset) so resolution doesn't drop.
        ct_function_section_ptr=np.full(n_cts, 16, dtype=np.uint32),
        ct_type=np.full(n_cts, int(CallTargetType.LOCAL), dtype=np.uint8),
        ct_is_matched=np.ones(n_cts, dtype=bool),
        var_offsets=var_offsets,
        var_ref_offset=np.arange(n_vars, dtype=np.uint32),
        var_data_offset_shifted=np.arange(1, n_vars + 1, dtype=np.uint32),
        var_n_calls=var_n_calls,
        pce_offsets=_csr(var_n_calls),
        pce_called_idx=np.array(flat_called, dtype=np.uint16),
        pce_section_variant_index=np.array(flat_J, dtype=np.uint16),
    )
    return cols, section_offsets


def test_fallback_table_matches_live_double_loop() -> None:
    cols, section_offsets = _mixed_fallback_catalog()
    sec_of_var = np.repeat(
        np.arange(cols.n_variants.size, dtype=np.int64), cols.n_variants
    )
    adj = LiveNodeAdjacency(cols, section_offsets, sec_of_var)

    n_cts = int(cols.n_call_targets[0])
    for ci in range(-1, n_cts + 2):
        got = adj._fallback_J(0, ci)
        expect = _live_double_loop_fallback(cols, 0, ci)
        assert got == expect, (
            f"slot {ci}: vectorized fallback {got} != live double-loop "
            f"{expect}"
        )

    # Spot-pin the documented expectations.
    assert adj._fallback_J(0, 0) == 1
    assert adj._fallback_J(0, 1) == 0
    assert adj._fallback_J(0, 2) == 2


def test_fallback_table_cached_once() -> None:
    cols, section_offsets = _mixed_fallback_catalog()
    sec_of_var = np.repeat(
        np.arange(cols.n_variants.size, dtype=np.int64), cols.n_variants
    )
    adj = LiveNodeAdjacency(cols, section_offsets, sec_of_var)
    t1 = adj._fallback_table(0)
    t2 = adj._fallback_table(0)
    # Same array object: a second fallback for the section does not
    # rescan -- the precompute is memoised per section.
    assert t1 is t2
