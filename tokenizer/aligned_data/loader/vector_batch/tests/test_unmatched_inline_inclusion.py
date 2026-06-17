"""Flag-ON unmatched-outline inlining over the inclusion BFS.

The real corpus's matched/unmatched arm split means an unmatched call
target's body lives in the OTHER arm (unreachable from a single-arm walk),
so the surfacing path almost never fires on production binaries. This
synthetic catalog DELIBERATELY places matched callees behind unmatched
outlines WITHIN one arm so the surfacing + recursion + cap are exercised
end-to-end through :func:`compute_row_inclusions` (the shared inclusion BFS
both vector_batch and -- structurally identically -- batch_decode drive).

The fixtures assert that with the flag ON the matched callees behind
unmatched outlines ENTER outline detection (appear in the emitted node
list) where with the flag OFF the unmatched outline shells did; and that
the depth cap blocks a matched callee sitting one hop too deep. The tests
are mutation-sensitive: breaking the surfacing, the recursion, or the cap
flips an assertion.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.loader.vector_batch._inclusion import (
    compute_row_inclusions,
)


def _csr(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def _build_catalog(
    *,
    n_variants,
    section_offsets,
    ct_function_section_ptr,
    ct_is_matched,
    n_call_targets,
    var_n_calls,
    pce_called_idx,
    pce_section_variant_index,
):
    """Assemble a minimal :class:`ColumnarSections` (all-LOCAL ct_type)."""
    n_variants = np.asarray(n_variants, dtype=np.int64)
    n_call_targets = np.asarray(n_call_targets, dtype=np.int64)
    var_n_calls = np.asarray(var_n_calls, dtype=np.int64)
    n_ct = int(n_call_targets.sum())
    cols = ColumnarSections(
        function_name_ptr=np.arange(n_variants.size, dtype=np.uint32),
        is_duplicated=np.zeros(n_variants.size, dtype=bool),
        n_call_targets=n_call_targets,
        n_variants=n_variants,
        ct_offsets=_csr(n_call_targets),
        ct_function_name_ptr=np.zeros(n_ct, dtype=np.uint32),
        ct_function_section_ptr=np.asarray(
            ct_function_section_ptr, dtype=np.uint32
        ),
        ct_type=np.full(n_ct, int(CallTargetType.LOCAL), dtype=np.uint8),
        ct_is_matched=np.asarray(ct_is_matched, dtype=bool),
        var_offsets=_csr(n_variants),
        var_ref_offset=np.zeros(int(_csr(n_variants)[-1]), dtype=np.uint32),
        var_data_offset_shifted=np.zeros(
            int(_csr(n_variants)[-1]), dtype=np.uint32
        ),
        var_n_calls=var_n_calls,
        pce_offsets=_csr(var_n_calls),
        pce_called_idx=np.asarray(pce_called_idx, dtype=np.uint16),
        pce_section_variant_index=np.asarray(
            pce_section_variant_index, dtype=np.uint16
        ),
    )
    return cols, np.asarray(section_offsets, dtype=np.int64)


def _emitted_callee_nodes(inclusions):
    """The non-root emitted node sets per row (root at index 0 dropped)."""
    return [set(inc.emitted_nodes[1:].tolist()) for inc in inclusions]


def _run(cols, section_offsets, *, unmatched_inline, depth=3):
    # Root section 0, two sampled variants (v0, v1) forming ONE decider
    # group. Two rows so columnwise-ALL does not trivially exclude a callee
    # reached by only one variant.
    return compute_row_inclusions(
        cols,
        section_offsets,
        root_sections=np.array([0, 0], dtype=np.int64),
        root_sampled_variants=np.array([0, 1], dtype=np.int64),
        root_groups=np.array([0, 0], dtype=np.int64),
        max_depth=10,
        need_excluded_pool=False,
        unmatched_inline=unmatched_inline,
        unmatched_inline_depth=depth,
    )


def test_unmatched_to_matched_surfaces_under_flag():
    """root -> unmatched U -> matched M; flag ON surfaces M, drops U.

    Sections: 0 root (2 var), 1 U (unmatched, 1 var), 2 M (matched, 1 var).
    root v0 calls U (slot0); root v1 calls nothing. U calls M (slot0).
    With the flag OFF, v0 splices U (the unmatched shell). With it ON, U is
    looked THROUGH: M is surfaced at level 1 in place of U.
    """
    section_offsets = [0x10, 0x20, 0x30]
    cols, offs = _build_catalog(
        n_variants=[2, 1, 1],
        section_offsets=section_offsets,
        # ct slots: [0]=root->U (unmatched), [1]=U->M (matched).
        n_call_targets=[1, 1, 0],
        ct_function_section_ptr=[0x20, 0x30],
        ct_is_matched=[False, True],
        # nodes: root v0=0, root v1=1, U=2, M=3.
        # var_n_calls per node: root_v0 calls slot0; root_v1 none; U calls
        # slot0; M none.
        var_n_calls=[1, 0, 1, 0],
        pce_called_idx=[0, 0],  # root_v0 slot0 ; U slot0
        pce_section_variant_index=[0, 0],
    )
    off = _run(cols, offs, unmatched_inline=False)
    on = _run(cols, offs, unmatched_inline=True)

    u_node = int(cols.var_offsets[1])  # U's only node
    m_node = int(cols.var_offsets[2])  # M's only node

    off_callees = _emitted_callee_nodes(off)
    on_callees = _emitted_callee_nodes(on)
    # Flag OFF: v0 splices the unmatched shell U (single-variant reach -> not
    # all-variants-excluded), NOT M (M is behind U, depth 2, not yet via the
    # cap-less walk at this shape since U is included+descends -> M at level
    # 2). So U is present OFF.
    assert u_node in off_callees[0]
    # Flag ON: U is replaced by M at level 1; the shell U is gone.
    assert m_node in on_callees[0]
    assert u_node not in on_callees[0]


def test_unmatched_unmatched_matched_recurses():
    """root -> U1 -> U2 -> M; flag ON (cap>=2) surfaces M behind 2 outlines."""
    section_offsets = [0x10, 0x20, 0x30, 0x40]
    cols, offs = _build_catalog(
        n_variants=[2, 1, 1, 1],
        section_offsets=section_offsets,
        # slots: [0]=root->U1 (unmatched), [1]=U1->U2 (unmatched),
        # [2]=U2->M (matched).
        n_call_targets=[1, 1, 1, 0],
        ct_function_section_ptr=[0x20, 0x30, 0x40],
        ct_is_matched=[False, False, True],
        # nodes: root v0=0, root v1=1, U1=2, U2=3, M=4.
        # pce_called_idx is SECTION-LOCAL: each caller's single slot is its
        # local index 0.
        var_n_calls=[1, 0, 1, 1, 0],
        pce_called_idx=[0, 0, 0],  # root_v0 slot0 ; U1 slot0 ; U2 slot0
        pce_section_variant_index=[0, 0, 0],
    )
    m_node = int(cols.var_offsets[3])  # M's node (section idx 3)
    on = _run(cols, offs, unmatched_inline=True, depth=3)
    on_callees = _emitted_callee_nodes(on)
    assert m_node in on_callees[0]


def test_depth_cap_blocks_too_deep_matched():
    """root -> U1 -> U2 -> U3 -> M; cap=2 must NOT surface M (M is 3 hops).

    Reaching M requires recursing INTO U3 (a 3rd unmatched hop). At cap=2
    the recursion stops before resolving U3's children, so M is not
    surfaced; at cap=3 it is.
    """
    section_offsets = [0x10, 0x20, 0x30, 0x40, 0x50]
    cols, offs = _build_catalog(
        n_variants=[2, 1, 1, 1, 1],
        section_offsets=section_offsets,
        # slots: [0]root->U1, [1]U1->U2, [2]U2->U3, [3]U3->M.
        n_call_targets=[1, 1, 1, 1, 0],
        ct_function_section_ptr=[0x20, 0x30, 0x40, 0x50],
        ct_is_matched=[False, False, False, True],
        # nodes: root v0=0, root v1=1, U1=2, U2=3, U3=4, M=5.
        # pce_called_idx is SECTION-LOCAL: each caller's single slot is 0.
        var_n_calls=[1, 0, 1, 1, 1, 0],
        pce_called_idx=[0, 0, 0, 0],
        pce_section_variant_index=[0, 0, 0, 0],
    )
    m_node = int(cols.var_offsets[4])  # M's node (section idx 4)
    on_cap2 = _emitted_callee_nodes(_run(cols, offs, unmatched_inline=True, depth=2))
    on_cap3 = _emitted_callee_nodes(_run(cols, offs, unmatched_inline=True, depth=3))
    assert m_node not in on_cap2[0]
    assert m_node in on_cap3[0]
