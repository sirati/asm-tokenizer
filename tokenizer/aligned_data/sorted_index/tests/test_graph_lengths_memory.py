"""Memory + scale regression guards for the graph length compute.

The edge resolver historically materialised the dense ``variants x
call_targets`` candidate product (parallel int64 arrays) -- tens of GB
on six-figure-section catalogs (the z3 OOM, exit 137 under a 20 GB
cgroup). The factored recurrence (see :mod:`.._graph_lengths`) must
keep every allocation bounded by the catalog's on-disk table sizes:

* the allocation probe builds a many-variant synthetic catalog whose
  dense product is ~100x its linear structures, injects a flat synthetic
  body-length array (so only resolver + DP allocations are traced), and
  asserts the tracemalloc peak stays far below ONE dense candidate
  array;
* the structural guard pins the resolver's output arrays to their
  contract sizes (``total_cts`` / ``<= total first per-call entries``);
* the scale smoke runs a few-thousand-section catalog through all four
  production depths and asserts sane wall time.

Synthetic catalogs are built directly as :class:`ColumnarSections`
(public columnar API; plain arrays) -- the on-disk writer path is
oracle-tested in ``test_graph_lengths.py`` and would dominate the
runtime here without adding coverage of the compute.
"""

from __future__ import annotations

import gc
import time
import tracemalloc

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections
from tokenizer.aligned_data.sorted_index._graph_lengths import (
    compute_node_lengths,
)
from tokenizer.aligned_data.sorted_index._graph_lengths._adjacency import (
    LiveNodeAdjacency,
)


def _csr(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def _forward_dag_catalog(
    n_sections: int, n_vars: int, n_cts: int, n_calls: int
) -> tuple:
    """Synthetic catalog: section ``s``'s slot ``j`` calls ``s+1+j``.

    Forward edges only (a DAG -- the cycle replay never triggers, so
    the probes time/trace the resolver + DP, not the exact DFS). Every
    section has ``n_vars`` variants; each variant carries per-call
    entries for its section's first ``n_calls`` slots with its own
    rank stamped as ``J`` (in range for every callee).
    """
    secs = np.arange(n_sections, dtype=np.int64)
    n_ct_per = np.minimum(n_cts, np.maximum(0, n_sections - 1 - secs))
    ct_offsets = _csr(n_ct_per)
    total_cts = int(ct_offsets[-1])
    sec_of_ct = np.repeat(secs, n_ct_per)
    j_within = np.arange(total_cts, dtype=np.int64) - ct_offsets[sec_of_ct]
    callee = sec_of_ct + 1 + j_within

    section_offsets = (secs + 1) * 16

    n_variants = np.full(n_sections, n_vars, dtype=np.int64)
    var_offsets = _csr(n_variants)
    total_vars = int(var_offsets[-1])
    sec_of_var = np.repeat(secs, n_variants)
    rank_of_var = np.arange(total_vars, dtype=np.int64) - var_offsets[
        sec_of_var
    ]

    var_n_calls = np.minimum(n_calls, n_ct_per[sec_of_var])
    pce_offsets = _csr(var_n_calls)
    total_pce = int(pce_offsets[-1])
    var_of_pce = np.repeat(
        np.arange(total_vars, dtype=np.int64), var_n_calls
    )
    pce_called = (
        np.arange(total_pce, dtype=np.int64) - pce_offsets[var_of_pce]
    )

    cols = ColumnarSections(
        function_name_ptr=secs.astype(np.uint32),
        is_duplicated=np.zeros(secs.size, dtype=bool),
        n_call_targets=n_ct_per,
        n_variants=n_variants,
        ct_offsets=ct_offsets,
        ct_function_name_ptr=np.zeros(total_cts, dtype=np.uint32),
        ct_function_section_ptr=section_offsets[callee].astype(np.uint32),
        ct_type=np.full(
            total_cts, int(CallTargetType.LOCAL), dtype=np.uint8
        ),
        ct_is_matched=np.ones(total_cts, dtype=bool),
        var_offsets=var_offsets,
        var_ref_offset=np.arange(total_vars, dtype=np.uint32),
        var_data_offset_shifted=np.arange(
            1, total_vars + 1, dtype=np.uint32
        ),
        var_n_calls=var_n_calls,
        pce_offsets=pce_offsets,
        pce_called_idx=pce_called.astype(np.uint16),
        pce_section_variant_index=rank_of_var[var_of_pce].astype(np.uint16),
    )
    return cols, section_offsets


def _synthetic_body_lengths(cols: ColumnarSections) -> np.ndarray:
    """Flat synthetic per-variant body length for an injected catalog.

    The synthetic catalog has no ``_data.bin``; the probes target the
    resolver + DP, whose inputs are the catalog columns plus SOME int64
    body-length vector (the build now consumes the realized-length
    sidecar, so the BFS takes this array directly).
    """
    return np.ones(int(cols.var_n_calls.size), dtype=np.int64)


def test_resolver_memory_stays_linear() -> None:
    # 800 sections x 100 variants x ~175 call targets: the dense
    # candidate product is ~14M edges (~110 MB per int64 array; the
    # historical resolver pinned SEVERAL in parallel), while the
    # catalog's linear tables stay small (total_cts ~140k, total
    # per-call entries ~160k -- the resolver's working set is a
    # bounded handful of columns of THOSE sizes).
    cols, section_offsets = _forward_dag_catalog(
        n_sections=800, n_vars=100, n_cts=200, n_calls=2
    )
    dense_product = int(
        (cols.n_variants * cols.n_call_targets).sum()
    )
    dense_array_bytes = dense_product * 8
    assert dense_array_bytes > 100 * 2**20  # the contrast is real

    body = _synthetic_body_lengths(cols)
    gc.collect()
    tracemalloc.start()
    try:
        compute_node_lengths(cols, section_offsets, body, [1])
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Sanity: numpy allocations ARE traced (a few linear arrays alone
    # exceed 1 MB), so a silently-untracked run cannot pass.
    assert peak > 2**20
    # The teeth: far below ONE dense candidate array. Any reintroduced
    # array proportional to total_vars * avg_call_targets trips this.
    assert peak < dense_array_bytes // 3, (
        f"resolver+DP peak {peak / 2**20:.1f} MiB suggests a dense "
        f"variants x call_targets allocation "
        f"({dense_array_bytes / 2**20:.1f} MiB per array)"
    )


def test_live_adjacency_children_bounded_by_node_calls() -> None:
    # The live adjacency builds only the offset->idx map at construction
    # (no graph-wide edge product); each node's child list is bounded by
    # that node's directly-called slot count.
    cols, section_offsets = _forward_dag_catalog(
        n_sections=50, n_vars=8, n_cts=10, n_calls=4
    )
    sec_of_var = np.repeat(
        np.arange(cols.n_variants.size, dtype=np.int64), cols.n_variants
    )
    adjacency = LiveNodeAdjacency(cols, section_offsets, sec_of_var)
    for node in range(int(cols.var_offsets[-1])):
        children, child_secs, child_types, child_matched = adjacency(node)
        assert (
            children.size
            == child_secs.size
            == child_types.size
            == child_matched.size
        )
        assert children.size <= int(cols.var_n_calls[node])
        # Children are valid flat variant indices.
        assert bool((children >= 0).all())
        assert bool((children < cols.var_offsets[-1]).all())


def test_scale_smoke_few_thousand_sections() -> None:
    # Production depth set on a few-thousand-section catalog; the
    # bound is deliberately loose (CI boxes under load) -- the dense
    # resolver at this shape would page-thrash long past it.
    cols, section_offsets = _forward_dag_catalog(
        n_sections=4000, n_vars=10, n_cts=40, n_calls=8
    )
    body = _synthetic_body_lengths(cols)
    t0 = time.perf_counter()
    results = compute_node_lengths(
        cols, section_offsets, body, [0, 1, 2, 3]
    )
    elapsed = time.perf_counter() - t0
    assert set(results) == {0, 1, 2, 3}
    assert all(
        arr.size == int(cols.var_offsets[-1]) for arr in results.values()
    )
    assert elapsed < 30.0, f"scale smoke took {elapsed:.1f}s"
