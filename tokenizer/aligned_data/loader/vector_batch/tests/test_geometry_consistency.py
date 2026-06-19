"""Read-side catalog<->geometry stale-sidecar guard regression tests.

Pins the fix for ml-project's cross-depth ``IndexError``: a
realized-geometry sidecar STALE relative to a rebuilt (longer)
``_sections.bin`` catalog must be caught LOUDLY at handle-open
(:func:`...session_handles.open_vector_batch_handles` ->
:func:`..._geometry_consistency.check_geometry_matches_catalog`), not
surface as a bare ``IndexError`` deep in the prepass's node-indexed axis
gather (:func:`..._geometry._flatten_emission`, the ``body_axis[node]``
gather).

Three teeth:

* the guard FIRES on a catalog<->geometry mismatch, with a loud,
  actionable message naming the binary + arm + both counts;
* WITHOUT the guard, the SAME truncated geometry reaches
  :func:`compute_batch_geometry` and raises the bare ``IndexError`` at
  the node-indexed axis gather -- so the guard is converting a real
  latent crash, and the test fails if the guard is removed;
* the guard is a NO-OP on a consistent corpus (the synthetic fixture's
  own geometry passes unchanged).
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.vector_batch import compute_batch_geometry
from tokenizer.aligned_data.loader.vector_batch._geometry_consistency import (
    check_geometry_matches_catalog,
)
from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    open_vector_batch_handles,
)
from tokenizer.aligned_data.realized_lengths import RealizedGeometryReader
from tokenizer.aligned_data.realized_lengths._geometry_format import (
    GEOMETRY_MATCHED_ARM,
    read_geometry_pair,
    write_geometry_pair,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
)

from ._byte_identity_harness import _BINARY_NAME, _prepare
from ._synthetic import build_synthetic_corpus


def _truncated_geometry(corpus, keep: int) -> RealizedGeometryReader:
    """An internally-consistent geometry reader carrying only ``keep`` nodes.

    Mirrors a real STALE ``_realized.bin``: its own CSR terminator matches
    its (shorter) axes, so the reader's internal self-check
    (``csr[-1] == body_lengths.size``) passes -- exactly the case that
    slips through every existing guard. ``keep`` truncates all three axes
    and rebuilds a per-section CSR clamped to ``keep`` so no section's
    bound exceeds the carried axis length.
    """
    full_csr = np.asarray(corpus.var_offsets, dtype=np.int64)
    trunc_csr = np.minimum(full_csr, keep).astype(np.uint32)
    return RealizedGeometryReader(
        body_lengths=corpus.body_len[:keep].astype(np.uint32),
        id_counts=corpus.id_count[:keep].astype(np.uint32),
        value_counts=corpus.value_count[:keep].astype(np.uint32),
        csr=trunc_csr,
    )


def test_guard_fires_on_truncated_geometry():
    """A geometry carrying fewer nodes than the catalog raises loudly."""
    corpus = build_synthetic_corpus()
    total_vars = int(corpus.cols.var_offsets[-1])
    keep = max(1, total_vars // 4)
    stale = _truncated_geometry(corpus, keep)

    with pytest.raises(ValueError) as excinfo:
        check_geometry_matches_catalog(
            corpus.cols,
            stale,
            binary_name="zbin",
            arm_name="matched",
        )
    msg = str(excinfo.value)
    # Loud + actionable: names the binary, the arm, BOTH counts, and the
    # regenerate path (the build-side contract's style).
    assert "stale realized-geometry sidecar" in msg
    assert "zbin" in msg
    assert "matched" in msg
    assert str(total_vars) in msg
    assert str(keep) in msg
    assert "re-run the realized-geometry generator" in msg


def test_guard_fires_on_section_count_mismatch():
    """A geometry whose per-node total matches but section count differs.

    Constructed so the variant axis length equals the catalog's variant
    total (the first check passes) but the CSR carries a different section
    count -- proving the section-count arm is independently load-bearing.
    """
    corpus = build_synthetic_corpus()
    total_vars = int(corpus.cols.var_offsets[-1])
    n_sections = int(corpus.cols.n_variants.size)
    # Same node count, but collapse every variant into ONE section: the CSR
    # is [0, total_vars] -> n_sections == 1 != the catalog's section count.
    one_section_csr = np.array([0, total_vars], dtype=np.uint32)
    geom = RealizedGeometryReader(
        body_lengths=corpus.body_len.astype(np.uint32),
        id_counts=corpus.id_count.astype(np.uint32),
        value_counts=corpus.value_count.astype(np.uint32),
        csr=one_section_csr,
    )
    assert int(geom.body_lengths.size) == total_vars  # first check passes
    assert int(geom.n_sections) != n_sections  # second check must fire

    with pytest.raises(ValueError) as excinfo:
        check_geometry_matches_catalog(
            corpus.cols, geom, binary_name="zbin", arm_name="matched"
        )
    msg = str(excinfo.value)
    assert "sections" in msg
    assert str(n_sections) in msg
    assert "re-run the realized-geometry generator" in msg


def test_truncated_geometry_crashes_compute_without_guard():
    """Teeth: the SAME truncation reaches compute_batch_geometry and raises
    the bare IndexError at the node-indexed axis gather.

    This is the latent crash the guard converts. The roots stay low-index
    (node 0/1) but the depth-2 BFS splices high-index callee nodes (B=3,
    C=4) that exceed the truncated axis -> the ``body_axis[node]`` gather
    in ``_flatten_emission`` raises ``IndexError``. If this stopped
    reproducing, the guard would be guarding nothing.
    """
    corpus = build_synthetic_corpus()
    total_vars = int(corpus.cols.var_offsets[-1])
    keep = max(1, total_vars // 4)
    stale = _truncated_geometry(corpus, keep)

    with pytest.raises(IndexError):
        compute_batch_geometry(
            cols=corpus.cols,
            section_offsets=corpus.section_offsets,
            geometry=stale,
            variants_u8=corpus.variants_u8,
            root_sections=np.zeros(2, dtype=np.int64),
            root_sampled_variants=np.array([0, 1], dtype=np.int64),
            root_groups=np.zeros(2, dtype=np.int64),
            seq_len=100,
            max_depth=3,
        )


def test_open_handles_raises_on_stale_on_disk_geometry(tmp_path):
    """Boundary wiring: open_vector_batch_handles trips the guard on a real
    corpus whose on-disk geometry pair is stale.

    Builds a real corpus + its RLG3 sidecars, then OVERWRITES the matched
    geometry pair with an internally-consistent-but-shorter one (its own
    CSR terminator matches its truncated axes, so the reader's self-check
    passes -- exactly a real stale ``_realized.bin``). The handle-open path
    must raise the loud stale-sidecar error. If the guard call is removed
    from ``open_vector_batch_handles``, this open succeeds and the test
    fails -- the guard's wiring is load-bearing here.
    """
    base = _prepare(build_combined_fixture, tmp_path)
    arm = GEOMETRY_MATCHED_ARM
    geometry_path = arm.geometry_path(base, _BINARY_NAME)
    index_path = arm.index_path(base, _BINARY_NAME)

    body, ids, values, csr = read_geometry_pair(geometry_path, index_path)
    n_full = int(body.size)
    keep = max(1, n_full // 4)
    assert keep < n_full, "fixture must carry >1 node to truncate"
    # Copy out of the mmap views BEFORE overwriting the backing files --
    # writing a file under its own live mmap is a SIGBUS.
    body = np.array(body[:keep])
    ids = np.array(ids[:keep])
    values = np.array(values[:keep])
    trunc_csr = np.minimum(np.asarray(csr, dtype=np.int64), keep)
    write_geometry_pair(
        geometry_path,
        index_path,
        body_lengths=body,
        id_counts=ids,
        value_counts=values,
        csr_offsets=trunc_csr,
    )

    with pytest.raises(ValueError, match="stale realized-geometry sidecar"):
        open_vector_batch_handles(base, _BINARY_NAME)


def test_guard_noop_on_consistent_corpus():
    """The fixture's OWN (consistent) geometry passes the guard unchanged."""
    corpus = build_synthetic_corpus()
    # Returns None, raises nothing -- detection-only, zero output change.
    assert (
        check_geometry_matches_catalog(
            corpus.cols,
            corpus.geometry,
            binary_name="zbin",
            arm_name="matched",
        )
        is None
    )
