"""Byte-identity gate: matched-arm combined + many-variant fixtures.

THE highest-risk case: ``build_combined_fixture`` carries a MULTI-VARIANT
section (``multi_fn``, 4 variants) AND an ASYMMETRIC call graph
(``caller_fn`` -> ``callee_fn`` with disjoint vkey sets -> a
MISSING_VARIANT_INDEX splice edge), so subset-vs-full FLAG-A inclusion
differs across rows. The straddler cut is exercised by a tight
``context_len``; a multi-section batch by pointing at several sections.
The many-variant single-section fixture exercises FLAG-A subset-vs-full
divergence under <4-of-N sampling. Both paths assert full byte-identity
via the shared :mod:`._byte_identity_harness`.
"""

from __future__ import annotations

import pytest

from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
)

from ._byte_identity_harness import (
    _assert_token_identity,
    _nonempty_matched_idxs,
    _prepare,
    _run_both,
)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_byte_identity_combined_fixture_full_depth(seed, tmp_path):
    """Whole combined corpus (multi-variant + asymmetric splice), a wide
    context so no straddler -- every row's full body must match."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=4096,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_byte_identity_combined_straddler_cut(seed, tmp_path):
    """A tight context_len forces a mid-function straddler cut on the
    longer rows -- the cut column must land identically."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=12,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)


@pytest.mark.parametrize("seed", [0, 5])
def test_byte_identity_many_variant_section(seed, tmp_path):
    """The many-variant single section -- subset sampling of <4 of the
    variants exercises FLAG-A subset-vs-full divergence per row."""
    base = _prepare(build_many_variant_section_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=2,
        context_len=512,
        max_depth=2,
        seed=seed,
    )
    _assert_token_identity(ref, new)


def test_byte_identity_depth_zero_roots_only(tmp_path):
    """depth 0: every row is root-only (no splice) -- the prefix + root
    body + self-token must match with no callee assembly."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=256,
        max_depth=0,
        seed=0,
    )
    _assert_token_identity(ref, new)
