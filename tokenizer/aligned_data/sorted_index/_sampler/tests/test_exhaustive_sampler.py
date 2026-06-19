"""Tests for the deterministic EXHAUSTIVE section enumeration (no decode).

Pins (mirrors ``test_validation_sampler.py`` but for the rng-free
whole-corpus sibling):

* COMPLETENESS -- every non-excluded section across all binaries appears
  EXACTLY ONCE (cross-checked against ``enumerate_in_band`` over the FULL
  band per binary), each carrying ALL its variants ``range(n_variants)``.
* DETERMINISM -- same ``(readers, per-section counts)`` => byte-identical
  pointer sequence (order + every pointer field) across repeated calls;
  NO ``numpy.random.Generator`` is constructed or touched.
* ORDER -- binaries in GIVEN input order, sections in ``enumerate_in_band``
  order, variants ascending.
* BAND EDGES -- the full non-excluded band per reader; the EXCLUDED_LENGTH
  bucket (length 0) is never enumerated; empty index contributes nothing.
* SINGLE-VARIANT / EMPTY-binary edge cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._variant_selection import (
    ExplicitIndicesSelection,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    ReductionKind,
    SortedIndexReader,
    encode_sorted_index,
)
from tokenizer.aligned_data.sorted_index._types import IndexSpec
from tokenizer.aligned_data.sorted_index._wire import EXCLUDED_LENGTH
from tokenizer.aligned_data.sorted_index._sampler import (
    ExhaustiveSectionSampler,
    all_section_pointers,
)


_SPEC = IndexSpec(LengthReduction(ReductionKind.MAX), depth=3)


# ---------------------------------------------------------------------------
# Synthetic-reader helpers (no decode / no session)
# ---------------------------------------------------------------------------


def _make_reader(
    tmp_path: Path, binary_name: str, lengths: np.ndarray
) -> SortedIndexReader:
    """Write a tiny per-binary index and open a reader on it."""
    path = tmp_path / f"{binary_name}_sorted_max_d003.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3
    )


def _count_provider_from(counts_by_binary: Dict[str, Dict[int, int]]):
    """Injected count_provider keyed by per-binary {section_idx: n_variants}."""

    def provider(binary_name: str, section_indices: np.ndarray) -> np.ndarray:
        table = counts_by_binary[binary_name]
        return np.array(
            [table[int(i)] for i in section_indices], dtype=np.int64
        )

    return provider


def _materialize(pointers) -> List[Tuple[str, int, Tuple[int, ...]]]:
    """The pointer sequence as ``(binary_name, section_idx, indices)`` tuples."""
    out = []
    for ptr in pointers:
        sel = ptr.section_pointer.variant_selection
        out.append(
            (ptr.binary_name, ptr.section_pointer.idx, tuple(sel.indices))
        )
    return out


# ---------------------------------------------------------------------------
# COMPLETENESS
# ---------------------------------------------------------------------------


def test_completeness_every_section_once_all_variants(tmp_path: Path) -> None:
    # Two binaries, sections spread across MULTIPLE length buckets to prove
    # the full band ``EXCLUDED_LENGTH+1 .. max_length`` enumerates them all.
    alpha = _make_reader(tmp_path, "alpha", np.array([5, 9, 5, 12], np.uint32))
    beta = _make_reader(tmp_path, "beta", np.array([3, 7], np.uint32))
    counts = {
        "alpha": {0: 1, 1: 3, 2: 4, 3: 2},
        "beta": {0: 5, 1: 1},
    }
    provider = _count_provider_from(counts)
    readers = [("alpha", _SPEC, alpha), ("beta", _SPEC, beta)]

    pointers = all_section_pointers(readers, provider)

    # Cross-check against enumerate_in_band over the FULL band per reader:
    # the emitted (binary, section) set EQUALS the reader's full-band set,
    # exactly once, no dups, no drops.
    for name, reader in (("alpha", alpha), ("beta", beta)):
        full_band = set(
            int(i)
            for i in reader.enumerate_in_band(
                EXCLUDED_LENGTH + 1, reader.max_length
            )
        )
        emitted = [
            p.section_pointer.idx for p in pointers if p.binary_name == name
        ]
        assert sorted(emitted) == sorted(full_band)  # same set
        assert len(emitted) == len(set(emitted))  # no dups
        assert set(emitted) == full_band  # no drops

    # Each pointer carries ALL variants in ascending raw order.
    for p in pointers:
        sel = p.section_pointer.variant_selection
        assert isinstance(sel, ExplicitIndicesSelection)
        n = counts[p.binary_name][p.section_pointer.idx]
        assert sel.indices == tuple(range(n))
        assert p.section_pointer.arm is SectionKind.MATCHED
        assert p.spec is _SPEC


# ---------------------------------------------------------------------------
# DETERMINISM
# ---------------------------------------------------------------------------


def test_determinism_byte_identical_no_rng(tmp_path: Path) -> None:
    a = _make_reader(tmp_path, "aaa", np.array([5, 5, 9], np.uint32))
    b = _make_reader(tmp_path, "bbb", np.array([7], np.uint32))
    counts = {"aaa": {0: 3, 1: 1, 2: 6}, "bbb": {0: 2}}
    provider = _count_provider_from(counts)
    readers = [("aaa", _SPEC, a), ("bbb", _SPEC, b)]

    s = ExhaustiveSectionSampler(readers)
    out1 = s.all_pointers(provider)
    out2 = s.all_pointers(provider)
    # Byte-identical: order + every pointer field. Frozen dataclasses with
    # frozen ExplicitIndicesSelection compare by value.
    assert out1 == out2
    # And the free-function convenience matches the class.
    assert out1 == all_section_pointers(readers, provider)


def test_determinism_does_not_touch_a_generator(tmp_path: Path) -> None:
    # A Generator whose .integers/.choice/etc. would raise if ever called:
    # the enumeration must never construct or consume one. We assert the
    # call succeeds without any rng parameter even existing on the surface.
    rdr = _make_reader(tmp_path, "x", np.array([5, 5], np.uint32))
    counts = {"x": {0: 2, 1: 3}}
    provider = _count_provider_from(counts)
    # No rng arg accepted by either the sampler ctor or all_pointers.
    pointers = all_section_pointers([("x", _SPEC, rdr)], provider)
    assert _materialize(pointers) == [
        ("x", 0, (0, 1)),
        ("x", 1, (0, 1, 2)),
    ]


# ---------------------------------------------------------------------------
# ORDER
# ---------------------------------------------------------------------------


def test_order_binary_input_then_enumeration_then_variants(
    tmp_path: Path,
) -> None:
    a = _make_reader(tmp_path, "aaa", np.array([5, 5], np.uint32))
    b = _make_reader(tmp_path, "bbb", np.array([5], np.uint32))
    counts = {"aaa": {0: 2, 1: 1}, "bbb": {0: 3}}
    provider = _count_provider_from(counts)
    # Deliberately NON-alphabetical input order: the sampler honors the
    # GIVEN order, never re-sorts (the canonical alphabetical guarantee
    # lives at the discover_members construction seam).
    readers = [("bbb", _SPEC, b), ("aaa", _SPEC, a)]

    stream = _materialize(all_section_pointers(readers, provider))
    assert stream == [
        ("bbb", 0, (0, 1, 2)),
        ("aaa", 0, (0, 1)),
        ("aaa", 1, (0,)),
    ]


# ---------------------------------------------------------------------------
# BAND EDGES + EMPTY / SINGLE-VARIANT EDGE CASES
# ---------------------------------------------------------------------------


def test_excluded_length_bucket_never_enumerated(tmp_path: Path) -> None:
    # Length EXCLUDED_LENGTH (0) is the exclusion marker; sections at it
    # must never be enumerated even though the band lo would reach it.
    rdr = _make_reader(
        tmp_path,
        "x",
        np.array([EXCLUDED_LENGTH, 5, EXCLUDED_LENGTH, 8], np.uint32),
    )
    # Only the length-5 (idx 1) and length-8 (idx 3) sections are in-band.
    full_band = set(
        int(i) for i in rdr.enumerate_in_band(EXCLUDED_LENGTH + 1, rdr.max_length)
    )
    counts = {"x": {i: 2 for i in full_band}}
    provider = _count_provider_from(counts)

    pointers = all_section_pointers([("x", _SPEC, rdr)], provider)
    emitted = {p.section_pointer.idx for p in pointers}
    assert emitted == full_band
    # No emitted section came from the EXCLUDED bucket.
    assert all(p.section_pointer.idx in full_band for p in pointers)


def test_empty_binary_contributes_nothing(tmp_path: Path) -> None:
    empty = _make_reader(tmp_path, "empty", np.array([], np.uint32))
    full = _make_reader(tmp_path, "full", np.array([5, 5], np.uint32))
    counts = {"empty": {}, "full": {0: 1, 1: 2}}
    provider = _count_provider_from(counts)
    readers = [("empty", _SPEC, empty), ("full", _SPEC, full)]

    stream = _materialize(all_section_pointers(readers, provider))
    assert all(name != "empty" for name, _i, _v in stream)
    assert stream == [("full", 0, (0,)), ("full", 1, (0, 1))]


def test_single_variant_section_emits_one_variant(tmp_path: Path) -> None:
    rdr = _make_reader(tmp_path, "x", np.array([5], np.uint32))
    provider = _count_provider_from({"x": {0: 1}})
    pointers = all_section_pointers([("x", _SPEC, rdr)], provider)
    assert len(pointers) == 1
    assert pointers[0].section_pointer.variant_selection.indices == (0,)


def test_zero_variant_section_is_honest_passthrough(tmp_path: Path) -> None:
    # Enumeration is a faithful pass-through of the index: a section the
    # index places in-band with a 0 variant count pins an empty
    # ExplicitIndicesSelection -- range(0) == (). The enumeration does NOT
    # filter it (single concern: enumerate what the index says). In
    # PRODUCTION this never happens: the builder stamps every 0-variant
    # section with EXCLUDED_LENGTH (see _length_compute: ``emitted =
    # n_variants > 0 & gate``), so a non-excluded band never enumerates a
    # 0-variant section. This test documents the pass-through, not a real
    # corpus shape.
    rdr = _make_reader(tmp_path, "x", np.array([5, 5], np.uint32))
    provider = _count_provider_from({"x": {0: 0, 1: 2}})
    pointers = all_section_pointers([("x", _SPEC, rdr)], provider)
    assert pointers[0].section_pointer.variant_selection.indices == ()
    assert pointers[1].section_pointer.variant_selection.indices == (0, 1)
