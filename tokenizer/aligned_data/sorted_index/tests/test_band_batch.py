"""Band threading + session/decode split tests for the batch helper.

Two concerns covered against a real tiny two-binary corpus (built via
the production writers through ``build_combined_fixture``):

* **Band threading**: :func:`open_length_bucketed_batch` accepts a
  ``band=(lo, hi)`` keyword that samples sections whose length key lies
  in ``[lo, hi]`` rather than at the exact ``target_length`` bucket --
  including the motivating case where the exact bucket is EMPTY but the
  band is populated. ``band=None`` reproduces the pre-band behaviour.
* **Session / decode split**: :func:`decode_pointer_batch` is the
  session-agnostic core. It takes a mapping of ALREADY-OPEN sessions,
  raises when a pointer names an absent session, and -- driven with the
  same seeded RNG state -- produces byte-identical arrays to
  :func:`open_length_bucketed_batch`.

The synthetic sorted index assigns each sampleable section index a
distinct length key so a band can straddle several buckets while the
exact ``target_length`` bucket is empty. Section index 0 in the
combined fixture is the 0-variant ``func_zero`` -- it is parked at an
out-of-band length and never sampled.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Dict, Iterator

import numpy as np
import pytest

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    MultiBinarySortedIndexSampler,
    ReductionKind,
    SortedIndexReader,
    decode_pointer_batch,
    encode_sorted_index,
    open_length_bucketed_batch,
)

from .fixtures import build_combined_fixture, make_test_vocab_manager


_BINARY_NAME_A = "binA"
_BINARY_NAME_B = "binB"

# Sampleable section indices in the combined fixture (index 0 is the
# 0-variant ``func_zero`` -- never sampled). Each gets a distinct length
# so a band can straddle several buckets.
_SECTION_LENGTHS: Dict[int, int] = {1: 10, 2: 20, 3: 30, 4: 40}
# A length far outside any band the tests probe -- parks func_zero so it
# never lands in a sampled band.
_PARKED_LENGTH = 1000


# ---------------------------------------------------------------------------
# Fixture builders (mirror test_batch_helper.py)
# ---------------------------------------------------------------------------


def _build_multi_binary_fixture(tmp_path: Path) -> Path:
    """Lay down two combined-corpus binaries under one memmap dir."""
    memmap_dir = tmp_path / "memmap"
    memmap_dir.mkdir()
    for binary_name in (_BINARY_NAME_A, _BINARY_NAME_B):
        scratch = tmp_path / f"scratch_{binary_name}"
        scratch.mkdir()
        combined_base = build_combined_fixture(scratch)
        for entry in combined_base.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.startswith("sortbin"):
                continue
            new_name = binary_name + entry.name[len("sortbin"):]
            (memmap_dir / new_name).write_bytes(entry.read_bytes())
    return memmap_dir


def _write_banded_sorted_index(memmap_dir: Path, binary_name: str) -> Path:
    """Lay down a sorted index assigning each section a distinct length.

    Sampleable sections (1..4) land at :data:`_SECTION_LENGTHS`; the
    0-variant ``func_zero`` (index 0) is parked at
    :data:`_PARKED_LENGTH`. The result is a multi-bucket index where a
    band can cover several distinct length keys.
    """
    num_sections = max(_SECTION_LENGTHS) + 1
    lengths = np.full(num_sections, _PARKED_LENGTH, dtype=np.uint32)
    for idx, length in _SECTION_LENGTHS.items():
        lengths[idx] = length
    path = memmap_dir / f"{binary_name}_sorted_max_d003.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return path


def _open_sampler(memmap_dir: Path) -> MultiBinarySortedIndexSampler:
    readers = {}
    for name in (_BINARY_NAME_A, _BINARY_NAME_B):
        path = _write_banded_sorted_index(memmap_dir, name)
        readers[name] = SortedIndexReader(
            path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
        )
    return MultiBinarySortedIndexSampler(readers)


def _make_session_factory(memmap_dir: Path):
    vocab_manager = make_test_vocab_manager()
    @contextmanager
    def session_factory(binary_name: str) -> Iterator[BinarySession]:
        dataset = BinaryDataset(memmap_dir, binary_name, vocab_manager=vocab_manager)
        with dataset.open_session() as session:
            yield session
    return session_factory


# ---------------------------------------------------------------------------
# Deliverable-2.1 -- band threading end-to-end
# ---------------------------------------------------------------------------


def test_band_threading_empty_exact_bucket_nonempty_band(
    tmp_path: Path,
) -> None:
    """A target_length whose exact bucket is EMPTY still yields a batch
    when its band is populated -- the motivating band case.

    Sections sit at lengths {10, 20, 30, 40}. ``target_length=25`` has
    an empty exact bucket, but band ``[20, 30]`` covers sections 2 (len
    20) and 3 (len 30) across both binaries. The sampled pointers' on-
    disk section indices (whose length keys are :data:`_SECTION_LENGTHS`)
    must all fall in the band, and the helper must produce a valid batch.
    """
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    sampler = _open_sampler(memmap_dir)
    factory = _make_session_factory(memmap_dir)

    # Exact bucket at 25 is empty; the band [20, 30] is not.
    assert sampler.count_at(25) == 0
    assert sampler.count_in_band(20, 30) > 0

    # An un-banded call at target_length=25 must raise (empty pool).
    with pytest.raises(ValueError, match="empty sampler pool at"):
        open_length_bucketed_batch(
            factory, sampler,
            target_length=25, batch_size=4,
            context_len=16, num_variants_per_section=2,
            max_depth=2, rng=np.random.default_rng(0),
        )

    # The on-disk section indices whose length keys land in [20, 30].
    band_section_indices = {
        idx for idx, length in _SECTION_LENGTHS.items()
        if 20 <= length <= 30
    }
    assert band_section_indices == {2, 3}

    # Sample pointers directly to check every drawn section's length key
    # is inside the band (the column-0 section_idx in the decode result
    # is a per-binary local position, NOT the on-disk index, so the band
    # constraint is asserted at the pointer layer where it lives).
    pointers = sampler.sample_section_pointers(
        target_length=25, count=8,
        rng=np.random.default_rng(99), band=(20, 30),
    )
    assert pointers
    for p in pointers:
        idx = p.section_pointer.idx
        assert idx in band_section_indices, (
            f"banded draw escaped the band: section idx {idx} has length "
            f"key {_SECTION_LENGTHS.get(idx)}, band [20, 30]"
        )

    # The banded helper call succeeds at the empty-exact-bucket target.
    rng = np.random.default_rng(42)
    batch_size = 4
    num_variants_per_section = 2
    result = open_length_bucketed_batch(
        factory, sampler,
        target_length=25, batch_size=batch_size,
        context_len=16, num_variants_per_section=num_variants_per_section,
        max_depth=2, rng=rng, band=(20, 30),
    )

    expected_rows = batch_size * num_variants_per_section
    inner = result.inner
    assert inner.tokens.shape == (expected_rows, 16)
    assert result.binary_id_per_row.shape == (expected_rows,)
    assert result.binary_names == sorted([_BINARY_NAME_A, _BINARY_NAME_B])


def test_band_empty_pool_message_mentions_band(tmp_path: Path) -> None:
    """An empty band raises a ValueError naming the band."""
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    sampler = _open_sampler(memmap_dir)
    factory = _make_session_factory(memmap_dir)
    with pytest.raises(ValueError, match=r"empty sampler pool in band"):
        open_length_bucketed_batch(
            factory, sampler,
            target_length=0, batch_size=2,
            context_len=16, num_variants_per_section=2,
            max_depth=2, rng=np.random.default_rng(0),
            band=(500, 600),  # no section length lands here.
        )


# ---------------------------------------------------------------------------
# Deliverable-2.2 -- band=None back-compat
# ---------------------------------------------------------------------------


def test_band_none_back_compat(tmp_path: Path) -> None:
    """``band=None`` reproduces the exact-bucket behaviour.

    Sampling at the populated ``target_length=20`` (on-disk section 2)
    without a band yields a valid batch; the drawn pointers all trace to
    the exact bucket's lone member.
    """
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    sampler = _open_sampler(memmap_dir)
    factory = _make_session_factory(memmap_dir)

    # Exact bucket at 20 holds only on-disk section index 2 per binary.
    pointers = sampler.sample_section_pointers(
        target_length=20, count=4, rng=np.random.default_rng(5),
    )
    assert pointers
    assert all(p.section_pointer.idx == 2 for p in pointers)

    batch_size = 2
    num_variants_per_section = 2
    rng = np.random.default_rng(7)
    result = open_length_bucketed_batch(
        factory, sampler,
        target_length=20, batch_size=batch_size,
        context_len=16, num_variants_per_section=num_variants_per_section,
        max_depth=2, rng=rng,  # band defaults to None.
    )

    expected_rows = batch_size * num_variants_per_section
    inner = result.inner
    assert inner.tokens.shape == (expected_rows, 16)
    assert inner.tokens.dtype == np.uint16
    assert inner.identity_row_offsets.shape == (expected_rows + 1,)
    assert inner.number_row_offsets.shape == (expected_rows + 1,)
    assert result.binary_id_per_row.shape == (expected_rows,)
    assert result.binary_names == sorted([_BINARY_NAME_A, _BINARY_NAME_B])


# ---------------------------------------------------------------------------
# Deliverable-2.4 -- decode_pointer_batch contract + array equality
# ---------------------------------------------------------------------------


def test_decode_pointer_batch_missing_session_raises(tmp_path: Path) -> None:
    """A pointer naming a binary absent from ``sessions`` is a hard error."""
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    sampler = _open_sampler(memmap_dir)
    factory = _make_session_factory(memmap_dir)

    # Sample a band batch that draws from both binaries.
    pointers = sampler.sample_section_pointers(
        target_length=0, count=8,
        rng=np.random.default_rng(123), band=(10, 40),
    )
    assert {p.binary_name for p in pointers} == {
        _BINARY_NAME_A, _BINARY_NAME_B,
    }

    # Open ONLY binA's session; binB pointers have no session.
    with ExitStack() as stack:
        sessions: Dict[str, BinarySession] = {
            _BINARY_NAME_A: stack.enter_context(factory(_BINARY_NAME_A)),
        }
        with pytest.raises(ValueError, match=_BINARY_NAME_B):
            decode_pointer_batch(
                sessions, pointers,
                context_len=16, num_variants_per_section=2,
                max_depth=2, rng=np.random.default_rng(0),
            )


def test_decode_pointer_batch_empty_pointers_raises(tmp_path: Path) -> None:
    """An empty pointer batch raises (no work to decode)."""
    with pytest.raises(ValueError, match="empty pointer batch"):
        decode_pointer_batch(
            {}, [],
            context_len=16, num_variants_per_section=2,
            max_depth=2, rng=np.random.default_rng(0),
        )


def test_decode_pointer_batch_matches_open_length_bucketed_batch(
    tmp_path: Path,
) -> None:
    """``decode_pointer_batch`` IS the core of the bucketed helper.

    Both paths start from identically-seeded RNGs and perform the same
    operations in the same order:

    * Path A: ``open_length_bucketed_batch`` samples (advancing the RNG)
      then decodes (advancing it further).
    * Path B: we replay the SAME sample call on an identically-seeded
      RNG (recovering the same pointers AND the same RNG state), open
      sessions by hand, then call ``decode_pointer_batch`` with that RNG.

    Because both RNGs undergo identical operations in identical order,
    every output array must be byte-identical.
    """
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    sampler = _open_sampler(memmap_dir)
    factory = _make_session_factory(memmap_dir)

    batch_size = 4
    num_variants_per_section = 2
    context_len = 24
    band = (10, 40)

    # ---- Path A: the full helper. ----
    rng_a = np.random.default_rng(2718)
    result_a = open_length_bucketed_batch(
        factory, sampler,
        target_length=0, batch_size=batch_size,
        context_len=context_len,
        num_variants_per_section=num_variants_per_section,
        max_depth=2, rng=rng_a, band=band,
    )

    # ---- Path B: sample explicitly, hand-open sessions, decode core. ----
    rng_b = np.random.default_rng(2718)
    pointers = sampler.sample_section_pointers(
        target_length=0, count=batch_size, rng=rng_b, band=band,
    )
    assert pointers, "fixture must produce a non-empty band pool"
    sampled_binaries = {p.binary_name for p in pointers}
    with ExitStack() as stack:
        sessions: Dict[str, BinarySession] = {
            name: stack.enter_context(factory(name))
            for name in sampler.binary_names
            if name in sampled_binaries
        }
        result_b = decode_pointer_batch(
            sessions, pointers,
            context_len=context_len,
            num_variants_per_section=num_variants_per_section,
            max_depth=2, rng=rng_b,
        )

    # ---- Byte-equality on every output array. ----
    inner_a, inner_b = result_a.inner, result_b.inner
    np.testing.assert_array_equal(inner_a.tokens, inner_b.tokens)
    np.testing.assert_array_equal(inner_a.identities, inner_b.identities)
    np.testing.assert_array_equal(
        inner_a.identity_row_offsets, inner_b.identity_row_offsets,
    )
    np.testing.assert_array_equal(
        inner_a.numbers_significant, inner_b.numbers_significant,
    )
    np.testing.assert_array_equal(
        inner_a.numbers_sign_exponent, inner_b.numbers_sign_exponent,
    )
    np.testing.assert_array_equal(
        inner_a.number_row_offsets, inner_b.number_row_offsets,
    )
    np.testing.assert_array_equal(
        inner_a.batch_idx_to_section_variant,
        inner_b.batch_idx_to_section_variant,
    )
    np.testing.assert_array_equal(
        result_a.binary_id_per_row, result_b.binary_id_per_row,
    )
    assert result_a.binary_names == result_b.binary_names
