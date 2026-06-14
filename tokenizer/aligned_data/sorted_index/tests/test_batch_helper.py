"""End-to-end tests for :func:`open_length_bucketed_batch` (plan D7).

Builds two synthetic per-binary memmap directories (using the existing
``build_combined_fixture`` against two distinct base dirs + binary
names), hand-writes a sorted-index file for each binary via the
encoder (so the test does NOT depend on the not-yet-shipped builder),
opens a per-binary :class:`SortedIndexReader`, wraps them in a
:class:`MultiBinarySortedIndexSampler`, and runs the full batch
helper pipeline.

The synthetic combined fixture lays out matched sections at indices:

* 0 -- ``func_zero`` (0 variants; do NOT sample this)
* 1 -- ``solo_a`` (1 variant)
* 2 -- ``multi_fn`` (4 variants)
* 3 -- ``caller_fn`` (1 variant)
* 4 -- ``callee_fn`` (2 variants)

To avoid the 0-variant trap we only place sections 1..4 into the
hand-built sorted index.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pytest

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    MultiBinarySortedIndexSampler,
    ReductionKind,
    SortedIndexReader,
    encode_sorted_index,
    open_length_bucketed_batch,
)

from .fixtures import build_combined_fixture, make_test_vocab_manager


_BINARY_NAME_A = "binA"
_BINARY_NAME_B = "binB"


# ---------------------------------------------------------------------------
# Multi-binary fixture builder
# ---------------------------------------------------------------------------


def _build_multi_binary_fixture(tmp_path: Path) -> Path:
    """Lay down two combined-corpus binaries under ``tmp_path``.

    The existing ``build_combined_fixture`` hardcodes binary_name =
    ``sortbin`` and writes to ``tmp_path/combined``. We invoke it
    against two child dirs and then rename the on-disk artefacts so
    each child directory carries a uniquely-named binary catalog.
    """
    memmap_dir = tmp_path / "memmap"
    memmap_dir.mkdir()
    for binary_name in (_BINARY_NAME_A, _BINARY_NAME_B):
        scratch = tmp_path / f"scratch_{binary_name}"
        scratch.mkdir()
        # build_combined_fixture writes to ``scratch / 'combined'`` with
        # binary_name 'sortbin'; rename every artefact's prefix into
        # ``memmap_dir`` so the two binaries co-exist in one directory.
        combined_base = build_combined_fixture(scratch)
        for entry in combined_base.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.startswith("sortbin"):
                continue
            new_name = binary_name + entry.name[len("sortbin"):]
            (memmap_dir / new_name).write_bytes(entry.read_bytes())
    return memmap_dir


def _write_synthetic_sorted_index(
    memmap_dir: Path,
    binary_name: str,
    section_indices: List[int],
    length: int,
) -> Path:
    """Lay down a sorted-index file where every entry sits at one length.

    ``section_indices`` enumerates the original-order section indices
    that should land in the single length bucket; the encoder produces a
    blob with one populated bucket at ``length`` and the body in
    stable-sort order (which, since all keys are equal, preserves input
    order). We deliberately stamp every section the test wants to be
    sampleable into that single bucket so :func:`SortedIndexReader.
    sample_section_indices` returns exactly those indices at
    ``target_length=length``.
    """
    # Lengths array is sized to the largest section_index + 1; entries
    # not in ``section_indices`` get a different length so they end up
    # in a different bucket and are NOT sampled at ``target_length``.
    num_sections = max(section_indices) + 1
    lengths = np.full(num_sections, length + 100, dtype=np.uint32)
    for idx in section_indices:
        lengths[idx] = length
    path = memmap_dir / f"{binary_name}_sorted_max_d003.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_factory(memmap_dir: Path):
    """Build a session_factory closing over the multi-binary memmap dir."""
    vocab_manager = make_test_vocab_manager()
    @contextmanager
    def session_factory(binary_name: str) -> Iterator[BinarySession]:
        dataset = BinaryDataset(memmap_dir, binary_name, vocab_manager=vocab_manager)
        with dataset.open_session() as session:
            yield session
    return session_factory


def _open_sampler(memmap_dir: Path, length: int) -> MultiBinarySortedIndexSampler:
    """Build the per-binary readers + sampler over the multi-binary fixture."""
    sampleable_indices = [1, 2, 3, 4]   # skip the 0-variant section_idx=0
    readers = {}
    for name in (_BINARY_NAME_A, _BINARY_NAME_B):
        path = _write_synthetic_sorted_index(
            memmap_dir, name, sampleable_indices, length,
        )
        readers[name] = SortedIndexReader(
            path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
        )
    return MultiBinarySortedIndexSampler(readers)


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_open_length_bucketed_batch_end_to_end(tmp_path: Path) -> None:
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    target_length = 50
    sampler = _open_sampler(memmap_dir, target_length)
    factory = _make_session_factory(memmap_dir)
    rng = np.random.default_rng(42)
    batch_size = 4
    num_variants_per_section = 2
    context_len = 32

    result = open_length_bucketed_batch(
        factory,
        sampler,
        target_length=target_length,
        batch_size=batch_size,
        context_len=context_len,
        num_variants_per_section=num_variants_per_section,
        max_depth=2,
        rng=rng,
    )

    inner = result.inner
    expected_rows = batch_size * num_variants_per_section
    assert inner.tokens.shape == (expected_rows, context_len)
    assert inner.tokens.dtype == np.uint16

    # row offsets are batch_size + 1.
    assert inner.identity_row_offsets.shape == (expected_rows + 1,)
    assert inner.number_row_offsets.shape == (expected_rows + 1,)

    # binary_id_per_row carries the right cardinality.
    assert result.binary_id_per_row.shape == (expected_rows,)
    # Every binary_id is a valid index into binary_names.
    assert set(int(b) for b in result.binary_id_per_row) <= {0, 1}
    assert result.binary_names == sorted([_BINARY_NAME_A, _BINARY_NAME_B])


def test_open_length_bucketed_batch_binary_id_matches_sampled_order(
    tmp_path: Path,
) -> None:
    """``binary_id_per_row`` must enumerate binaries in alphabetical
    order matching :attr:`MultiBinarySortedIndexSampler.binary_names`."""
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    target_length = 50
    sampler = _open_sampler(memmap_dir, target_length)
    factory = _make_session_factory(memmap_dir)
    # Seed deterministically and check the row segments line up with
    # alphabetical binary order:
    rng = np.random.default_rng(7)
    num_variants_per_section = 2
    result = open_length_bucketed_batch(
        factory,
        sampler,
        target_length=target_length,
        batch_size=4,
        context_len=16,
        num_variants_per_section=num_variants_per_section,
        max_depth=2,
        rng=rng,
    )
    # The output's binary_id_per_row is the run-length sequence
    # (0...0, 1...1) in alphabetical order; check monotone-non-
    # decreasing.
    bin_ids = result.binary_id_per_row.tolist()
    assert bin_ids == sorted(bin_ids), (
        f"binary_id_per_row must be sorted by alphabetical binary_name "
        f"order; got {bin_ids}"
    )


# ---------------------------------------------------------------------------
# Rejections (D7 contract)
# ---------------------------------------------------------------------------


def test_open_length_bucketed_batch_rejects_keep_intermediate(
    tmp_path: Path,
) -> None:
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    target_length = 50
    sampler = _open_sampler(memmap_dir, target_length)
    factory = _make_session_factory(memmap_dir)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="keep_intermediate"):
        open_length_bucketed_batch(
            factory,
            sampler,
            target_length=target_length,
            batch_size=2,
            context_len=16,
            num_variants_per_section=2,
            max_depth=2,
            rng=rng,
            keep_intermediate=True,
        )


def test_open_length_bucketed_batch_rejects_empty_pool(tmp_path: Path) -> None:
    memmap_dir = _build_multi_binary_fixture(tmp_path)
    target_length = 50
    sampler = _open_sampler(memmap_dir, target_length)
    factory = _make_session_factory(memmap_dir)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="empty sampler pool"):
        open_length_bucketed_batch(
            factory,
            sampler,
            # ask for a length that is not in any binary's index:
            target_length=999_999,
            batch_size=2,
            context_len=16,
            num_variants_per_section=2,
            max_depth=2,
            rng=rng,
        )
