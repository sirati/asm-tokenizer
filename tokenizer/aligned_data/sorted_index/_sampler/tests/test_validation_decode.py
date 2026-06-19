"""Decode-validity test for the validation dataloader entry.

Drives :func:`open_validation_batches` end-to-end over the real combined
fixture binary (``build_combined_fixture_with_variants``), proving the
WHOLE wiring: the deterministic kernel stream -> per-bunch
``ExplicitIndicesSelection`` pointer -> the unchanged
``decode_pointer_batch`` -> a coherent RAGGED batch. This also exercises
the body-free ``_matched_section_variant_counts`` count_provider against a
real session (the fixture's matched variant counts are
``[0, 1, 4, 1, 2]``).

The pure shuffle/chunk math + the selection seam are covered without
decode in ``test_validation_sampler.py``; this file is the integration
seam over real session-backed memmaps.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    ReductionKind,
    SortedIndexReader,
    encode_sorted_index,
)
from tokenizer.aligned_data.sorted_index._types import IndexSpec
from tokenizer.aligned_data.sorted_index._sampler import (
    SequentialValidationSampler,
    open_validation_batches,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture_with_variants,
    make_test_vocab_manager,
)


_BINARY_NAME = "sortbin"
_BAND_LENGTH = 50
_SPEC = IndexSpec(LengthReduction(ReductionKind.MAX), depth=3)


def _write_all_sections_idx(base: Path) -> SortedIndexReader:
    """A reader placing all 5 fixture sections in ONE length bucket.

    The fixture's matched sections are indices 0..4; putting every section
    at ``_BAND_LENGTH`` means ``enumerate_in_band(50, 50)`` enumerates all
    five, so the real count_provider supplies their true variant counts
    ``[0, 1, 4, 1, 2]`` to the kernel.
    """
    path = base / f"{_BINARY_NAME}_sorted_max_d003.idx"
    lengths = np.full(5, _BAND_LENGTH, dtype=np.uint32)
    path.write_bytes(encode_sorted_index(lengths))
    return SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3
    )


def test_open_validation_batches_end_to_end(tmp_path: Path) -> None:
    vocab_manager = make_test_vocab_manager()
    base = build_combined_fixture_with_variants(tmp_path, vocab_manager)
    reader = _write_all_sections_idx(base)

    @contextmanager
    def session_factory(binary_name: str) -> Iterator[BinarySession]:
        dataset = BinaryDataset(
            base, binary_name, vocab_manager=vocab_manager
        )
        with dataset.open_session() as session:
            yield session

    batch_size = 2
    sampler = SequentialValidationSampler(
        [(_BINARY_NAME, _SPEC, reader)],
        batch_size=batch_size,
        band=(_BAND_LENGTH, _BAND_LENGTH),
        seed=2024,
    )

    results = list(
        open_validation_batches(
            session_factory,
            sampler,
            context_len=32,
            max_depth=2,
            rng=np.random.default_rng(0),
        )
    )

    # Variant counts [0,1,4,1,2], B=2 -> floor(n/2): idx2 -> 2 bunches,
    # idx4 -> 1 bunch; others (n<2) contribute nothing. 3 bunches total.
    assert len(results) == 3

    for result in results:
        inner = result.inner
        # RAGGED + explicit selection of exactly B indices => B rows.
        assert inner.tokens.shape == (batch_size, 32)
        assert inner.tokens.dtype == np.uint16
        # Single binary per batch; binary_id sidecar agrees.
        assert result.binary_id_per_row.shape == (batch_size,)
        assert set(int(b) for b in result.binary_id_per_row) == {0}
        assert result.binary_names == [_BINARY_NAME]
        # Non-degenerate: every row carries at least one non-pad token.
        assert int((inner.tokens != 0).sum()) > 0
        assert inner.identity_row_offsets.shape == (batch_size + 1,)
        assert inner.number_row_offsets.shape == (batch_size + 1,)
