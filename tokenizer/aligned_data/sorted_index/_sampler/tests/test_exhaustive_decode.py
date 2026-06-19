"""Decode-validity test for the exhaustive dataloader entry.

Drives :func:`open_exhaustive_batches` end-to-end over the real combined
fixture binary, proving the WHOLE wiring: the rng-free whole-corpus
enumeration -> per-section all-variants ``ExplicitIndicesSelection``
pointer -> the unchanged ``decode_pointer_batch`` (RAGGED) -> a coherent
batch whose ``batch_idx_to_section_variant`` maps each row back to its
correct ``(section, variant)``.

The fixture's matched variant counts are ``[0, 1, 4, 1, 2]`` (section 0
has zero variants -> an empty selection -> zero rows), so the exhaustive
pass over all five sections yields ``1 + 4 + 1 + 2 = 8`` decoded rows.

The pure enumeration / completeness / determinism / order math is covered
without decode in ``test_exhaustive_sampler.py``; this file is the
integration seam over real session-backed memmaps.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Tuple

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
    ExhaustiveSectionSampler,
    open_exhaustive_batches,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture_with_variants,
    make_test_vocab_manager,
)


_BINARY_NAME = "sortbin"
_BAND_LENGTH = 50
_SPEC = IndexSpec(LengthReduction(ReductionKind.MAX), depth=3)
# The fixture's matched arm has 5 sections; section 0 carries 0 variants.
# The PRODUCTION builder stamps every 0-variant section with the wire
# EXCLUDED_LENGTH marker (see ``_length_compute``: ``emitted = n_variants >
# 0 & gate``), so a real non-excluded band NEVER enumerates a 0-variant
# section. We mirror that here: section 0 is stamped EXCLUDED, sections
# 1..4 (variant counts 1, 4, 1, 2) sit in the non-excluded band.
_FIXTURE_VARIANT_COUNTS = {1: 1, 2: 4, 3: 1, 4: 2}


def _write_all_sections_idx(base: Path) -> SortedIndexReader:
    """A reader placing the variant-bearing fixture sections in one bucket.

    Mirrors the production builder's exclusion stamp: section 0 (0
    variants) is stamped at the wire ``EXCLUDED_LENGTH`` marker so the FULL
    non-excluded band enumerates exactly the variant-bearing sections 1..4
    (true counts 1, 4, 1, 2). Section indices in the index file are the
    matched-arm idx, so the per-length lengths array is indexed by section.
    """
    from tokenizer.aligned_data.sorted_index._wire import EXCLUDED_LENGTH

    path = base / f"{_BINARY_NAME}_sorted_max_d003.idx"
    lengths = np.full(5, _BAND_LENGTH, dtype=np.uint32)
    lengths[0] = EXCLUDED_LENGTH  # 0-variant section -> excluded, as built.
    path.write_bytes(encode_sorted_index(lengths))
    return SortedIndexReader(
        path, reduction=LengthReduction(ReductionKind.MAX), depth=3
    )


def _expected_matched_variant_rows() -> List[Tuple[int, int]]:
    """Every ``(matched_idx, variant_idx)`` the exhaustive pass should decode.

    Section-major over the non-excluded sections (matched idx 1..4 in
    ascending enumeration order), each emitting ``range(n)`` variants in
    ascending order. The excluded 0-variant section 0 is never enumerated.
    """
    rows: List[Tuple[int, int]] = []
    for matched_idx in sorted(_FIXTURE_VARIANT_COUNTS):
        for var in range(_FIXTURE_VARIANT_COUNTS[matched_idx]):
            rows.append((matched_idx, var))
    return rows


def _decode_and_reconstruct(
    session_factory,
    sampler: ExhaustiveSectionSampler,
    reader: SortedIndexReader,
    *,
    group_size: int,
) -> List[Tuple[int, int]]:
    """Decode via ``open_exhaustive_batches`` + reconstruct real AsmRowIds.

    The decode's ``batch_idx_to_section_variant`` section column is the
    GROUP-LOCAL pointer position (0-based enumerate order within the group
    fed to ``decode_pointer_batch``), NOT the matched-arm idx; the variant
    column is the variant SLOT, which under ``ExplicitIndicesSelection`` of
    ``range(n)`` equals the raw variant index (identity). So the eval-side
    AsmRowId reconstruction maps ``group_local_pos -> the pointer at that
    position`` to recover ``(matched_idx, variant)``.

    This helper mirrors that reconstruction: it rebuilds the SAME canonical
    pointer sequence the entry decodes (via the sampler + a session-backed
    count provider), regroups it by ``group_size``, decodes through the
    entry, and for each decoded row maps back through the owning group's
    pointer list -- returning the recovered ``(matched_idx, variant)`` rows
    in decode-emission order.
    """
    sentinel = np.iinfo(np.uint32).max
    decoded: List[Tuple[int, int]] = []
    with session_factory(_BINARY_NAME) as count_session:

        def count_provider(binary_name, section_indices):
            return count_session._matched_section_variant_counts(
                section_indices
            )

        pointers = sampler.all_pointers(count_provider)

    groups = [
        pointers[s : s + group_size]
        for s in range(0, len(pointers), group_size)
    ]
    results = list(
        open_exhaustive_batches(
            session_factory,
            sampler,
            group_size=group_size,
            context_len=32,
            max_depth=2,
            rng=np.random.default_rng(0),
        )
    )
    assert len(results) == len(groups)
    for group, result in zip(groups, results):
        inner = result.inner
        mapping = inner.batch_idx_to_section_variant
        n_rows = mapping.shape[0]
        assert mapping.shape[1] == 2
        # RAGGED: no padding sentinel rows.
        assert int((mapping == sentinel).sum()) == 0
        assert inner.tokens.shape == (n_rows, 32)
        assert inner.tokens.dtype == np.uint16
        assert int((inner.tokens != 0).sum()) > 0  # non-degenerate content
        assert result.binary_names == [_BINARY_NAME]
        assert result.binary_id_per_row.shape == (n_rows,)
        assert set(int(b) for b in result.binary_id_per_row) == {0}
        assert result.depth_per_row.shape == (n_rows,)
        for r in range(n_rows):
            group_pos = int(mapping[r, 0])
            variant = int(mapping[r, 1])
            matched_idx = group[group_pos].section_pointer.idx
            decoded.append((matched_idx, variant))
    return decoded


def test_open_exhaustive_batches_end_to_end(tmp_path: Path) -> None:
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

    sampler = ExhaustiveSectionSampler([(_BINARY_NAME, _SPEC, reader)])
    group_size = 8  # > total pointers -> one RAGGED batch over everything.

    decoded = _decode_and_reconstruct(
        session_factory, sampler, reader, group_size=group_size
    )

    expected_rows = _expected_matched_variant_rows()
    total_rows = len(expected_rows)  # 1 + 4 + 1 + 2 == 8

    # COMPLETENESS at the decode seam: every (matched_idx, variant) once.
    assert len(decoded) == total_rows
    assert sorted(decoded) == sorted(expected_rows)
    assert len(set(decoded)) == total_rows  # no dup row

    # ROW RECONSTRUCTION + ORDER: a single RAGGED group preserves the
    # section-major / variant-ascending order the enumeration emits.
    assert decoded == expected_rows


def test_exhaustive_batches_fixed_grouping_no_drop(tmp_path: Path) -> None:
    # group_size that does NOT divide the total row count -> the last group
    # is smaller, and NO (section, variant) is dropped.
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

    sampler = ExhaustiveSectionSampler([(_BINARY_NAME, _SPEC, reader)])

    # 4 non-excluded sections (matched idx 1..4), group_size 3 -> pointer
    # groups [1,2,3], [4]: the last group is SMALLER (no drop). Sections 2
    # and 4 carry multiple variants, so a group spans several RAGGED rows;
    # the group-local section position resets to 0 per group, so the
    # AsmRowId reconstruction (group_pos -> pointer) is what recovers the
    # real matched idx.
    decoded = _decode_and_reconstruct(
        session_factory, sampler, reader, group_size=3
    )

    expected = _expected_matched_variant_rows()
    assert sorted(decoded) == sorted(expected)  # no (matched, variant) lost
    assert len(set(decoded)) == len(expected)  # no dup row
    assert decoded == expected  # section-major / variant-ascending order
