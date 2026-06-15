"""Oracle equivalence for the realized-geometry sidecars (both arms).

For every fixture corpus, each (section, variant) sidecar TRIPLE must
equal the SCALAR source-of-truth geometry for that record -- independent
of the generator's bulk dedup path:

* ``body_len`` == ``expand_tokens(...).predicted_full_length - 1`` (the
  post-promotion post-strip body length, self/identity token excluded),
  AND == the legacy ``_lengths.bin`` body the realized-length sidecar
  stores for the same binary.
* ``(id_count, value_count)`` == ``count_surviving`` over the record's
  post-promotion body stream (``expanded_token_ids[1:]``, i.e. the self
  token dropped, exactly as the geometry engine excludes it).

Both arms are covered via the production :class:`BinarySession`
load_matched / load_unmatched walks, in the same section-major catalog
order the sidecar body is stored in.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    expand_tokens,
)
from tokenizer.aligned_data.loader.batch_decode._surviving_counts import (
    count_surviving,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    build_inline_decode_state,
)
from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    UnmatchedFunctionSpec,
    build_corpus,
    make_simple_variant,
)
from tokenizer.aligned_data.realized_lengths import (
    GEOMETRY_MATCHED_ARM,
    GEOMETRY_UNMATCHED_ARM,
    MATCHED_ARM,
    UNMATCHED_ARM,
    RealizedGeometryReader,
    RealizedLengths,
    generate_realized_geometry,
    generate_realized_lengths,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
    build_missing_variant_index_fixture,
)
from tokenizer.tokens import Category


def _scalar_triple(raw: np.ndarray) -> Tuple[int, int, int]:
    """Contributing ``(body_len, id_count, value_count)`` via the scalar truth."""
    state = build_inline_decode_state(
        np.asarray(raw, dtype=np.uint16), format_version=1
    )
    stub = SimpleNamespace(state=state, encounter_category=Category.LOCAL_FUNC)
    expanded = expand_tokens(stub)
    body_len = expanded.predicted_full_length - 1
    # Drop the prepended self-token so the band counts exclude it exactly
    # as the geometry engine does; count over the whole body.
    body_stream = expanded.expanded_token_ids[1:]
    counts = count_surviving(body_stream, body_stream.shape[0])
    return (
        body_len,
        counts.surviving_identity_count,
        counts.surviving_number_chunk_count,
    )


def _binary_name(base: Path) -> str:
    (idx_file,) = (
        p
        for p in base.glob("*_index.bin")
        if not p.name.endswith("_unmatched_index.bin")
    )
    return idx_file.name[: -len("_index.bin")]


def _matched_oracle(base: Path, name: str) -> List[Tuple[int, int, int]]:
    dataset = BinaryDataset(base, name, vocab_manager=None)
    out: List[Tuple[int, int, int]] = []
    with dataset.open_session() as session:
        idx = 0
        while True:
            try:
                matched = session.load_matched(idx)
            except IndexError:
                break
            for fd in matched.variants:
                out.append(_scalar_triple(fd.tokens))
            idx += 1
    return out


def _unmatched_oracle(base: Path, name: str) -> List[Tuple[int, int, int]]:
    dataset = BinaryDataset(base, name, vocab_manager=None)
    out: List[Tuple[int, int, int]] = []
    with dataset.open_session() as session:
        idx = 0
        while True:
            try:
                fd = session.load_unmatched(idx)
            except IndexError:
                break
            out.append(_scalar_triple(fd.tokens))
            idx += 1
    return out


def _assert_geometry_equivalent(base, name, geom_arm, len_arm, oracle) -> None:
    expected = np.asarray(oracle, dtype=np.int64).reshape(-1, 3)
    reader = RealizedGeometryReader.open(base, name, geom_arm)
    try:
        np.testing.assert_array_equal(
            np.asarray(reader.body_lengths, dtype=np.int64), expected[:, 0]
        )
        np.testing.assert_array_equal(
            np.asarray(reader.id_counts, dtype=np.int64), expected[:, 1]
        )
        np.testing.assert_array_equal(
            np.asarray(reader.value_counts, dtype=np.int64), expected[:, 2]
        )
    finally:
        reader.close()
    # body_len column must equal the legacy _lengths.bin body byte-for-byte.
    length_reader = RealizedLengths.open(base, name, len_arm)
    try:
        np.testing.assert_array_equal(
            np.asarray(length_reader.lengths, dtype=np.int64),
            np.asarray(reader.body_lengths, dtype=np.int64),
        )
    finally:
        length_reader.close()


@pytest.mark.parametrize(
    "builder",
    [
        build_combined_fixture,
        build_many_variant_section_fixture,
        build_missing_variant_index_fixture,
    ],
)
def test_matched_geometry_matches_scalar_oracle(builder, tmp_path: Path) -> None:
    base = builder(tmp_path)
    name = _binary_name(base)
    generate_realized_lengths(base, name)
    generate_realized_geometry(base, name)
    _assert_geometry_equivalent(
        base, name, GEOMETRY_MATCHED_ARM, MATCHED_ARM, _matched_oracle(base, name)
    )


def _corpus_with_both_arms(tmp_path: Path) -> Path:
    matched = (
        MatchedFunctionSpec(
            func_name="m_multi",
            variants=tuple(
                make_simple_variant(("m_multi", i), token_seed=i + 1, n_tokens=6 + i)
                for i in range(3)
            ),
            called=(),
        ),
        MatchedFunctionSpec(
            func_name="m_solo",
            variants=(make_simple_variant(("m_solo", 0), token_seed=20, n_tokens=9),),
            called=(),
        ),
    )
    unmatched = (
        UnmatchedFunctionSpec(
            func_name="u_two",
            versions=(
                make_simple_variant(("u_two", 0), token_seed=30, n_tokens=5),
                make_simple_variant(("u_two", 1), token_seed=31, n_tokens=11),
            ),
            called=(),
        ),
        UnmatchedFunctionSpec(
            func_name="u_one",
            versions=(make_simple_variant(("u_one", 0), token_seed=40, n_tokens=7),),
            called=(),
        ),
    )
    build_corpus(tmp_path, "bothbin", matched=matched, unmatched=unmatched)
    return tmp_path


def test_both_arms_geometry_match_scalar_oracle(tmp_path: Path) -> None:
    base = _corpus_with_both_arms(tmp_path)
    generate_realized_lengths(base, "bothbin")
    generate_realized_geometry(base, "bothbin")
    _assert_geometry_equivalent(
        base, "bothbin", GEOMETRY_MATCHED_ARM, MATCHED_ARM,
        _matched_oracle(base, "bothbin"),
    )
    _assert_geometry_equivalent(
        base, "bothbin", GEOMETRY_UNMATCHED_ARM, UNMATCHED_ARM,
        _unmatched_oracle(base, "bothbin"),
    )
