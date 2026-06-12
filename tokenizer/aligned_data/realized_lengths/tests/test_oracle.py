"""Oracle equivalence for the realized-length sidecars (both arms).

For every fixture corpus, each (section, variant) sidecar length must
equal the legacy SCALAR contributing-length path for that record -- the
same oracle style ``test_bulk_expand_lengths`` uses:
``expand_tokens(...).predicted_full_length - 1`` (the post-promotion
post-strip body length, self/identity token excluded) over the
variant's raw token stream loaded through the production
:class:`BinarySession` (independent of the generator's bulk compute).

Both arms are covered: matched variants via
:meth:`BinarySession.load_matched`; unmatched records (one per variant,
section-major) via :meth:`BinarySession.load_unmatched`, walked in the
same section-major catalog order the sidecar body is stored in.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    expand_tokens,
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
    MATCHED_ARM,
    UNMATCHED_ARM,
    RealizedLengths,
    generate_realized_lengths,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
    build_missing_variant_index_fixture,
)
from tokenizer.tokens import Category


def _scalar_body_length(raw: np.ndarray) -> int:
    """Contributing body length via the scalar source of truth (self excluded)."""
    state = build_inline_decode_state(
        np.asarray(raw, dtype=np.uint16), format_version=1
    )
    stub = SimpleNamespace(state=state, encounter_category=Category.LOCAL_FUNC)
    return expand_tokens(stub).predicted_full_length - 1


def _binary_name(base: Path) -> str:
    (idx_file,) = (
        p
        for p in base.glob("*_index.bin")
        if not p.name.endswith("_unmatched_index.bin")
    )
    return idx_file.name[: -len("_index.bin")]


def _matched_oracle_body(base: Path, name: str) -> List[int]:
    """Section-major scalar body length per matched (section, variant)."""
    dataset = BinaryDataset(base, name, vocab_manager=None)
    out: List[int] = []
    with dataset.open_session() as session:
        idx = 0
        while True:
            try:
                matched = session.load_matched(idx)
            except IndexError:
                break
            for fd in matched.variants:
                out.append(_scalar_body_length(fd.tokens))
            idx += 1
    return out


def _unmatched_oracle_body(base: Path, name: str) -> List[int]:
    """Section-major scalar body length per unmatched record (one per variant)."""
    dataset = BinaryDataset(base, name, vocab_manager=None)
    out: List[int] = []
    with dataset.open_session() as session:
        idx = 0
        while True:
            try:
                fd = session.load_unmatched(idx)
            except IndexError:
                break
            out.append(_scalar_body_length(fd.tokens))
            idx += 1
    return out


def _assert_arm_equivalent(base: Path, name: str, arm, oracle_body) -> None:
    reader = RealizedLengths.open(base, name, arm)
    try:
        np.testing.assert_array_equal(
            np.asarray(reader.lengths, dtype=np.int64),
            np.asarray(oracle_body, dtype=np.int64),
        )
    finally:
        reader.close()


@pytest.mark.parametrize(
    "builder",
    [
        build_combined_fixture,
        build_many_variant_section_fixture,
        build_missing_variant_index_fixture,
    ],
)
def test_matched_arm_matches_scalar_oracle(builder, tmp_path: Path) -> None:
    base = builder(tmp_path)
    name = _binary_name(base)
    generate_realized_lengths(base, name)
    _assert_arm_equivalent(
        base, name, MATCHED_ARM, _matched_oracle_body(base, name)
    )


def _corpus_with_both_arms(tmp_path: Path) -> Path:
    """A corpus carrying BOTH a non-trivial matched and unmatched arm."""
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


def test_both_arms_match_scalar_oracle(tmp_path: Path) -> None:
    base = _corpus_with_both_arms(tmp_path)
    generate_realized_lengths(base, "bothbin")
    _assert_arm_equivalent(
        base, "bothbin", MATCHED_ARM, _matched_oracle_body(base, "bothbin")
    )
    _assert_arm_equivalent(
        base, "bothbin", UNMATCHED_ARM, _unmatched_oracle_body(base, "bothbin")
    )
