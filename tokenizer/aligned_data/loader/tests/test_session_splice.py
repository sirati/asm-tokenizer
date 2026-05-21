"""Tests for ``BinarySession`` decoded-splice wiring.

Three concerns covered:

1. Token-id caching in ``__enter__`` / ``__exit__`` (lazy resolution
   + per-session lifecycle).
2. ``_idx_for_section_offset`` round-trip + miss + bad-arm contract.
3. ``splice_with_callees`` end-to-end against a synthetic 2-function
   matched corpus with a real call_target edge.

The fixture builds on the existing ``_session_fixture`` corpus
builder, but swaps in a vocab carrying both the variant-decoder
strings AND the 15 v2 type-tokens at predictable ids so the
:mod:`decoded.category_tokens` resolvers can run. Token payloads in
the matched-arm functions are hand-crafted into valid v2 streams
(real-token id >= 256 followed by inline-digit bytes) so
``decode_raw_tokens`` exercises identity decoding + the multi-chunk
promotion path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.tokens import Category, TokenType

from ._corpus import (
    MatchedFunctionSpec,
    VariantSpec,
    build_corpus_with_registry,
)
from ._session_fixture import (
    _FakeArm,
    _FakeVocab,
    _VariantStubRegistry,
    _write_variants_bin,
)


# ---------------------------------------------------------------------------
# Vocab + corpus helpers
# ---------------------------------------------------------------------------


# Slots 256..259 are reserved for the variant-record strings
# (arch:x64, comp:gcc, cver:gcc:13.2.0, opt:O2 in order). Slots 260+
# carry the 15 v2 type-tokens in the order below.
_BASE_VOCAB_STRINGS = (
    "arch:x64",
    "comp:gcc",
    "cver:gcc:13.2.0",
    "opt:O2",
)

# (TokenType, in-vocab id) -- offsets from 260.
_V2_TYPE_TOKENS = (
    TokenType.BLOCK_V2,
    TokenType.LOCAL_FUNC,
    TokenType.PLT_FUNC,
    TokenType.EXT_FUNC,
    TokenType.RO_DATA_PTR,
    TokenType.RW_DATA_PTR,
    TokenType.STRING_PTR,
    TokenType.JUMP_TABLE,
    TokenType.VALUED_CONST_V2,
    TokenType.FLOAT16,
    TokenType.BFLOAT16,
    TokenType.FLOAT32,
    TokenType.FLOAT64,
    TokenType.FLOAT80,
    TokenType.FLOAT128,
)


def _v2_id(token_type: TokenType) -> int:
    """Vocab id of ``token_type`` in the splice-test layout."""
    return 260 + _V2_TYPE_TOKENS.index(token_type)


class _SpliceFakeVocab(_FakeVocab):
    """``_FakeVocab`` extended with an ``id_to_token_type`` ndarray.

    Variant-decoder strings stay at ids 256-259 (tagged UNRESOLVED so
    the decoder rounds-trips them as bare strings); v2 type-tokens
    occupy ids 260..274 with their matching TokenType. The array
    capacity is sized to ``max(id)+1`` so the resolver's ``np.where``
    finds each tag at exactly one id.
    """

    def __init__(self) -> None:
        super().__init__(list(_BASE_VOCAB_STRINGS))
        capacity = 260 + len(_V2_TYPE_TOKENS) + 8
        arr = np.full(capacity, TokenType.UNRESOLVED, dtype=np.int8)
        for offset, token_type in enumerate(_V2_TYPE_TOKENS):
            arr[260 + offset] = int(token_type)
        self.id_to_token_type = arr


def _craft_v2_tokens(
    *,
    block_count: int = 1,
    local_func_identities: tuple[int, ...] = (),
) -> np.ndarray:
    """Hand-craft a valid v2 wire-format uint16 stream.

    Layout: ``block_count`` BLOCK_V2 tokens, followed by one
    LOCAL_FUNC token per identity in ``local_func_identities`` each
    trailed by one inline-digit byte carrying the identity value
    (caller picks values in ``0..0xFE``). First position is always a
    real-token (BLOCK_V2 >= 256) so the v2 codec precondition holds.
    """
    out: List[int] = []
    block_id = _v2_id(TokenType.BLOCK_V2)
    local_func_id = _v2_id(TokenType.LOCAL_FUNC)
    for _ in range(block_count):
        out.append(block_id)
    for identity in local_func_identities:
        if not 0 <= identity <= 0xFE:
            raise ValueError(
                f"identity {identity} out of v2 inline-digit range [0, 0xFE]"
            )
        out.append(local_func_id)
        out.append(identity)
    return np.array(out, dtype=np.uint16)


def _variant_spec(
    vkey, *, block_count: int = 1, local_func_identities: tuple[int, ...] = ()
) -> VariantSpec:
    """A ``VariantSpec`` with hand-crafted v2 tokens.

    block_runlength / insn_runlength carry placeholder shapes -- the
    splice path only reads ``tokens``, but the writer requires the
    arrays to be non-empty + dtype-correct.
    """
    tokens = _craft_v2_tokens(
        block_count=block_count,
        local_func_identities=local_func_identities,
    )
    block_rl = np.array([len(tokens)], dtype=np.uint8)
    # Insn runlength: 2-element schedule to satisfy the writer.
    insn_rl = np.array(
        [2, max(1, len(tokens) - 2)], dtype=np.uint8
    )
    return VariantSpec(vkey=vkey, tokens=tokens, block_rl=block_rl, insn_rl=insn_rl)


def _build_splice_corpus(
    tmp_path: Path,
    matched_specs: tuple[MatchedFunctionSpec, ...],
) -> Dict[str, Any]:
    """Lay down a corpus + assemble the metadata bag the session reads.

    Mirrors :func:`build_synthetic_binary` but with the splice-aware
    vocab + caller-supplied matched specs. The unmatched arm is left
    empty unless a spec opts in -- splice tests focus on the matched
    path because matched arm pre-cached ``bin_starts`` give clean
    section-offset round-trips.
    """
    base = tmp_path
    binary_name = "splicebin"
    vocab = _SpliceFakeVocab()
    variant_offset = _write_variants_bin(base, binary_name, vocab)
    variant_ref_hex = f"{variant_offset:x}"

    # Map every variant key in every spec to the single hand-laid
    # variants.bin record so the resolver round-trip stays cheap.
    vkeys = []
    for spec in matched_specs:
        for variant in spec.variants:
            vkeys.append(variant.vkey)
    variants_registry = _VariantStubRegistry(
        {vkey: variant_ref_hex for vkey in vkeys}
    )

    corpus = build_corpus_with_registry(
        base,
        binary_name,
        matched=matched_specs,
        unmatched=(),
        variants=variants_registry,
    )

    pair = read_csv_section_index_arrays(corpus.matched_index_bin)
    assert pair is not None
    bin_starts, bin_lengths = pair
    func_names = [spec.func_name for spec in matched_specs]
    matched_arm = _FakeArm(
        starts=np.zeros(0, dtype=np.int64),
        func_names=func_names,
        bin_starts=bin_starts,
        bin_lengths=bin_lengths,
    )

    metadata = {
        "matched_arm": matched_arm,
        "unmatched_arm": None,
        "offset_to_filename": {
            variant_offset: f"{binary_name}-x64-gcc-13.2.0-O2"
        },
        "line_to_name": {},
    }
    return {
        "base_path": base,
        "binary_name": binary_name,
        "vocab": vocab,
        "metadata": metadata,
        "matched_arm": matched_arm,
        "bin_starts": bin_starts,
    }


@pytest.fixture
def splice_corpus_single(tmp_path: Path) -> Dict[str, Any]:
    """Single matched function, no callees -- depth=0 + cache tests.

    Body emits zero LOCAL_FUNC tokens because the function has no
    declared callees: every caller-local id WOULD resolve out-of-range
    in the FID-resolution decode (plan Decisions 22, 28). The cache
    tests only need a valid two-variant function with a real section,
    so dropping the orphan identity slots is the correct shape for the
    new design.
    """
    spec = MatchedFunctionSpec(
        func_name="only_fn",
        variants=(
            _variant_spec(("only", 0)),
            _variant_spec(("only", 1)),
        ),
        called=(),
    )
    return _build_splice_corpus(tmp_path, (spec,))


@pytest.fixture
def splice_corpus_caller_callee(tmp_path: Path) -> Dict[str, Any]:
    """Three matched functions: ``caller`` calls ``callee`` + ``other``,
    ``callee`` calls ``other`` (FID unification fixture).

    The caller body emits two LOCAL_FUNC tokens (caller-local ids 0, 1
    resolving to ``callee`` + ``other`` per the encounter-ordered
    call_targets table). ``callee``'s body emits one LOCAL_FUNC token
    referencing ``other`` again. After splice + compaction the same
    callee FID (``other``) appearing in both the caller's reference
    AND the spliced callee's reference shares ONE compacted id --
    that's the FID-unification invariant the new design buys.
    """
    other_spec = MatchedFunctionSpec(
        func_name="other",
        variants=(_variant_spec(("other", 0)),),
        called=(),
    )
    callee_spec = MatchedFunctionSpec(
        func_name="callee",
        variants=(_variant_spec(("callee", 0), local_func_identities=(0,)),),
        called=("other",),
    )
    caller_spec = MatchedFunctionSpec(
        func_name="caller",
        variants=(
            _variant_spec(
                ("caller", 0), local_func_identities=(0, 1)
            ),
        ),
        called=("callee", "other"),
    )
    return _build_splice_corpus(
        tmp_path, (other_spec, callee_spec, caller_spec)
    )


def _caller_idx(corpus: Dict[str, Any]) -> int:
    """Return the matched-arm idx for ``caller`` in the caller_callee corpus."""
    func_names: List[str] = corpus["matched_arm"].func_names
    return func_names.index("caller")


def _callee_idx(corpus: Dict[str, Any]) -> int:
    func_names: List[str] = corpus["matched_arm"].func_names
    return func_names.index("callee")


@pytest.fixture
def splice_corpus_multi_variant(tmp_path: Path) -> Dict[str, Any]:
    """Single matched function with five distinct-bodied variants.

    Each variant carries a distinct ``token_seed``-equivalent body
    (the BLOCK_V2-prefixed wire stream's length depends on
    ``local_func_identities``); the per-variant ``func_name`` stays
    ``"multi"`` so the matched arm groups them into one function with
    five variants. No callees -- this fixture exists to exercise the
    N-variant sampling rule (D3) without dragging callee resolution
    into the assertion surface.
    """
    spec = MatchedFunctionSpec(
        func_name="multi",
        variants=tuple(
            _variant_spec(("multi", i), block_count=i + 1)
            for i in range(5)
        ),
        called=(),
    )
    return _build_splice_corpus(tmp_path, (spec,))


# ---------------------------------------------------------------------------
# Token-id cache
# ---------------------------------------------------------------------------


def test_token_id_cache_populated_on_first_call_and_reused(
    splice_corpus_single, monkeypatch
):
    """First splice triggers resolution; second reuses the cache."""
    from tokenizer.aligned_data.loader.decoded import category_tokens

    call_count = {"category": 0, "number": 0}
    original_cat = category_tokens.resolve_category_token_ids
    original_num = category_tokens.resolve_number_token_ids

    def _spy_cat(vm):
        call_count["category"] += 1
        return original_cat(vm)

    def _spy_num(vm):
        call_count["number"] += 1
        return original_num(vm)

    monkeypatch.setattr(
        "tokenizer.aligned_data.loader._session_splice."
        "resolve_category_token_ids",
        _spy_cat,
        raising=False,
    )
    monkeypatch.setattr(
        "tokenizer.aligned_data.loader._session_splice."
        "resolve_number_token_ids",
        _spy_num,
        raising=False,
    )
    # The mixin imports the resolvers lazily inside _get_token_id_caches,
    # so patch the source module too -- safer than betting on import time.
    monkeypatch.setattr(category_tokens, "resolve_category_token_ids", _spy_cat)
    monkeypatch.setattr(category_tokens, "resolve_number_token_ids", _spy_num)

    fb = splice_corpus_single
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        # No splice yet -- caches untouched.
        assert sess._category_token_ids is None
        assert sess._number_token_ids is None

        sess.splice_with_callees(0, arm="matched", max_depth=0)[0]
        assert call_count == {"category": 1, "number": 1}
        assert sess._category_token_ids is not None
        assert sess._number_token_ids is not None

        sess.splice_with_callees(0, arm="matched", max_depth=0)[0]
        # Cache hit: no further resolver calls.
        assert call_count == {"category": 1, "number": 1}


def test_token_id_cache_cleared_on_exit(splice_corpus_single):
    """Per-session caches reset on ``__exit__``."""
    fb = splice_corpus_single
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with sess:
        sess.splice_with_callees(0, arm="matched", max_depth=0)[0]
        assert sess._category_token_ids is not None
    assert sess._category_token_ids is None
    assert sess._number_token_ids is None

    # Re-enter the same instance: caches re-populate on first splice.
    with sess:
        assert sess._category_token_ids is None
        sess.splice_with_callees(0, arm="matched", max_depth=0)[0]
        assert sess._category_token_ids is not None


# ---------------------------------------------------------------------------
# _idx_for_section_offset
# ---------------------------------------------------------------------------


def test_idx_for_section_offset_matched_roundtrip(splice_corpus_caller_callee):
    """Each known ``bin_starts`` offset round-trips to its function idx."""
    fb = splice_corpus_caller_callee
    bin_starts = fb["bin_starts"]
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        for expected_idx in range(len(bin_starts)):
            offset = int(bin_starts[expected_idx])
            assert sess._idx_for_section_offset(offset, "matched") == expected_idx


def test_idx_for_section_offset_returns_none_on_miss(splice_corpus_single):
    fb = splice_corpus_single
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        # Pick an offset we KNOW is not in bin_starts.
        bogus = int(fb["bin_starts"][0]) + 0xABCD
        assert sess._idx_for_section_offset(bogus, "matched") is None


def test_idx_for_section_offset_rejects_unknown_arm(splice_corpus_single):
    fb = splice_corpus_single
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(ValueError, match="unknown arm"):
            sess._idx_for_section_offset(0, "garbage")


def test_idx_for_section_offset_unmatched_arm_absent(splice_corpus_single):
    """The single-fn corpus has no unmatched arm; lookup returns ``None``."""
    fb = splice_corpus_single
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        assert sess._idx_for_section_offset(0, "unmatched") is None


# ---------------------------------------------------------------------------
# splice_with_callees
# ---------------------------------------------------------------------------


def test_splice_depth_zero_matched_equals_decode_only(splice_corpus_single):
    """``max_depth=0`` returns the decoded root unchanged.

    The fixture's only function has no LOCAL_FUNC tokens (no declared
    callees), so the post-compaction identity arrays are empty for
    every Category. The depth-0 splice and a direct FID-resolved
    decode of the same body match byte-for-byte.
    """
    from tokenizer.aligned_data.loader._session_splice import (
        _build_fids_per_category,
    )
    from tokenizer.aligned_data.loader.decoded.extract import _decode_to_staging
    from tokenizer.aligned_data.loader.decoded.splice import _compact_ids

    fb = splice_corpus_single
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        spliced = sess.splice_with_callees(0, arm="matched", max_depth=0)[0]
        cat_ids, num_ids = sess._get_token_id_caches()
        matched = sess.load_matched(0)
        section, _section_offset, _matched = (
            sess._load_matched_section_and_variants(0)
        )
        fids = _build_fids_per_category(section)
        staging = _decode_to_staging(
            matched.variants[0].full_token_stream(),
            id_token_ids=cat_ids,
            number_token_ids=num_ids,
            fids_per_category=fids,
            func_name=matched.func_name,
            metadata=matched.variants[0].metadata,
        )
        baseline_identities = {
            c: _compact_ids(staging.identities[c]) for c in Category
        }
    assert np.array_equal(spliced.real_tokens, staging.real_tokens)
    for category in Category:
        assert np.array_equal(
            spliced.identities[category],
            baseline_identities[category],
        )


def test_splice_depth_one_unifies_shared_fid(splice_corpus_caller_callee):
    """``max_depth=1`` splices the callee body in and the same callee FID
    referenced from both caller and callee compacts to ONE id.

    Setup (see ``splice_corpus_caller_callee``):
      * caller calls ``callee`` (local id 0) + ``other`` (local id 1)
      * callee calls ``other`` (local id 0 within its own scope)

    Caller's body emits LOCAL_FUNC tokens at caller-local ids ``(0, 1)``;
    these resolve to ``[FID(callee), FID(other)]``. Callee's body emits
    one LOCAL_FUNC token at its-own caller-local id ``0`` which resolves
    to ``FID(other)`` -- the SAME FID the caller already references.

    Depth=0 (caller only) sees two distinct FIDs -> compacted to
    ``[0, 1]``. Depth=1 (caller + callee body) concatenates a third
    occurrence of ``FID(other)``; compaction aliases that third slot to
    the same compacted id ``other`` already got from the caller's slot,
    giving ``[0, 1, 1]``. That alias is the FID-unification invariant
    -- one callee, one compacted id, regardless of where in the splice
    tree it's referenced.
    """
    fb = splice_corpus_caller_callee
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        caller_idx = _caller_idx(fb)
        depth0 = sess.splice_with_callees(caller_idx, arm="matched", max_depth=0)[0]
        depth1 = sess.splice_with_callees(caller_idx, arm="matched", max_depth=1)[0]

    assert depth1.real_tokens.shape[0] > depth0.real_tokens.shape[0], (
        "depth=1 must grow the stream by the callee body"
    )

    # Caller's two LOCAL_FUNC slots: callee + other -> two distinct
    # compacted ids in encoder-allocation order.
    assert depth0.identities[Category.LOCAL_FUNC].tolist() == [0, 1]
    # Depth=1 appends callee's one LOCAL_FUNC slot (referencing other);
    # that slot's FID matches the caller's second slot, so compaction
    # aliases to the same compacted id (== 1).
    assert depth1.identities[Category.LOCAL_FUNC].tolist() == [0, 1, 1]


def test_splice_unmatched_arm_supported(tmp_path):
    """``arm="unmatched"`` round-trips a single-record decoded view.

    The unmatched arm needs its own metadata bag (the splice corpus
    fixture omits it). Build a minimal one with one unmatched function
    + the splice vocab so ``max_depth=0`` returns the decoded body.
    """
    from tokenizer.aligned_data.loader._sections_bin_walk import (
        unmatched_region_start,
    )
    from tokenizer.aligned_data.index_format import read_index_arrays
    from tokenizer.aligned_data.loader.decoded.extract import decode_raw_tokens
    from tokenizer.aligned_data.loader.function_names_loader import (
        load_function_names,
    )

    from ._corpus import UnmatchedFunctionSpec

    base = tmp_path
    binary_name = "umbin"
    vocab = _SpliceFakeVocab()
    variant_offset = _write_variants_bin(base, binary_name, vocab)
    variant_ref_hex = f"{variant_offset:x}"

    unmatched_spec = UnmatchedFunctionSpec(
        func_name="lonely_um",
        versions=(_variant_spec(("um", 0), local_func_identities=(0,)),),
        called=(),
    )
    variants_registry = _VariantStubRegistry({("um", 0): variant_ref_hex})
    corpus = build_corpus_with_registry(
        base, binary_name, matched=(), unmatched=(unmatched_spec,),
        variants=variants_registry,
    )

    starts = read_index_arrays(corpus.unmatched_index_bin)
    section_offset = unmatched_region_start(corpus.matched_index_bin)
    record_to_section_idx = np.zeros(len(starts), dtype=np.uint32)
    unmatched_arm = _FakeArm(
        starts=starts,
        func_names=["lonely_um"],
        section_starts=np.array([section_offset], dtype=np.int64),
        record_to_section_idx=record_to_section_idx,
    )
    _, line_to_name = load_function_names(corpus.function_names_sidecar)
    metadata = {
        "matched_arm": None,
        "unmatched_arm": unmatched_arm,
        "offset_to_filename": {
            variant_offset: f"{binary_name}-x64-gcc-13.2.0-O2"
        },
        "line_to_name": line_to_name,
    }

    with BinarySession(base, binary_name, vocab, metadata) as sess:
        spliced = sess.splice_with_callees(0, arm="unmatched", max_depth=0)[0]
        # Splicer decodes the FULL wire stream; baseline must do the same.
        cat_ids, num_ids = sess._get_token_id_caches()
        fd = sess.load_unmatched(0)
        baseline = decode_raw_tokens(
            fd.full_token_stream(),
            id_token_ids=cat_ids,
            number_token_ids=num_ids,
            func_name=fd.func_name,
            metadata=fd.metadata,
        )
    assert np.array_equal(spliced.real_tokens, baseline.real_tokens)


def test_splice_rejects_unknown_arm(splice_corpus_single):
    fb = splice_corpus_single
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(ValueError, match="unknown arm"):
            sess.splice_with_callees(0, arm="bogus", max_depth=0)


# ---------------------------------------------------------------------------
# N-variant sampling (D2 / D3)
# ---------------------------------------------------------------------------


def test_splice_rejects_max_variants_zero(splice_corpus_multi_variant):
    """``max_variants < 1`` is programmer error, raises ``ValueError``."""
    fb = splice_corpus_multi_variant
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(ValueError, match="max_variants"):
            sess.splice_with_callees(0, arm="matched", max_depth=0, max_variants=0)


def test_splice_returns_list_for_single_variant(splice_corpus_multi_variant):
    """``max_variants=1`` (default) returns a single-element list."""
    fb = splice_corpus_multi_variant
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        result = sess.splice_with_callees(0, arm="matched", max_depth=0)
        assert isinstance(result, list)
        assert len(result) == 1


def test_splice_returns_n_streams_for_max_variants_n(splice_corpus_multi_variant):
    """``max_variants=N`` returns ``N`` streams with distinct primary bodies."""
    fb = splice_corpus_multi_variant
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        rng = np.random.default_rng(seed=12345)
        result = sess.splice_with_callees(
            0, arm="matched", max_depth=0, max_variants=3, rng=rng
        )
        assert len(result) == 3
        # Each variant has block_count=i+1, so its decoded real_tokens
        # length encodes the variant index (one real-token per BLOCK_V2).
        lengths = sorted(int(df.real_tokens.shape[0]) for df in result)
        # All distinct (each variant has a different block_count).
        assert len(set(lengths)) == 3


def test_splice_max_variants_clamps_when_exceeds_total(splice_corpus_multi_variant):
    """``max_variants > len(variants)`` clamps to ``len(variants)``.

    The fixture has 5 variants; asking for 99 yields 5 (all of them).
    No rng is needed because the all-variants branch skips sampling
    and returns the existing index order.
    """
    fb = splice_corpus_multi_variant
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        result = sess.splice_with_callees(
            0, arm="matched", max_depth=0, max_variants=99
        )
        assert len(result) == 5
        # Each variant has a distinct block_count, so each real_tokens
        # array has a distinct length spanning [1, 5].
        lengths = sorted(int(df.real_tokens.shape[0]) for df in result)
        assert lengths == [1, 2, 3, 4, 5]


def test_splice_seeded_rng_is_deterministic(splice_corpus_multi_variant):
    """Two calls with the same seed produce the same variant selection."""
    fb = splice_corpus_multi_variant
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        rng_a = np.random.default_rng(seed=20260521)
        rng_b = np.random.default_rng(seed=20260521)
        result_a = sess.splice_with_callees(
            0, arm="matched", max_depth=0, max_variants=2, rng=rng_a
        )
        result_b = sess.splice_with_callees(
            0, arm="matched", max_depth=0, max_variants=2, rng=rng_b
        )
    lengths_a = [int(df.real_tokens.shape[0]) for df in result_a]
    lengths_b = [int(df.real_tokens.shape[0]) for df in result_b]
    assert lengths_a == lengths_b


def test_splice_unseeded_rng_diverges_across_calls(splice_corpus_multi_variant):
    """``rng=None`` constructs a fresh non-deterministic generator per call.

    With 5 variants and ``max_variants=2`` there are ``C(5, 2) = 10``
    possible selections. Running 32 trials drives the probability that
    every pair matches the first below ``(1/10)^31 ~= 10^-31`` -- if the
    sampler were silently deterministic, every trial would share the
    first's signature. We accept the test passes once any divergence
    appears.
    """
    fb = splice_corpus_multi_variant
    signatures: List[tuple] = []
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        for _ in range(32):
            result = sess.splice_with_callees(
                0, arm="matched", max_depth=0, max_variants=2
            )
            sig = tuple(sorted(int(df.real_tokens.shape[0]) for df in result))
            signatures.append(sig)
    assert len(set(signatures)) > 1, (
        f"unseeded rng produced identical selection across 32 trials: "
        f"{signatures[0]!r}"
    )


# ---------------------------------------------------------------------------
# Inlining-equivalence flag (D5)
# ---------------------------------------------------------------------------
#
# The session-level integration tests below cover the flag's wiring
# through the walker callback (D6). Rich per-variant divergence cases
# (some variants call K, some don't) live in the walker's own unit
# tests under ``decoded/tests/`` -- the loader corpus builder shares
# one ``called`` set across every variant of a function, so the only
# session-observable inlining-flag state with this corpus is the
# "all variants call K" branch (and its negation).


def test_splice_inlining_flag_off_matches_legacy_depth_one(
    splice_corpus_caller_callee,
):
    """Flag OFF + depth=1 keeps splicing the callee body in (regression).

    Identical to ``test_splice_depth_one_unifies_shared_fid`` but
    spelled out as the flag-OFF regression cell. With the flag OFF
    the walker uses standard cycle + present checks only -- the
    inlining-equivalence narrowing never fires.
    """
    fb = splice_corpus_caller_callee
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        caller_idx = _caller_idx(fb)
        depth1 = sess.splice_with_callees(
            caller_idx,
            arm="matched",
            max_depth=1,
            inlined_equivalent_call_targets_only=False,
        )[0]
        depth0 = sess.splice_with_callees(
            caller_idx,
            arm="matched",
            max_depth=0,
            inlined_equivalent_call_targets_only=False,
        )[0]
    assert depth1.real_tokens.shape[0] > depth0.real_tokens.shape[0]
    assert depth1.identities[Category.LOCAL_FUNC].tolist() == [0, 1, 1]


def test_splice_inlining_flag_on_skips_universally_called_targets(
    splice_corpus_caller_callee,
):
    """Flag ON + all variants call K -> walker skips K.

    The caller fixture has one variant which trivially "all variants
    call" every call_target. Under the flag the walker must skip
    every K -- depth=1 collapses to the same shape as depth=0.
    """
    fb = splice_corpus_caller_callee
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        caller_idx = _caller_idx(fb)
        depth0 = sess.splice_with_callees(
            caller_idx,
            arm="matched",
            max_depth=0,
            inlined_equivalent_call_targets_only=True,
        )[0]
        depth1 = sess.splice_with_callees(
            caller_idx,
            arm="matched",
            max_depth=1,
            inlined_equivalent_call_targets_only=True,
        )[0]
    assert depth1.real_tokens.shape[0] == depth0.real_tokens.shape[0]
    assert np.array_equal(depth1.real_tokens, depth0.real_tokens)
