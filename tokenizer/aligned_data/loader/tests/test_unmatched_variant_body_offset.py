"""Unmatched J-variant body load slices the J-variant's OWN data record.

An unmatched section stores one DISTINCT body record per variant (each
variant block carries its own ``data_offset_shifted``). The callee walk's
J-resolution can pick any variant, so the body load MUST slice the
J-resolved variant's own record -- NOT the section's first record.

This pins the audit probe: a 3-version unmatched section whose bodies
DIFFER, loaded at variant index 2, must surface variant-2's body. The
pre-fix HEAD (``f882dc0``) always loaded the first record (variant 0), so
``load_callee_body`` for a J=2 edge spliced variant-0's bytes -- the bug
this test fails on and the fix closes. The variant-0 load stays
byte-identical to the legacy behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._callee_walk._resolve import (
    ResolvedCalleeMeta,
    load_callee_body,
)
from tokenizer.aligned_data.loader.batch_decode._callee_walk._walker import (
    _load_root_bodies,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from ._corpus import (
    UnmatchedFunctionSpec,
    build_corpus_with_registry,
    make_simple_variant,
)
from ._session_fixture import (
    _FakeArm,
    _FakeVocab,
    _VariantStubRegistry,
    _write_variants_bin,
)


def _build_multi_variant_unmatched(tmp_path: Path) -> Dict[str, Any]:
    """Corpus with ONE unmatched function carrying 3 distinct-body versions.

    Each version has a distinct ``token_seed`` -> distinct token stream ->
    distinct ``_unmatched_data.bin`` record + variant-block
    ``data_offset_shifted``. All versions point at the single hand-laid
    ``_variants.bin`` record (axes identical); the bodies differ, which is
    the property the load must respect.
    """
    from tokenizer.aligned_data.csv_section_index import (
        read_csv_section_index_arrays,
    )
    from tokenizer.aligned_data.index_format import read_index_arrays
    from tokenizer.aligned_data.loader._sections_bin_walk import (
        unmatched_region_start,
    )

    base = tmp_path
    binary_name = "multibin"
    vocab = _FakeVocab(["arch:x64", "comp:gcc", "cver:gcc:13.2.0", "opt:O2"])
    variant_offset = _write_variants_bin(base, binary_name, vocab)
    variant_ref_hex = f"{variant_offset:x}"

    # 3 versions, distinct token seeds -> distinct bodies + offsets.
    u_vkeys = [("unmatched", i) for i in range(3)]
    versions = tuple(
        make_simple_variant(u_vkeys[i], token_seed=10 + i, n_tokens=4 + i)
        for i in range(3)
    )
    unmatched_specs = (
        UnmatchedFunctionSpec(
            func_name="lonely_func", versions=versions, called=()
        ),
    )
    # A matched function is needed so the unmatched region has a stable
    # start offset (mirrors the shared fixture's layout).
    from ._corpus import MatchedFunctionSpec

    matched_specs = (
        MatchedFunctionSpec(
            func_name="my_func",
            variants=(
                make_simple_variant(("matched", 0), token_seed=1, n_tokens=8),
                make_simple_variant(("matched", 1), token_seed=2, n_tokens=6),
            ),
            called=(),
        ),
    )

    registry = _VariantStubRegistry(
        {vk: variant_ref_hex for vk in u_vkeys}
        | {("matched", 0): variant_ref_hex, ("matched", 1): variant_ref_hex}
    )
    corpus = build_corpus_with_registry(
        base, binary_name,
        matched=matched_specs, unmatched=unmatched_specs,
        variants=registry,
    )

    starts = read_index_arrays(corpus.unmatched_index_bin)
    assert starts is not None and len(starts) == 3
    unmatched_section_offset = unmatched_region_start(corpus.matched_index_bin)
    record_to_section_idx = np.zeros(len(starts), dtype=np.uint32)
    unmatched_arm = _FakeArm(
        starts=starts,
        func_names=["lonely_func"],
        section_starts=np.array([unmatched_section_offset], dtype=np.int64),
        record_to_section_idx=record_to_section_idx,
    )

    m_pair = read_csv_section_index_arrays(corpus.matched_index_bin)
    assert m_pair is not None
    bin_starts, bin_lengths = m_pair
    matched_arm = _FakeArm(
        starts=np.zeros(0, dtype=np.int64),
        func_names=["my_func"],
        bin_starts=bin_starts,
        bin_lengths=bin_lengths,
    )

    from tokenizer.aligned_data.loader.function_names_loader import (
        load_function_names,
    )
    from tokenizer.aligned_data.loader.extern_providers_loader import (
        load_extern_providers,
    )
    _, line_to_name = load_function_names(corpus.function_names_sidecar)
    line_to_provider = load_extern_providers(corpus.extern_providers_sidecar)

    metadata = {
        "matched_arm": matched_arm,
        "unmatched_arm": unmatched_arm,
        "offset_to_filename": {variant_offset: "multibin-x64-gcc-13.2.0-O2"},
        "line_to_name": line_to_name,
        "line_to_provider": line_to_provider,
    }
    return {
        "base_path": base,
        "binary_name": binary_name,
        "vocab": vocab,
        "metadata": metadata,
        "expected_tokens": [v.tokens for v in versions],
    }


def test_unmatched_callee_body_loads_jresolved_variant(tmp_path) -> None:
    """``load_callee_body`` on a J=2 unmatched edge splices variant-2's body.

    FAILS on f882dc0 (always loaded the first record -> variant-0 bytes),
    PASSES on the data_offset_shifted slice.
    """
    fb = _build_multi_variant_unmatched(tmp_path)
    ds = BinaryDataset(
        fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"]
    )
    expected = fb["expected_tokens"]
    with ds.open_session() as sess:
        base_idx = sess._idx_for_section_offset(
            sess.get_metadata("unmatched_arm").section_starts[0],
            SectionKind.UNMATCHED.value,
        )
        section, _offset = sess._unmatched_section_meta(base_idx)
        assert len(section.variants) == 3

        for j in range(3):
            meta = ResolvedCalleeMeta(
                section=section,
                variant_idx=j,
                section_offset=int(section.section_offset),
                function_name_ptr=0,
                call_target_type=CallTargetType.LOCAL,
                callee_idx=base_idx,
            )
            body = load_callee_body(sess, SectionKind.UNMATCHED, meta)
            np.testing.assert_array_equal(
                body.tokens, expected[j],
                err_msg=(
                    f"unmatched J={j} edge spliced the wrong body: "
                    f"got {body.tokens.tolist()}, want {expected[j].tolist()}"
                ),
            )


def test_unmatched_root_bodies_one_per_variant(tmp_path) -> None:
    """``_load_root_bodies`` returns one body per unmatched variant.

    Post-revert it returned a single-element list -> an IndexError when a
    root is sampled at slot > 0. The list must be parallel to
    ``section.variants`` with each body being that variant's own record.
    """
    fb = _build_multi_variant_unmatched(tmp_path)
    ds = BinaryDataset(
        fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"]
    )
    expected = fb["expected_tokens"]
    with ds.open_session() as sess:
        base_idx = sess._idx_for_section_offset(
            sess.get_metadata("unmatched_arm").section_starts[0],
            SectionKind.UNMATCHED.value,
        )
        section, _offset = sess._unmatched_section_meta(base_idx)
        bodies = _load_root_bodies(sess, SectionKind.UNMATCHED, section)
        assert len(bodies) == 3
        for j in range(3):
            np.testing.assert_array_equal(bodies[j].tokens, expected[j])
