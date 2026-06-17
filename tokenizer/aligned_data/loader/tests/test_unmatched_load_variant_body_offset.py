"""The public ``load_unmatched`` path slices each record's OWN variant body.

Companion to :mod:`test_unmatched_variant_body_offset` (the callee-walk
path). This pins the OTHER unmatched body-load path -- the public
``load_unmatched`` API via ``_load_unmatched_record_and_section`` -- onto
the same robust ``section.variants[slot].data_offset_shifted`` slice the
callee walk uses, so neither path depends on the writer's
emit-order==vref-order lock-step.

The writer emits the per-record index entries (``starts[]``) in ENCOUNTER
order but flushes the section's variant blocks SORTED by
``variant_ref_offset``. A corpus whose encounter order DESCENDS by
``variant_ref_offset`` therefore has its variant blocks REVERSED relative
to the index entries -- genuine drift. Pre-convergence the path sliced the
body at the positional ``starts[idx]`` (encounter order) while resolving
the per-slot axes against ``section.variants[idx - base]`` (sorted order),
so the body and axes referred to DIFFERENT variants and the now-removed
drift check fired ``ValueError``. After the convergence the body, axes, and
variant tokens all come from ``section.variants[slot]`` -- a single,
self-consistent source -- and the drifted corpus loads cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from ._corpus import (
    MatchedFunctionSpec,
    UnmatchedFunctionSpec,
    build_corpus_with_registry,
    make_simple_variant,
)
from ._session_fixture import (
    _FakeVocab,
    _VariantStubRegistry,
    write_variants_slim_csv,
)


# Distinct canonical-4 axes per record: three genuinely-distinct
# ``_variants.bin`` records, one per variant, at three distinct offsets.
_AXES = (
    {"arch": "x86_64", "compiler": "gcc", "compilerversion": "13.2.0", "opt": "-O2"},
    {"arch": "arm32", "compiler": "clang", "compilerversion": "5.0", "opt": "-O0"},
    {"arch": "riscv64", "compiler": "gcc", "compilerversion": "12.1.0", "opt": "-O3"},
)


def _write_distinct_variants_bin(
    base: Path, binary_name: str, vocab: _FakeVocab
) -> List[int]:
    """Lay down ``_variants.bin`` with one DISTINCT record per axis set.

    Returns the byte offset of each record in write order. Hand-laid (not
    via the fixture builder) because the resolver needs real
    ``encode_record`` bytes at each offset to round-trip back to the axes.
    """
    from tokenizer.variant_tokens.encoder import encode_record

    offsets: List[int] = []
    variants_path = base / f"{binary_name}_variants.bin"
    with open(variants_path, "wb") as f:
        for axes in _AXES:
            class _V:
                arch = axes["arch"]
                compiler = axes["compiler"]
                compilerversion = axes["compilerversion"]
                opt = axes["opt"]
                extra_metadata: Dict[str, Any] = {}

            offsets.append(f.tell())
            f.write(encode_record(_V(), vocab).tobytes())
    write_variants_slim_csv(
        base,
        binary_name,
        {
            off: f"{binary_name}-{ax['arch']}-{ax['compiler']}-"
            f"{ax['compilerversion']}-{ax['opt']}"
            for off, ax in zip(offsets, _AXES)
        },
    )
    return offsets


def _build_drifted_unmatched(tmp_path: Path) -> Dict[str, Any]:
    """Corpus: ONE unmatched function, 3 distinct-body versions, drifted.

    Each version has a distinct ``token_seed`` (distinct body record) and a
    distinct ``variant_ref_offset``. The versions are emitted so that the
    ENCOUNTER order DESCENDS by ``variant_ref_offset`` -- the writer's
    ascending sort then REVERSES the variant blocks relative to the index
    entries, so ``section.variants[j]`` is encounter-version ``2 - j``.
    """
    base = tmp_path
    binary_name = "driftbin"
    vocab = _FakeVocab(
        [
            "arch:x64", "arch:arm32", "arch:riscv64",
            "comp:gcc", "comp:clang",
            "cver:gcc:13.2.0", "cver:clang:5.0", "cver:gcc:12.1.0",
            "opt:O2", "opt:O0", "opt:O3",
        ]
    )
    variant_offsets = _write_distinct_variants_bin(base, binary_name, vocab)

    # Versions in encounter order 0,1,2; assign the variant offsets in
    # DESCENDING order so the writer's ascending sort reverses the blocks.
    u_vkeys = [("unmatched", i) for i in range(3)]
    versions = tuple(
        make_simple_variant(u_vkeys[i], token_seed=10 + i, n_tokens=4 + i)
        for i in range(3)
    )
    encounter_ref_offsets = list(reversed(variant_offsets))  # descending
    registry_map = {
        u_vkeys[i]: f"{encounter_ref_offsets[i]:x}" for i in range(3)
    }

    # A matched function anchors the unmatched region's start offset.
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
    registry_map[("matched", 0)] = f"{variant_offsets[0]:x}"
    registry_map[("matched", 1)] = f"{variant_offsets[0]:x}"

    unmatched_specs = (
        UnmatchedFunctionSpec(
            func_name="lonely_func", versions=versions, called=()
        ),
    )
    registry = _VariantStubRegistry(registry_map)
    build_corpus_with_registry(
        base, binary_name,
        matched=matched_specs, unmatched=unmatched_specs,
        variants=registry,
    )

    # ``section.variants`` is sorted ascending by variant_ref_offset, which
    # maps slot j -> encounter-version (2 - j). The per-slot expected body
    # follows that reversal.
    expected_tokens_by_slot = [versions[2 - j].tokens for j in range(3)]
    return {
        "base_path": base,
        "binary_name": binary_name,
        "vocab": vocab,
        "expected_tokens_by_slot": expected_tokens_by_slot,
    }


def _unmatched_base_idx(sess) -> Tuple[int, int]:
    """Return ``(base_record_idx, n_variants)`` for the single section."""
    arm = sess.get_metadata("unmatched_arm")
    section_offset = int(arm.section_starts[0])
    base_idx = sess._idx_for_section_offset(
        section_offset, SectionKind.UNMATCHED.value
    )
    section, _off = sess._unmatched_section_meta(base_idx)
    return base_idx, len(section.variants)


def _section_variant_offset(sess, base_idx: int, slot: int) -> int:
    """The byte offset of ``section.variants[slot]``'s own body record."""
    from tokenizer.aligned_data.index_format import ALIGNMENT_SHIFT

    section, _off = sess._unmatched_section_meta(base_idx)
    return section.variants[slot].data_offset_shifted << ALIGNMENT_SHIFT


def test_load_unmatched_slices_each_records_own_variant_body(tmp_path) -> None:
    """``load_unmatched(base + j)`` returns ``section.variants[j]``'s body.

    On a drifted corpus the pre-convergence path RAISED (the now-removed
    drift check ``starts[idx] == variant.data_offset_shifted << 4`` fired
    because encounter order != sorted order). The convergence slices the
    body at the variant block's own ``data_offset_shifted``, so each
    per-record load surfaces its sorted-slot body and records that slot's
    own body offset -- no drift dependence.
    """
    fb = _build_drifted_unmatched(tmp_path)
    ds = BinaryDataset(
        fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"]
    )
    expected_tokens = fb["expected_tokens_by_slot"]
    with ds.open_session() as sess:
        base_idx, n_variants = _unmatched_base_idx(sess)
        assert n_variants == 3

        for slot in range(n_variants):
            fd = sess.load_unmatched(base_idx + slot)
            np.testing.assert_array_equal(
                fd.tokens, expected_tokens[slot],
                err_msg=(
                    f"load_unmatched(base+{slot}) spliced the wrong body: "
                    f"got {fd.tokens.tolist()}, "
                    f"want {expected_tokens[slot].tolist()}"
                ),
            )
            # ``data_offset`` (the per-record body locator the builder
            # records) is the slot's OWN variant block offset, not the
            # positional ``starts[base + slot]`` -- the property the drift
            # corpus exercises. The resolver-vocab axis round-trip is a
            # separate concern (test_unmatched_metadata_recovery).
            assert fd.metadata["data_offset"] == (
                _section_variant_offset(sess, base_idx, slot)
            )


def test_load_unmatched_record_and_section_matches_load_unmatched(tmp_path) -> None:
    """The internal ``_load_unmatched_record_and_section`` agrees with the
    public ``load_unmatched`` on the drifted corpus, and reports the owning
    section's BIN offset (not the per-record data offset)."""
    fb = _build_drifted_unmatched(tmp_path)
    ds = BinaryDataset(
        fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"]
    )
    with ds.open_session() as sess:
        base_idx, n_variants = _unmatched_base_idx(sess)
        section_offset = int(
            sess.get_metadata("unmatched_arm").section_starts[0]
        )
        for slot in range(n_variants):
            sec, off, fd = sess._load_unmatched_record_and_section(
                base_idx + slot
            )
            public_fd = sess.load_unmatched(base_idx + slot)
            assert off == section_offset
            assert int(sec.section_offset) == section_offset
            np.testing.assert_array_equal(fd.tokens, public_fd.tokens)
