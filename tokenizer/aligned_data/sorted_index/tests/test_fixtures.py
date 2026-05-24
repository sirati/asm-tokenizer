"""Smoke tests for the sorted-index test fixtures.

These tests prove that each fixture builds a valid per-binary memmap
directory and exposes the edge case it advertises.  They are NOT
sorted-index logic tests -- Phase 1+ unit tests
(``test_length_compute.py`` / ``test_builder.py`` / ...) consume the
fixtures and assert on the builder's outputs.

The smokes verify three things per fixture:

* ``BinaryDataset(base, binary_name)`` opens cleanly (the function-
  names sidecar + matched/unmatched arm loaders accept the on-disk
  byte layout).
* ``BinaryDataset.open_session()`` enters + exits cleanly (the
  ``sections.bin`` prelude + ``_data.bin`` prelude validate on open).
* The targeted section's parsed bytes carry the edge-case the
  fixture's docstring claims (variant count or
  :data:`MISSING_VARIANT_INDEX` slot).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.aligned_data.loader._sections_bin_walk import (
    read_sections_bin_blob,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.matched_sections_bin import (
    MISSING_VARIANT_INDEX,
    parse_section_bin,
)

from .fixtures import (
    build_0_variant_section_fixture,
    build_1_variant_section_fixture,
    build_combined_fixture,
    build_many_variant_section_fixture,
    build_missing_variant_index_fixture,
)


_BINARY_NAME = "sortbin"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_dataset(base: Path) -> BinaryDataset:
    """Construct a :class:`BinaryDataset` against the fixture directory.

    ``vocab_manager=None`` is sufficient for the smoke: the matched-arm
    loader does not consult the vocab, and the smoke does not call
    :meth:`BinarySession.get_variant_by_ref` (the one path that
    requires it).
    """
    return BinaryDataset(base, _BINARY_NAME, vocab_manager=None)


def _parse_section(dataset: BinaryDataset, idx: int):
    """Re-parse the BIN section at ``matched_arm.bin_starts[idx]``.

    Goes through the public :func:`parse_section_bin` codec rather than
    :meth:`BinarySession.load_matched` because some edge-case sections
    (notably the 0-variant fixture) have no variants for
    ``parse_matched_section`` to walk.
    """
    bin_starts = dataset.matched_bin_starts
    assert idx < len(bin_starts), (
        f"idx {idx} out of range; matched arm has {len(bin_starts)} sections"
    )
    section_offset = int(bin_starts[idx])
    _raw, blob = read_sections_bin_blob(dataset.matched_sections_bin)
    section, _end = parse_section_bin(blob, section_offset)
    return section


# ---------------------------------------------------------------------------
# Individual fixtures
# ---------------------------------------------------------------------------


def test_zero_variant_fixture_loads(tmp_path: Path) -> None:
    base = build_0_variant_section_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session():
        pass  # session-open + close smoke; no inner asserts needed here.

    # Section[0] is ``func_zero`` -- declared with zero variants.
    section = _parse_section(dataset, 0)
    assert len(section.variants) == 0
    # The companion ``func_one`` is index 1 with one variant; sanity-
    # check it so the fixture genuinely tests "0-variant in a corpus
    # with other sections" rather than "everything is 0-variant".
    companion = _parse_section(dataset, 1)
    assert len(companion.variants) == 1


def test_one_variant_fixture_loads(tmp_path: Path) -> None:
    base = build_1_variant_section_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session():
        pass

    for idx in range(len(dataset.matched_bin_starts)):
        section = _parse_section(dataset, idx)
        assert len(section.variants) == 1


def test_many_variant_fixture_loads(tmp_path: Path) -> None:
    base = build_many_variant_section_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session():
        pass

    multi = _parse_section(dataset, 0)
    assert len(multi.variants) >= 3, (
        f"many-variant fixture should expose >= 3 variants in section[0]; "
        f"got {len(multi.variants)}"
    )


def test_missing_variant_index_fixture_loads(tmp_path: Path) -> None:
    base = build_missing_variant_index_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session():
        pass

    # ``caller_fn`` is section[0] -- its single variant's per-call
    # entries must carry at least one MISSING_VARIANT_INDEX slot.
    caller = _parse_section(dataset, 0)
    assert len(caller.variants) == 1
    per_call = caller.variants[0].per_call_entries
    assert per_call, "caller variant should have at least one per-call entry"
    missing_slots = [
        sv_idx
        for _called_idx, sv_idx in per_call
        if sv_idx == MISSING_VARIANT_INDEX
    ]
    assert missing_slots, (
        f"expected >= 1 MISSING_VARIANT_INDEX (0x{MISSING_VARIANT_INDEX:04X}) "
        f"slot in caller_fn's per_call_entries; got {per_call!r}"
    )

    # ``callee_fn`` is section[1] -- two variants whose own
    # ``variant_ref_offset`` values do NOT include the caller's vkey,
    # so the MISSING stamp on the caller's slot is structurally
    # correct rather than a writer accident.
    callee = _parse_section(dataset, 1)
    assert len(callee.variants) == 2
    callee_vrefs = {v.variant_ref_offset for v in callee.variants}
    caller_vref = caller.variants[0].variant_ref_offset
    assert caller_vref not in callee_vrefs, (
        f"caller vref {caller_vref!r} unexpectedly present in callee variant "
        f"table {callee_vrefs!r}; MISSING fixture invariant violated"
    )


# ---------------------------------------------------------------------------
# Combined fixture (sanity)
# ---------------------------------------------------------------------------


def test_combined_fixture_covers_every_edge_case(tmp_path: Path) -> None:
    base = build_combined_fixture(tmp_path)
    dataset = _open_dataset(base)
    with dataset.open_session():
        pass

    sections = [
        _parse_section(dataset, idx)
        for idx in range(len(dataset.matched_bin_starts))
    ]
    variant_counts = sorted(len(s.variants) for s in sections)
    # Spec order in :func:`build_combined_fixture`: 0, 1, 4, 1 (caller),
    # 2 (callee). Sorted: [0, 1, 1, 2, 4].
    assert variant_counts == [0, 1, 1, 2, 4], (
        f"combined fixture variant-count signature drifted from spec; "
        f"got {variant_counts}"
    )
    # MISSING slot must appear somewhere across the combined corpus.
    has_missing = any(
        sv_idx == MISSING_VARIANT_INDEX
        for s in sections
        for v in s.variants
        for _called_idx, sv_idx in v.per_call_entries
    )
    assert has_missing, (
        "combined fixture lost the MISSING_VARIANT_INDEX slot it advertises"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        build_0_variant_section_fixture,
        build_1_variant_section_fixture,
        build_many_variant_section_fixture,
        build_missing_variant_index_fixture,
        build_combined_fixture,
    ],
)
def test_fixture_is_deterministic(tmp_path: Path, builder) -> None:
    """Build the fixture twice into separate dirs; byte-equal output.

    No RNG is used inside the fixture -- both the spec helpers and the
    deterministic registry produce identical bytes for identical
    inputs. This catches accidental introduction of nondeterminism in
    future edits (e.g. swapping the dict-keyed registry for a
    set-iteration based one).
    """
    base_a = builder(tmp_path / "a")
    base_b = builder(tmp_path / "b")
    sections_a = (base_a / f"{_BINARY_NAME}_sections.bin").read_bytes()
    sections_b = (base_b / f"{_BINARY_NAME}_sections.bin").read_bytes()
    assert sections_a == sections_b, (
        "fixture rebuild produced different sections.bin bytes; "
        "fixtures must be deterministic so test assertions stay stable"
    )
