"""Synthetic per-binary memmap fixtures for the sorted-index test suite.

Single concern: lay down per-binary memmap directories covering the
edge cases the sorted-index builder needs to handle correctly:

* a section with zero variants (ALG-1's pre-filter);
* a section with exactly one variant (degenerate percentile case);
* a section with three or more variants (multi-variant length
  aggregation);
* a section whose per-call entries point at a callee whose variant
  table does not carry the caller variant's vkey, so
  :data:`MISSING_VARIANT_INDEX` (``0xFFFE``) is stamped on the slot
  at :meth:`SectionWriter.finalize` time.

Each fixture composes :mod:`MatchedFunctionSpec` /
:mod:`UnmatchedFunctionSpec` inputs and drives them through the
existing :func:`build_corpus_with_registry` helper -- which in turn
runs the production pass-2 writers + :class:`SectionWriter`.  No
direct ``SectionWriter`` choreography lives here; the spec API is
sufficient for every edge case below, so the fixtures stay free of
duplicated wire-format knowledge.

Callers (Phase 1+ unit tests for the sorted-index builder + reader)
feed the returned directory into ``BinaryDataset(base, binary_name)``
or parse the ``<binary>_sections.bin`` blob directly via
:func:`tokenizer.aligned_data.matched_sections_bin.parse_section_bin`
to drive their assertions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    UnmatchedFunctionSpec,
    build_corpus_with_registry,
    make_simple_variant,
)
from tokenizer.aligned_data.loader.tests._corpus.specs import VariantSpec
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.variant_tokens.prefixes import build_axis_strings
from tokenizer.variant_tokens.record import write_record


# Every fixture writes into ``<tmp_path>/<binary_name>`` so callers can
# point a per-binary BinaryDataset at the returned path without picking
# binary names out of the dataset shell's filename glob themselves.
_BINARY_NAME: str = "sortbin"


def make_test_vocab_manager():
    """A minimal valid unified ``VocabularyManager`` for DECODE-path tests.

    ``batch_decode`` HARD-REQUIRES a vocab to assemble the variant-axis
    prefix (a vocab-less decode would silently drop it), so every fixture
    that drives a decode must thread one into its ``BinaryDataset`` /
    ``IndexedMemmapCollection.discover``. Length/graph-only fixtures never
    decode and stay vocab-less by design (``build_*_fixture`` deliberately
    omits the vocab). Built via the production v1 stager + gate so the head-
    of-vocab invariants hold; the staging dir is ephemeral (the loaded VM is
    self-contained).
    """
    import tempfile

    from tokenizer.aligned_data.loader.tests._loader_test_support import (
        stage_v1_unified_vocab,
    )
    from tokenizer.aligned_data.loader.unified_vocab_gate import (
        load_and_validate_unified_vocab,
    )

    with tempfile.TemporaryDirectory() as scratch:
        return load_and_validate_unified_vocab(
            stage_v1_unified_vocab(Path(scratch))
        )


class _DeterministicVariantRegistry:
    """Registry that assigns each unique ``vkey`` a 16-byte-aligned offset.

    Mirrors the ``_StubVariantRegistry`` used by
    :mod:`.._corpus.builder` but lives here so the fixtures own their
    own registry instance (and the starting offset is non-zero so the
    on-disk MISSING-stamped slots are unambiguous in test assertions
    -- ``variant_ref_offset == 0`` would alias an unset placeholder).
    """

    def __init__(self) -> None:
        self._counter = 1  # start at 1 so the first offset is 0x10 (non-zero).
        self._refs: dict = {}

    def _ensure(self, vkey) -> int:
        if vkey not in self._refs:
            self._refs[vkey] = self._counter * 0x10
            self._counter += 1
        return self._refs[vkey]

    def ref(self, vkey) -> str:
        return f"{self._ensure(vkey):x}"

    def byte_offset(self, vkey) -> int:
        return self._ensure(vkey)


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def _variants_for(
    func_name: str, n: int, *, seed_base: int = 0
) -> Tuple[VariantSpec, ...]:
    """Build ``n`` distinct variants for ``func_name`` with stable vkeys.

    The ``token_seed`` is offset by ``seed_base`` so multiple fixtures
    can co-exist in one combined corpus without colliding on data-bin
    offsets -- distinct ``token_seed`` values produce distinct token
    streams + distinct on-disk record offsets, which the pass-1
    dedup-by-data-offset heuristic requires.
    """
    return tuple(
        make_simple_variant(
            (func_name, i),
            token_seed=seed_base + i + 1,
            n_tokens=8 + i,
        )
        for i in range(n)
    )


# ---------------------------------------------------------------------------
# Individual edge-case fixtures
# ---------------------------------------------------------------------------


def build_0_variant_section_fixture(tmp_path: Path) -> Path:
    """Memmap dir whose matched arm has at least one zero-variant section.

    Two matched functions are emitted:

    * ``func_zero`` -- declared with ``variants=()`` so
      :meth:`SectionWriter.begin_section` reserves a zero-wide jump
      table and the section's ``n_variants`` field stays ``0`` after
      :meth:`SectionWriter.end_section`. This is the section ALG-1's
      pre-filter must skip without invoking
      ``_select_variant_indices`` (which raises on ``n_variants <= 0``).
    * ``func_one`` -- a single-variant companion so the matched arm
      carries at least one non-degenerate section alongside the
      zero-variant one (the matched-arm loader walks every entry in
      ``matched_index.bin`` in encounter order, so the zero-variant
      section is index 0 and the single-variant section is index 1).

    The unmatched arm is intentionally empty -- pass-2 emits its
    sections after the matched arm regardless, and the matched edge
    case lives entirely on the matched side.
    """
    base = tmp_path / "zero_variant"
    base.mkdir(parents=True, exist_ok=True)
    matched = (
        MatchedFunctionSpec(func_name="func_zero", variants=(), called=()),
        MatchedFunctionSpec(
            func_name="func_one",
            variants=_variants_for("func_one", 1, seed_base=10),
            called=(),
        ),
    )
    build_corpus_with_registry(
        base,
        _BINARY_NAME,
        matched=matched,
        unmatched=(),
        variants=_DeterministicVariantRegistry(),
    )
    return base


def build_1_variant_section_fixture(tmp_path: Path) -> Path:
    """Memmap dir whose matched arm carries a single-variant section.

    The degenerate-percentile case ALG-1 must collapse to the one
    variant's length verbatim. The fixture lays down two such sections
    (different function names + different ``token_seed`` ranges so the
    BIN catalog has two distinct singleton entries to assert on).
    """
    base = tmp_path / "one_variant"
    base.mkdir(parents=True, exist_ok=True)
    matched = (
        MatchedFunctionSpec(
            func_name="solo_a",
            variants=_variants_for("solo_a", 1, seed_base=0),
            called=(),
        ),
        MatchedFunctionSpec(
            func_name="solo_b",
            variants=_variants_for("solo_b", 1, seed_base=20),
            called=(),
        ),
    )
    build_corpus_with_registry(
        base,
        _BINARY_NAME,
        matched=matched,
        unmatched=(),
        variants=_DeterministicVariantRegistry(),
    )
    return base


def build_many_variant_section_fixture(tmp_path: Path) -> Path:
    """Memmap dir whose matched arm carries a multi-variant section.

    ALG-1's multi-variant aggregation path (max + percentile reductions)
    needs at least three variants to distinguish ``p50`` from ``max``
    on a non-uniform length distribution. The fixture emits one
    four-variant function (``token_seed`` strictly increasing so the
    on-disk record sizes diverge) plus a single-variant companion so
    the BIN catalog has more than one entry.
    """
    base = tmp_path / "many_variant"
    base.mkdir(parents=True, exist_ok=True)
    matched = (
        MatchedFunctionSpec(
            func_name="multi_fn",
            variants=_variants_for("multi_fn", 4, seed_base=0),
            called=(),
        ),
        MatchedFunctionSpec(
            func_name="solo_companion",
            variants=_variants_for("solo_companion", 1, seed_base=30),
            called=(),
        ),
    )
    build_corpus_with_registry(
        base,
        _BINARY_NAME,
        matched=matched,
        unmatched=(),
        variants=_DeterministicVariantRegistry(),
    )
    return base


def build_missing_variant_index_fixture(tmp_path: Path) -> Path:
    """Memmap dir whose matched arm carries a section with a MISSING per-call slot.

    Setup:

    * ``caller_fn`` has one variant ``("caller_fn", 0)`` (vkey offset
      ``X``) and declares ``called=("callee_fn",)``.
    * ``callee_fn`` has two variants ``("callee_fn", 0)`` /
      ``("callee_fn", 1)`` with vkey offsets ``Y`` / ``Z`` (both
      distinct from ``X``).

    The pass-2 writer emits a single per-call entry on
    ``caller_fn``'s variant whose ``callee_vkey`` is the caller's own
    variant_ref_offset ``X`` (the Step-7 invariant pinned by
    ``_emit_variant_per_call_entries``).  At the callee's
    :meth:`SectionWriter.end_section`, the sibling-close walker
    re-parses the callee's variant table, finds no ``X`` entry, and
    leaves the slot ``UNRESOLVED``.  At :meth:`SectionWriter.finalize`
    the remaining-holes sweep stamps :data:`MISSING_VARIANT_INDEX`
    (``0xFFFE``) on the slot.

    The fixture's `caller_fn` section therefore has at least one
    per-call entry whose on-disk ``section_variant_index`` is ``0xFFFE``
    -- the case the ALG-1 length-walker must treat as "no inlined
    callee body" rather than indexing into the callee's variant array.
    """
    base = tmp_path / "missing_variant_index"
    base.mkdir(parents=True, exist_ok=True)
    matched = (
        MatchedFunctionSpec(
            func_name="caller_fn",
            variants=_variants_for("caller_fn", 1, seed_base=0),
            called=("callee_fn",),
        ),
        MatchedFunctionSpec(
            func_name="callee_fn",
            variants=_variants_for("callee_fn", 2, seed_base=40),
            called=(),
        ),
    )
    build_corpus_with_registry(
        base,
        _BINARY_NAME,
        matched=matched,
        unmatched=(),
        variants=_DeterministicVariantRegistry(),
    )
    return base


# ---------------------------------------------------------------------------
# Real variant-axis prefix wiring (decode-path fixtures)
# ---------------------------------------------------------------------------


class _VersionInfo:
    """``encode_record`` duck-type: one fully-specified variant identity.

    Carries the four positional axes (``arch`` / ``compiler`` /
    ``compilerversion`` / ``opt``) plus one metadata k/v so the encoded
    record is a MULTI-token axis (six tokens here), not a degenerate
    single-token prefix. A multi-token prefix is what makes the e2e's
    content assertion non-vacuous and catches an off-by-one that drops
    the LAST prefix token (a shape-only assertion would not).
    """

    arch = "x86_64"
    compiler = "gcc"
    compilerversion = "13.2.0"
    opt = "-O2"
    extra_metadata = {"lang": "c"}


class _SingleRecordVariantRegistry:
    """Registry whose every ``vkey`` resolves to one shared bin record.

    The companion to :func:`write_combined_variants_bin`: every
    section-row ``variant_ref`` cell (CSV) and ``variant_ref_offset``
    (BIN) points at the single :func:`write_record` output laid at
    ``record_offset``. The decode path therefore resolves a NON-EMPTY
    multi-token axis for every variant in the corpus -- the prefix
    content the e2e asserts on. Mirrors the
    ``_session_fixture._VariantStubRegistry`` idiom (all refs -> one
    record) so the resolver round-trip exercises the real axis decoder.
    """

    def __init__(self, record_offset: int) -> None:
        self._offset = record_offset
        self._hex = f"{record_offset:x}"

    def ref(self, vkey) -> str:
        return self._hex

    def byte_offset(self, vkey) -> int:
        return self._offset


def write_combined_variants_bin(
    base: Path, binary_name: str, vocab_manager
) -> int:
    """Lay down ``<binary>_variants.bin`` + ``_variants.csv``; return the offset.

    Single concern: produce the real on-disk variant sidecars the
    ``BinaryDataset`` discovery path reads so a decoded batch carries a
    non-empty variant-axis prefix. One :func:`write_record` record is
    written at byte 0 of the bin (encoded against ``vocab_manager`` via
    the production encoder); the slim CSV gives the resolver its
    ``offset -> filename`` row. The axis strings are registered on
    ``vocab_manager`` first (idempotent ``Variant_Axis`` adds) so
    ``encode_record``'s hard lookup succeeds -- the same VM must then be
    threaded into the session so encode/decode share one id map.

    Returns the byte offset (0) every section ``variant_ref`` should
    cite; pair with :class:`_SingleRecordVariantRegistry`.
    """
    version_info = _VersionInfo()
    # Register every axis string the record will reference (positional
    # axes + the metadata k/v). ``Variant_Axis`` is idempotent, so this
    # is a no-op for strings the staged vocab already carries.
    for axis_string in build_axis_strings(version_info):
        vocab_manager.Variant_Axis(axis_string)

    variants_path = base / f"{binary_name}_variants.bin"
    with open(variants_path, "wb") as handle:
        offset = write_record(handle, version_info, vocab_manager)

    # Slim companion CSV: ``# format=N`` prelude + ``filename,variant_id,
    # offset`` header + one row mapping the record offset to a filename
    # (the resolver's ``offset_to_filename`` lookup, KeyError on a miss).
    csv_path = base / f"{binary_name}_variants.csv"
    with open(csv_path, "w", newline="", encoding="ascii") as csv_handle:
        write_csv_prelude(csv_handle)
        csv_handle.write("filename,variant_id,offset\n")
        csv_handle.write(f"{binary_name}-x86_64-gcc-13.2.0-O2,00000000,{offset:x}\n")
    return offset


def build_combined_fixture_with_variants(
    tmp_path: Path, vocab_manager
) -> Path:
    """Combined fixture wired to a REAL ``_variants.bin`` prefix record.

    Same five-section structure as :func:`build_combined_fixture` but
    every variant_ref points at one shared multi-token axis record (laid
    by :func:`write_combined_variants_bin`) instead of the placeholder
    offsets the default registry stamps. The decode path therefore
    resolves a non-empty variant-axis prefix for every section -- the
    precondition for asserting prefix CONTENT (vs. the
    empty-prefix blind spot of the default fixture, whose corpus builder
    omits ``_variants.bin`` entirely).

    ``vocab_manager`` is mutated (axis strings registered) and MUST be
    the same VM threaded into the decoding session so the record's
    vocab ids resolve.
    """
    base = tmp_path / "combined_variants"
    base.mkdir(parents=True, exist_ok=True)
    record_offset = write_combined_variants_bin(base, _BINARY_NAME, vocab_manager)
    matched = (
        MatchedFunctionSpec(func_name="func_zero", variants=(), called=()),
        MatchedFunctionSpec(
            func_name="solo_a",
            variants=_variants_for("solo_a", 1, seed_base=0),
            called=(),
        ),
        MatchedFunctionSpec(
            func_name="multi_fn",
            variants=_variants_for("multi_fn", 4, seed_base=10),
            called=(),
        ),
        MatchedFunctionSpec(
            func_name="caller_fn",
            variants=_variants_for("caller_fn", 1, seed_base=20),
            called=("callee_fn",),
        ),
        MatchedFunctionSpec(
            func_name="callee_fn",
            variants=_variants_for("callee_fn", 2, seed_base=30),
            called=(),
        ),
    )
    build_corpus_with_registry(
        base,
        _BINARY_NAME,
        matched=matched,
        unmatched=(),
        variants=_SingleRecordVariantRegistry(record_offset),
    )
    return base


# ---------------------------------------------------------------------------
# Combined fixture (optional convenience for integration tests)
# ---------------------------------------------------------------------------


def build_combined_fixture(tmp_path: Path) -> Path:
    """Memmap dir combining every edge case in one corpus.

    Useful for builder + CLI integration tests that want to exercise
    all four ALG-1 branches against a single memmap directory rather
    than four separate ones.  The section order in
    ``matched_index.bin`` is the spec order below.
    """
    base = tmp_path / "combined"
    base.mkdir(parents=True, exist_ok=True)
    matched = (
        # 0-variant edge case.
        MatchedFunctionSpec(func_name="func_zero", variants=(), called=()),
        # 1-variant edge case.
        MatchedFunctionSpec(
            func_name="solo_a",
            variants=_variants_for("solo_a", 1, seed_base=0),
            called=(),
        ),
        # Many-variant edge case.
        MatchedFunctionSpec(
            func_name="multi_fn",
            variants=_variants_for("multi_fn", 4, seed_base=10),
            called=(),
        ),
        # MISSING_VARIANT_INDEX edge case: caller -> callee with
        # disjoint vkey sets. ``caller_fn`` closes first and its
        # per-call entry resolves to UNRESOLVED; ``callee_fn`` then
        # closes (no matching vkey in its variant table) and the slot
        # is stamped MISSING at finalize.
        MatchedFunctionSpec(
            func_name="caller_fn",
            variants=_variants_for("caller_fn", 1, seed_base=20),
            called=("callee_fn",),
        ),
        MatchedFunctionSpec(
            func_name="callee_fn",
            variants=_variants_for("callee_fn", 2, seed_base=30),
            called=(),
        ),
    )
    build_corpus_with_registry(
        base,
        _BINARY_NAME,
        matched=matched,
        unmatched=(),
        variants=_DeterministicVariantRegistry(),
    )
    return base


__all__ = (
    "build_0_variant_section_fixture",
    "build_1_variant_section_fixture",
    "build_many_variant_section_fixture",
    "build_missing_variant_index_fixture",
    "build_combined_fixture",
    "build_combined_fixture_with_variants",
    "write_combined_variants_bin",
)
