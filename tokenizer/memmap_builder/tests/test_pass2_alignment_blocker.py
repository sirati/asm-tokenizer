"""Matched-arm BIN section alignment invariants.

The matched-section writer drives :class:`SectionWriter`, which pads
each section trailer up to a 4-byte boundary so the per-section
``(bin_offset, bin_section_length)`` pair stored in
``<binary>_matched_index.bin`` is 4-byte aligned and the codec's
``>> 2`` shift stays valid. We pin the invariants here so a future
refactor can't silently drop the padding step and rely on coincidental
alignment.

What we assert per fixture:

  * Every ``bin_starts[i] % 4 == 0`` (the matched index codec requires
    it; double-checked at the array level).
  * Every ``bin_lengths[i] % 4 == 0`` (same).
  * Sections partition the BIN's matched region without gaps or
    overlap: ``bin_starts[i] + bin_lengths[i] == bin_starts[i+1]``.
"""

from __future__ import annotations

import io
from pathlib import Path

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.matched_sections_bin import SectionWriter
from tokenizer.aligned_data.memmap_format import (
    MATCHED_SECTIONS_BIN_PRELUDE_SIZE,
)
from tokenizer.memmap_builder._pass2 import write_matched_sections_pass2
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry


class _StubVariantRegistry:
    """Bare ``.ref(vkey) -> str`` / ``.byte_offset`` surface — pass 2's
    matched-arm CSV path needs the hex string for the variant_ref cell;
    the BIN path needs the integer for the variant_ref_offset slot.
    Both derive from the same per-vkey crc32 so the values are stable
    across runs without dragging the unified vocab encoder in.
    """

    def __init__(self) -> None:
        import zlib
        self._zlib = zlib

    def ref(self, vkey) -> str:
        return f"0x{self._zlib.crc32(repr(vkey).encode()) & 0xFFFF:x}"

    def byte_offset(self, vkey) -> int:
        return self._zlib.crc32(repr(vkey).encode()) & 0xFFFFFFFF


def _build_registry(*names: str) -> FunctionNamesRegistry:
    reg = FunctionNamesRegistry()
    for name in names:
        reg.add(name)
    reg.finalize()
    return reg


def test_matched_arm_one_function_section_is_4_aligned(
    tmp_path: Path,
) -> None:
    """One matched function -> the single section start equals the BIN
    prelude size (the first byte after the 16-byte file prelude), and
    the section length is a 4-byte multiple (the section trailer is
    padded by :class:`SectionWriter` up to the next 4-byte boundary)."""
    # 17-char function name -> the section's unpadded length is
    # non-4-aligned; the writer's padding step must round it up.
    funcname_17 = "a_seventeen_chars"
    assert len(funcname_17) == 17
    called_name = "another_one"

    registry = _build_registry(funcname_17, called_name)

    # Use EXTERN so the BIN's call_target resolves without needing a
    # callee section in the fixture (the alignment test exercises BIN
    # padding, not cross-section linking).
    called_typed = (called_name, CallTargetType.EXTERN)
    matched_data_entries = [
        {
            "func_name": funcname_17,
            "unique_called": [called_typed],
            "extern_libraries": {},
            "version_data": [
                {
                    "vkey": ("v0",),
                    "called": {called_typed},
                    "data_offset": 0,
                    "data_len": 16,
                    "token_len": 8,
                },
                {
                    "vkey": ("v1",),
                    "called": {called_typed},
                    "data_offset": 16,
                    "data_len": 32,
                    "token_len": 12,
                },
            ],
        },
    ]

    function_lookup = {
        (called_name, ("v0",)): (1024, 32, 0),
        (called_name, ("v1",)): (1056, 32, 0),
    }

    sections_path = tmp_path / "demo_sections.csv"
    index_path = tmp_path / "demo_matched_index.bin"
    bin_path = tmp_path / "demo_sections.bin"

    with open(sections_path, "w", newline="", encoding="ascii") as sf, \
         open(index_path, "wb") as idxf:
        write_csv_prelude(sf)
        warn_log = io.StringIO()
        section_writer = SectionWriter(bin_path)
        try:
            write_matched_sections_pass2(
                matched_data_entries,
                function_lookup,
                sf,
                idxf,
                warn_log,
                _StubVariantRegistry(),
                registry,
                section_writer,
                ExternProviderRegistry(),
                matched_func_names={funcname_17},
                sectioned_func_names={funcname_17},
            )
            section_writer.finalize()
        except BaseException:
            section_writer.close()
            raise

    pair = read_csv_section_index_arrays(index_path)
    assert pair is not None, "matched_index.bin should be non-empty"
    bin_starts, bin_lengths = pair

    # Exactly one function -> exactly one index entry.
    assert bin_starts.shape == (1,)
    assert bin_lengths.shape == (1,)

    bin_offset = int(bin_starts[0])
    bin_section_length = int(bin_lengths[0])

    # Alignment invariants.
    assert bin_offset % 4 == 0, (
        f"bin_starts[0]={bin_offset} must be 4-byte aligned"
    )
    assert bin_section_length % 4 == 0, (
        f"bin_lengths[0]={bin_section_length} must be 4-byte aligned"
    )

    # First section starts right after the file-level prelude; one
    # function -> the section runs to EOF.
    bin_size = bin_path.stat().st_size
    assert bin_offset == MATCHED_SECTIONS_BIN_PRELUDE_SIZE
    assert bin_offset + bin_section_length == bin_size


def test_matched_arm_two_functions_every_section_is_4_aligned(
    tmp_path: Path,
) -> None:
    """Two matched functions with mismatched name lengths -> both BIN
    sections land on 4-byte boundaries and abut without gaps.
    """
    fn_a = "a_seventeen_chars"  # 17 chars
    fn_b = "b_thirteen_ch"      # 13 chars
    assert len(fn_a) == 17
    assert len(fn_b) == 13
    called = "common_callee"

    registry = _build_registry(fn_a, fn_b, called)

    # Use EXTERN so the BIN's call_target resolves without a callee
    # section emit.
    called_typed = (called, CallTargetType.EXTERN)
    matched_data_entries = []
    for func_name, base_offset in ((fn_a, 0), (fn_b, 64)):
        matched_data_entries.append(
            {
                "func_name": func_name,
                "unique_called": [called_typed],
                "extern_libraries": {},
                "version_data": [
                    {
                        "vkey": (f"{func_name}-v0",),
                        "called": {called_typed},
                        "data_offset": base_offset,
                        "data_len": 32,
                        "token_len": 8,
                    },
                    {
                        "vkey": (f"{func_name}-v1",),
                        "called": {called_typed},
                        "data_offset": base_offset + 32,
                        "data_len": 16,
                        "token_len": 4,
                    },
                ],
            }
        )

    function_lookup = {}
    for entry in matched_data_entries:
        for vdata in entry["version_data"]:
            function_lookup[(called, vdata["vkey"])] = (4096, 64, 0)

    sections_path = tmp_path / "demo_sections.csv"
    index_path = tmp_path / "demo_matched_index.bin"
    bin_path = tmp_path / "demo_sections.bin"

    with open(sections_path, "w", newline="", encoding="ascii") as sf, \
         open(index_path, "wb") as idxf:
        write_csv_prelude(sf)
        section_writer = SectionWriter(bin_path)
        try:
            write_matched_sections_pass2(
                matched_data_entries,
                function_lookup,
                sf,
                idxf,
                io.StringIO(),
                _StubVariantRegistry(),
                registry,
                section_writer,
                ExternProviderRegistry(),
                matched_func_names={fn_a, fn_b},
                sectioned_func_names={fn_a, fn_b},
            )
            section_writer.finalize()
        except BaseException:
            section_writer.close()
            raise

    pair = read_csv_section_index_arrays(index_path)
    assert pair is not None
    bin_starts, bin_lengths = pair
    assert bin_starts.shape == (2,)

    # Encounter order is preserved (no avg-len sort anymore).
    starts = bin_starts.tolist()
    lengths = bin_lengths.tolist()
    pairs = list(zip(starts, lengths))
    bin_size = bin_path.stat().st_size
    assert pairs[0][0] == MATCHED_SECTIONS_BIN_PRELUDE_SIZE
    assert pairs[0][0] + pairs[0][1] == pairs[1][0]
    assert pairs[1][0] + pairs[1][1] == bin_size

    # Every bin_offset AND every bin_section_length is 4-byte aligned.
    for i, (start, length) in enumerate(pairs):
        assert start % 4 == 0, f"bin_starts[{i}]={start} not 4-aligned"
        assert length % 4 == 0, f"bin_lengths[{i}]={length} not 4-aligned"


def _run_matched_pass2(
    tmp_path: Path,
    matched_data_entries,
    function_lookup,
    registry,
):
    """Drive ``write_matched_sections_pass2`` and return ``(bin_size,
    bin_starts, bin_lengths)``."""
    sections_path = tmp_path / "demo_sections.csv"
    index_path = tmp_path / "demo_matched_index.bin"
    bin_path = tmp_path / "demo_sections.bin"
    matched_func_names = {entry["func_name"] for entry in matched_data_entries}
    with open(sections_path, "w", newline="", encoding="ascii") as sf, \
         open(index_path, "wb") as idxf:
        write_csv_prelude(sf)
        section_writer = SectionWriter(bin_path)
        try:
            write_matched_sections_pass2(
                matched_data_entries,
                function_lookup,
                sf,
                idxf,
                io.StringIO(),
                _StubVariantRegistry(),
                registry,
                section_writer,
                ExternProviderRegistry(),
                matched_func_names=matched_func_names,
                sectioned_func_names=matched_func_names,
            )
            section_writer.finalize()
        except BaseException:
            section_writer.close()
            raise
    pair = read_csv_section_index_arrays(index_path)
    assert pair is not None
    bin_starts, bin_lengths = pair
    return bin_path.stat().st_size, bin_starts.tolist(), bin_lengths.tolist()


def test_matched_arm_padding_exercises_odd_variant_counts(tmp_path: Path) -> None:
    """Functions with odd vs even variant counts drive the BIN section
    trailer through both pad-needed and pad-not-needed residues.

    A variant header is 10 bytes (``u32 variant_ref_offset | u32
    data_offset_shifted | u16 n_calls``); with no per-call entries the
    full variant block is 10 bytes. The section header is 8 bytes, a
    call_target entry is 12 bytes, so a 1-call_target / N-variant
    section is ``8 + 12 + 10*N = 20 + 10*N`` bytes -- 30 for N=1
    (needs +2 trailer pad), 40 for N=2 (no pad), 50 for N=3 (+2),
    60 for N=4 (no pad). This fixture mixes both parities to make
    the writer's pad logic run with rem != 0 AND rem == 0 in the same
    run.
    """
    callee = "callee"
    func_names = ["fn_one", "fn_two", "fn_three", "fn_four"]
    variant_counts = [1, 2, 3, 4]
    registry = _build_registry(*func_names, callee)

    callee_typed = (callee, CallTargetType.EXTERN)
    matched_data_entries = []
    for idx, (name, n_variants) in enumerate(zip(func_names, variant_counts)):
        version_data = [
            {
                "vkey": (f"{name}-v{v}",),
                "called": {callee_typed},
                "data_offset": idx * 256 + v * 32,
                "data_len": 16,
                "token_len": 4,
            }
            for v in range(n_variants)
        ]
        matched_data_entries.append(
            {
                "func_name": name,
                "unique_called": [callee_typed],
                "extern_libraries": {},
                "version_data": version_data,
            }
        )

    function_lookup = {
        (callee, vdata["vkey"]): (4096, 16, 0)
        for entry in matched_data_entries
        for vdata in entry["version_data"]
    }

    bin_size, starts, lengths = _run_matched_pass2(
        tmp_path, matched_data_entries, function_lookup, registry
    )

    assert len(starts) == len(func_names)
    assert len(lengths) == len(func_names)

    # Per-section alignment + adjacency.
    for i, (start, length) in enumerate(zip(starts, lengths)):
        assert start % 4 == 0, f"bin_starts[{i}]={start} not 4-aligned"
        assert length % 4 == 0, f"bin_lengths[{i}]={length} not 4-aligned"
    for i in range(len(starts) - 1):
        assert starts[i] + lengths[i] == starts[i + 1], (
            f"sections must abut: section[{i}] ends at "
            f"{starts[i] + lengths[i]}, section[{i + 1}] starts at "
            f"{starts[i + 1]}"
        )
    assert starts[-1] + lengths[-1] == bin_size

    # Lengths must vary across functions (different variant counts ->
    # different section widths) so the writer's pad logic runs against
    # multiple input residues. A constant section width would be
    # coincidentally correct -- the same trap the legacy 4-byte-by-
    # coincidence alignment had.
    observed_lengths = set(lengths)
    assert len(observed_lengths) >= 2, (
        f"matched-arm pad-residue coverage too narrow; observed "
        f"length set {observed_lengths}. Widen the variant-count spread."
    )
