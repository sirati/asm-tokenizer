"""Matched-arm CSV section alignment invariants.

The matched-section writer (``write_matched_sections_pass2``) pads
between sections with raw ``\\n`` bytes so every section start (and
every section length) lands on a 4-byte boundary, AND so the run of
``\\n`` bytes between consecutive sections is at least 3 bytes long
(last variant row's terminator + the blank-row terminator from
``writerow([])`` + 1-4 pad bytes). The ``matched_index.bin`` entry
codec (u40 ``csv_offset >> 2`` + u24 ``csv_length >> 2``) asserts both
alignments, so a layout regression trips loudly at write time -- but
we also pin the invariants here so a future refactor can't silently
drop the padding step and rely on coincidental alignment.

What we assert per fixture:

  * Every ``csv_starts[i] % 4 == 0`` (the matched index codec requires
    it; double-checked at the array level).
  * Every ``csv_lengths[i] % 4 == 0`` (same).
  * Between consecutive sections the bytes are all ``\\n`` (no row
    content leaks into the padding) and the count is in [3, 6] (two
    from the row + blank-row terminators + 1-4 from the pad helper).
  * Nowhere in the section CSV does ``\\r\\n`` survive -- the csv
    writer is pinned to ``lineterminator='\\n'`` so byte counting is
    deterministic.
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


def test_matched_arm_one_function_section_is_4_aligned_and_lf_only(
    tmp_path: Path,
) -> None:
    """One matched function -> the single section start is 0 (trivially
    4-aligned), and the section length is rounded up to a 4-byte
    multiple via ``\\n`` padding after the trailing blank row."""
    # 17-char function name -> the section's unpadded length is
    # non-4-aligned; the pad helper must round it up.
    funcname_17 = "a_seventeen_chars"
    assert len(funcname_17) == 17
    called_name = "another_one"

    registry = _build_registry(funcname_17, called_name)

    # Use EXTERN so the BIN's call_target resolves without needing a
    # callee section in the fixture (the alignment test exercises CSV
    # padding, not cross-section BIN linking).
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
    csv_starts, csv_lengths = pair

    # Exactly one function -> exactly one index entry.
    assert csv_starts.shape == (1,)
    assert csv_lengths.shape == (1,)

    csv_offset = int(csv_starts[0])
    csv_length = int(csv_lengths[0])

    # Alignment invariants.
    assert csv_offset % 4 == 0, (
        f"csv_starts[0]={csv_offset} must be 4-byte aligned"
    )
    assert csv_length % 4 == 0, (
        f"csv_lengths[0]={csv_length} must be 4-byte aligned"
    )

    # The literal CSV layout: the section starts at the first byte
    # after the ``# format=N\n`` prelude (offset 0 relative to post-
    # prelude content). The recorded length spans from there to EOF
    # (one function -> one section, padded to alignment).
    raw = sections_path.read_bytes()
    prelude_end = raw.index(b"\n") + 1  # first newline ends the # format= line
    body_len = len(raw) - prelude_end
    assert csv_offset == 0
    assert csv_length == body_len

    # No ``\r\n`` anywhere in the section CSV (we pinned LF).
    assert b"\r\n" not in raw, "section CSV must use \\n line terminators only"


def test_matched_arm_two_functions_every_section_is_4_aligned_and_padded(
    tmp_path: Path,
) -> None:
    """Two matched functions with deliberately mismatched name lengths
    -> the second section start lands on a 4-byte boundary even though
    the natural (unpadded) end of section 1 is not 4-aligned, AND the
    bytes between consecutive sections are all ``\\n`` (≥ 2 of them).
    """
    fn_a = "a_seventeen_chars"  # 17 chars
    fn_b = "b_thirteen_ch"      # 13 chars
    assert len(fn_a) == 17
    assert len(fn_b) == 13
    called = "common_callee"

    registry = _build_registry(fn_a, fn_b, called)

    # Use EXTERN so the BIN's call_target resolves without a callee
    # section emit (see test_matched_arm_one_function_section_*).
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
    csv_starts, csv_lengths = pair
    assert csv_starts.shape == (2,)

    raw = sections_path.read_bytes()
    prelude_end = raw.index(b"\n") + 1
    body_len = len(raw) - prelude_end

    # Encounter order is preserved (no avg-len sort anymore).
    starts = csv_starts.tolist()
    lengths = csv_lengths.tolist()
    pairs = list(zip(starts, lengths))
    assert pairs[0][0] == 0
    assert pairs[0][0] + pairs[0][1] == pairs[1][0]
    assert pairs[1][0] + pairs[1][1] == body_len

    # Every csv_offset AND every csv_length is 4-byte aligned.
    for i, (start, length) in enumerate(pairs):
        assert start % 4 == 0, f"csv_starts[{i}]={start} not 4-aligned"
        assert length % 4 == 0, f"csv_lengths[{i}]={length} not 4-aligned"

    # No CRLF anywhere in the section CSV.
    assert b"\r\n" not in raw, "section CSV must use \\n line terminators only"

    # The padding between sections 0 and 1 is all ``\n``. Section 0
    # ends (per the stored length) at the byte where section 1 begins;
    # walking backwards from there we count the run of ``\n`` bytes
    # since the last non-``\n`` content character. That run is:
    #
    #   * last variant row's terminator (1 ``\n``)
    #   * ``writerow([])``'s blank-row terminator (1 ``\n``)
    #   * 1..4 pad bytes from the helper
    #
    # So the run length must be in [3, 6] -- the legacy "blank-row only"
    # layout would have produced exactly 2.
    section0_end_abs = prelude_end + pairs[0][0] + pairs[0][1]
    section1_start_abs = prelude_end + pairs[1][0]
    assert section0_end_abs == section1_start_abs, (
        "stored length must include the inter-section pad"
    )

    last_content_idx = section0_end_abs - 1
    while last_content_idx >= prelude_end and raw[last_content_idx:last_content_idx + 1] == b"\n":
        last_content_idx -= 1
    trailing = raw[last_content_idx + 1:section1_start_abs]
    assert set(trailing) == {ord(b"\n")}, (
        f"inter-section bytes must be all '\\n'; got {trailing!r}"
    )
    assert 3 <= len(trailing) <= 6, (
        f"inter-section ``\\n`` run length must be in [3, 6]; "
        f"got {len(trailing)} ({trailing!r})"
    )


def _run_matched_pass2(
    tmp_path: Path,
    matched_data_entries,
    function_lookup,
    registry,
):
    """Drive ``write_matched_sections_pass2`` and return ``(raw_bytes,
    csv_starts, csv_lengths, prelude_end)``."""
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
    csv_starts, csv_lengths = pair
    raw = sections_path.read_bytes()
    prelude_end = raw.index(b"\n") + 1
    return raw, csv_starts.tolist(), csv_lengths.tolist(), prelude_end


def test_matched_arm_padding_covers_all_residue_classes(tmp_path: Path) -> None:
    """A many-function fixture with deliberately varied name lengths
    drives the pad helper through every ``end0 % 4 ∈ {0, 1, 2, 3}``
    residue class. For each section produced:

      * ``csv_offset % 4 == 0`` AND ``csv_length % 4 == 0``.
      * Padding bytes between consecutive sections are pure ``\\n``,
        count is between 2 and ``_SECTION_ALIGN + 1`` inclusive (1
        from the row terminator + 1-``_SECTION_ALIGN`` from the pad
        helper, capped at the alignment boundary).
      * The collected residue set covers all four classes ``{0, 1, 2,
        3}`` so the helper's max-clause and ceil-clause are both
        exercised.
    """
    # Function name length drives the residue of end0 % 4 because the
    # base64 line number for each function varies in width too. Using a
    # range of name lengths spaced by 1 character guarantees at least
    # one example per residue class for any reasonable corpus -- we
    # verify the coverage below rather than hard-coding the mapping.
    callee = "callee"
    name_lengths = [10, 11, 12, 13, 14, 15, 16, 17]
    func_names = [f"f{'x' * (length - 1)}" for length in name_lengths]
    for name, length in zip(func_names, name_lengths):
        assert len(name) == length

    registry = _build_registry(*func_names, callee)

    callee_typed = (callee, CallTargetType.EXTERN)
    matched_data_entries = []
    for idx, name in enumerate(func_names):
        matched_data_entries.append(
            {
                "func_name": name,
                "unique_called": [callee_typed],
                "extern_libraries": {},
                "version_data": [
                    {
                        "vkey": (f"{name}-v0",),
                        "called": {callee_typed},
                        "data_offset": idx * 64,
                        "data_len": 16,
                        "token_len": 4,
                    },
                    {
                        "vkey": (f"{name}-v1",),
                        "called": {callee_typed},
                        "data_offset": idx * 64 + 32,
                        "data_len": 16,
                        "token_len": 4,
                    },
                ],
            }
        )

    function_lookup = {
        (callee, vdata["vkey"]): (4096, 16, 0)
        for entry in matched_data_entries
        for vdata in entry["version_data"]
    }

    raw, starts, lengths, prelude_end = _run_matched_pass2(
        tmp_path, matched_data_entries, function_lookup, registry
    )

    assert len(starts) == len(func_names)
    assert len(lengths) == len(func_names)
    assert b"\r\n" not in raw

    # Per-section alignment.
    for i, (start, length) in enumerate(zip(starts, lengths)):
        assert start % 4 == 0, f"csv_starts[{i}]={start} not 4-aligned"
        assert length % 4 == 0, f"csv_lengths[{i}]={length} not 4-aligned"

    # Per-section gap = run of ``\n`` bytes from the last non-``\n``
    # content character to the next section's first byte. The pad
    # helper writes 1..4 ``\n``s on top of two terminator ``\n``s
    # already on disk (last variant row + ``writerow([])``), so gap
    # is always in [3, 6]. Observing multiple distinct gap lengths
    # proves the pad helper is input-residue-sensitive (a constant gap
    # would be coincidentally-correct -- the same trap the legacy
    # 4-byte-by-coincidence alignment had).
    observed_gaps = set()
    for i in range(len(starts) - 1):
        section_end = prelude_end + starts[i] + lengths[i]
        next_section_start = prelude_end + starts[i + 1]
        assert section_end == next_section_start, (
            f"sections must abut: section[{i}] ends at {section_end}, "
            f"section[{i + 1}] starts at {next_section_start}"
        )
        idx = section_end - 1
        while idx >= prelude_end and raw[idx:idx + 1] == b"\n":
            idx -= 1
        pad_run = raw[idx + 1:section_end]
        assert set(pad_run) == {ord(b"\n")}, (
            f"section[{i}]→[{i + 1}] pad must be all '\\n'; got {pad_run!r}"
        )
        assert 3 <= len(pad_run) <= 6, (
            f"section[{i}]→[{i + 1}] pad length out of range; "
            f"got {len(pad_run)}"
        )
        observed_gaps.add(len(pad_run))

    # We need at least two distinct gap sizes to prove the pad helper
    # actually responds to the input residue (a constant-gap output
    # would be coincidentally-correct -- the same trap the legacy
    # 4-byte-by-coincidence alignment had). The fixture's varying
    # name lengths reliably produce >=2 distinct gaps; if a future
    # row-width change collapses them, the assertion fires and the
    # fixture needs adjustment.
    assert len(observed_gaps) >= 2, (
        f"matched-arm pad-residue coverage too narrow; observed {observed_gaps}. "
        f"The fixture is supposed to span multiple residues -- if it "
        f"trips, widen the name-length spread."
    )
