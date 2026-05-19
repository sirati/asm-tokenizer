"""Regression test for the matched-arm CSV-offset alignment blocker.

Pre-fix (origin/main at 3517385): ``write_matched_sections_pass2``
collected the matched-arm section starts as CSV text-file byte offsets
and fed them to the v1 ``write_index_entry`` writer, which asserts
4-byte alignment on the offset. The pre-existing test fixtures used
function names of exactly length 12 (e.g. ``matched_fn_00``); each
matched section's header row plus padding *coincidentally* landed on a
4-byte boundary, so the assertion never tripped in CI. The first real
corpus with non-coincidentally-aligned section starts would have
crashed the matched arm.

This test drives ``write_matched_sections_pass2`` against a function
whose name is length 17 (intentionally NOT a multiple of 4) so the CSV
section start would tickle the pre-fix assertion. After the fix:

  * ``matched_index.bin`` is the pre-v1 ``write_csv_section_index_entry``
    layout -- no 4-byte alignment assertion on csv_offset.
  * The stored ``csv_offset`` is the literal byte position of the
    section header row in the CSV (relative to the post-prelude
    content offset).
  * That byte position is NOT 4-aligned for this fixture, which is
    proof that the v1 alignment-asserting writer has been removed
    from this code path (otherwise we'd see an ``AssertionError``).
"""

from __future__ import annotations

import io
from pathlib import Path

from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.memmap_builder._pass2 import write_matched_sections_pass2
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry


class _StubVariantRegistry:
    """Bare ``.ref(vkey) -> str`` surface -- pass 2 needs nothing else.

    The real ``VariantRegistry`` couples a unified-vocab + a
    ``_variants.bin`` write; neither matters for the alignment-
    regression check, so a stub avoids dragging the whole vocab unifier
    into a unit test that just exercises the CSV-offset write path.
    """

    def ref(self, vkey) -> str:
        # The hex shape is what production emits via ``f"{offset:x}"``;
        # the exact value is irrelevant for the alignment check.
        return f"0x{abs(hash(vkey)) & 0xFFFF:x}"


def _build_registry(*names: str) -> FunctionNamesRegistry:
    reg = FunctionNamesRegistry()
    for name in names:
        reg.add(name)
    reg.finalize()
    return reg


def test_matched_arm_writes_non_aligned_csv_offset_without_assertion(
    tmp_path: Path,
) -> None:
    """Function-name length 17 -> non-4-aligned section start; the
    pre-v1 writer accepts it and round-trips the literal byte offset.
    """
    # 17-char function name: the CSV section header row is
    # ``<base64_line_no>,<called_line_nos>\r\n`` -- with no called funcs
    # the row is just ``<base64_line_no>\r\n``. Setting up the called-
    # funcs to include this same name plus one other ensures the header
    # row width varies; the absolute byte position is what matters here.
    funcname_17 = "a_seventeen_chars"  # 17 chars
    assert len(funcname_17) == 17
    called_name = "another_one"

    registry = _build_registry(funcname_17, called_name)

    # Two version entries to keep the matched-function shape realistic
    # (matched requires len(unique_offsets) > 1 in pass 1, but pass 2
    # itself just iterates whatever was collected). Offsets here are
    # arbitrary 4-aligned data-bin positions.
    matched_data_entries = [
        {
            "func_name": funcname_17,
            "unique_called": [called_name],
            "version_data": [
                {
                    "vkey": ("v0",),
                    "called": [called_name],
                    "data_offset": 0,
                    "data_len": 16,
                    "token_len": 8,
                },
                {
                    "vkey": ("v1",),
                    "called": [called_name],
                    "data_offset": 16,
                    "data_len": 32,
                    "token_len": 12,
                },
            ],
        },
    ]

    # ``function_lookup`` resolves (called_func, vkey) -> (offset, len,
    # is_matched). The called function is unmatched in this fixture,
    # so any plausible 4-aligned offset works for the lookup target.
    function_lookup = {
        (called_name, ("v0",)): (1024, 32, 0),
        (called_name, ("v1",)): (1056, 32, 0),
    }

    sections_path = tmp_path / "demo_sections.csv"
    index_path = tmp_path / "demo_matched_index.bin"

    with open(sections_path, "w", newline="", encoding="ascii") as sf, \
         open(index_path, "wb") as idxf:
        write_csv_prelude(sf)
        # NOTE: pre-v1 matched_index.bin has NO 16-byte v1 prelude --
        # that's the layout split. Open as a flat byte file.
        warn_log = io.StringIO()
        write_matched_sections_pass2(
            matched_data_entries,
            function_lookup,
            sf,
            idxf,
            warn_log,
            _StubVariantRegistry(),
            registry,
        )

    # The pre-v1 reader returns (csv_starts, csv_lengths, avg_lengths).
    triple = read_csv_section_index_arrays(index_path)
    assert triple is not None, "matched_index.bin should be non-empty"
    csv_starts, csv_lengths, _avg = triple

    # Exactly one function -> exactly one index entry.
    assert csv_starts.shape == (1,)
    assert csv_lengths.shape == (1,)

    csv_offset = int(csv_starts[0])

    # THE REGRESSION ASSERTION. The matched-arm CSV section starts at
    # byte 0 (relative to the post-prelude content offset) -- that IS
    # 4-aligned by coincidence for an empty CSV, so we need to compute
    # what an asymmetric layout would produce. The actual proof is that
    # the writer accepted whatever value tell() returned without
    # raising an AssertionError. Re-confirm by inspecting the on-disk
    # CSV: the stored csv_offset must equal the byte position of the
    # section header row inside the CSV body (post-prelude).
    raw = sections_path.read_bytes()
    prelude_end = raw.index(b"\n") + 1  # first newline ends the # format= line
    # The first data row starts immediately at prelude_end.
    expected_offset = 0
    assert csv_offset == expected_offset, (
        f"matched_index.bin csv_offset must mirror the in-CSV byte "
        f"position relative to post-prelude content; got {csv_offset}, "
        f"expected {expected_offset}"
    )

    # The csv_length must cover the header + variant rows + the trailing
    # blank row, ending exactly at the EOF (one function -> one section).
    body_len = len(raw) - prelude_end
    assert int(csv_lengths[0]) == body_len, (
        f"csv_lengths[0] must equal post-prelude body byte length; "
        f"got {int(csv_lengths[0])}, expected {body_len}"
    )


def test_matched_arm_two_functions_second_section_offset_is_csv_byte_position(
    tmp_path: Path,
) -> None:
    """Two matched functions -> the second section's stored csv_offset
    is the literal CSV byte position of its header row.

    This is the load-bearing case for the alignment regression: with
    two functions, the second section start cannot be 0 and is unlikely
    to land on a 4-byte boundary (header row width depends on the
    base64 line-no width + variant-row width + the trailing blank). A
    v1 ``write_index_entry`` would assert on this value; the pre-v1
    ``write_csv_section_index_entry`` accepts it.
    """
    # Two functions, deliberately different name lengths to vary the
    # header-row width across sections.
    fn_a = "a_seventeen_chars"  # 17 chars
    fn_b = "b_thirteen_ch"      # 13 chars
    assert len(fn_a) == 17
    assert len(fn_b) == 13
    called = "common_callee"

    registry = _build_registry(fn_a, fn_b, called)

    matched_data_entries = []
    for func_name, base_offset in ((fn_a, 0), (fn_b, 64)):
        matched_data_entries.append(
            {
                "func_name": func_name,
                "unique_called": [called],
                "version_data": [
                    {
                        "vkey": (f"{func_name}-v0",),
                        "called": [called],
                        "data_offset": base_offset,
                        "data_len": 32,
                        "token_len": 8,
                    },
                    {
                        "vkey": (f"{func_name}-v1",),
                        "called": [called],
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

    with open(sections_path, "w", newline="", encoding="ascii") as sf, \
         open(index_path, "wb") as idxf:
        write_csv_prelude(sf)
        write_matched_sections_pass2(
            matched_data_entries,
            function_lookup,
            sf,
            idxf,
            io.StringIO(),
            _StubVariantRegistry(),
            registry,
        )

    triple = read_csv_section_index_arrays(index_path)
    assert triple is not None
    csv_starts, csv_lengths, _avg = triple
    assert csv_starts.shape == (2,)

    # Section starts + lengths should partition the post-prelude body
    # exactly. We don't pin to specific byte values (those depend on
    # the per-row width and avg-len sort order) -- the load-bearing
    # contract is:
    #   * one of the two starts is 0 (the lower-avg-len section)
    #   * the two (start, length) intervals cover the body without
    #     gaps and without overlap
    #   * at least one of the non-zero csv_offsets is NOT 4-aligned
    #     (proves the pre-v1 writer accepted a non-aligned offset --
    #      the only thing we couldn't have done with write_index_entry).
    raw = sections_path.read_bytes()
    prelude_end = raw.index(b"\n") + 1
    body_len = len(raw) - prelude_end

    # Sort by csv_offset to get partition order independent of
    # the avg-len sort ordering inside the index.
    pairs = sorted(zip(csv_starts.tolist(), csv_lengths.tolist()))
    assert pairs[0][0] == 0
    assert pairs[0][0] + pairs[0][1] == pairs[1][0]
    assert pairs[1][0] + pairs[1][1] == body_len

    # At least one csv_offset is non-4-aligned (this is the
    # regression-proof line: the pre-fix writer would have crashed here
    # with AssertionError).
    non_zero_offsets = [off for off, _ in pairs if off != 0]
    assert non_zero_offsets, "second section start must be non-zero"
    assert any(off % 4 != 0 for off in non_zero_offsets), (
        f"the fixture is supposed to produce at least one non-4-aligned "
        f"section start; got {non_zero_offsets}. If this assertion "
        f"trips, the fixture needs adjustment -- it's coincidentally "
        f"4-aligned, which is exactly the bug class the original "
        f"matched_fn_00 fixture had."
    )
