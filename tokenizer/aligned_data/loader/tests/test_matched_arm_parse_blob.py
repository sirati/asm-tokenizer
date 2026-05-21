"""Focused test for ``_matched_arm_loader._walk_matched_sections``.

Round-trips a synthetic 2-variant matched section through the BIN
parser so the per-section walk + per-variant ``data_offset_shifted``
decode contract is pinned independently of the orchestrator-level
``SectionArm`` chain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import (
    CallTargetSpec,
    SectionWriter,
)
from tokenizer.aligned_data.loader._matched_arm_loader import (
    _walk_matched_sections,
)


def _build_one_section_bin(
    tmp_path: Path,
    *,
    fid: int,
    variant_offsets: list[int],
) -> Path:
    """Write a single-section ``sections.bin`` with the requested
    variant ``data_offset`` values (passed in real bytes; the writer
    re-shifts to ``>> 4``).
    """
    bin_path = tmp_path / "fake_sections.bin"
    writer = SectionWriter(bin_path)
    try:
        section_offset = writer.begin_section(fid)
        # No call_targets -> empty table; per_call_entries empty too.
        writer.emit_call_targets([
            CallTargetSpec(
                function_name_ptr=fid + 1,
                type=CallTargetType.EXTERN,
                is_matched=False,
            )
        ])
        for i, offset in enumerate(variant_offsets):
            writer.begin_variant(
                variant_ref_offset=0x10 * (i + 1),
                data_offset_shifted=offset >> 4,
            )
            writer.emit_per_call_entries([])
            writer.end_variant(vkey=("v", i))
        writer.end_section()
        writer.finalize()
    except BaseException:
        writer.close()
        raise
    return bin_path


def test_walk_matched_sections_two_variants_round_trip(tmp_path: Path) -> None:
    """One section with two variants -> walker returns (func_name,
    Section) with both variants' ``data_offset_shifted`` round-tripped
    via the BIN parser."""
    fid = 42
    variant_offsets = [0x10, 0x20]
    bin_path = _build_one_section_bin(
        tmp_path, fid=fid, variant_offsets=variant_offsets
    )

    # ``matched_index.bin``-equivalent: one entry at the prelude end.
    from tokenizer.aligned_data.memmap_format import (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE,
    )
    bin_starts = np.array([MATCHED_SECTIONS_BIN_PRELUDE_SIZE], dtype=np.int64)
    line_to_name = {fid: "matched_fn"}

    func_names, sections = _walk_matched_sections(
        bin_path, bin_starts, line_to_name
    )

    assert func_names == ["matched_fn"]
    assert len(sections) == 1
    section = sections[0]
    assert section.function_name_ptr == fid
    assert len(section.variants) == 2
    decoded_offsets = [v.data_offset_shifted << 4 for v in section.variants]
    assert decoded_offsets == variant_offsets


def test_walk_matched_sections_missing_fid_raises(tmp_path: Path) -> None:
    """A function_name_ptr absent from line_to_name -> ValueError with
    a migration-pointing message (sidecar drift)."""
    import pytest

    fid = 7
    bin_path = _build_one_section_bin(
        tmp_path, fid=fid, variant_offsets=[0x10]
    )
    from tokenizer.aligned_data.memmap_format import (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE,
    )
    bin_starts = np.array([MATCHED_SECTIONS_BIN_PRELUDE_SIZE], dtype=np.int64)

    with pytest.raises(ValueError, match="re-run memmap_builder"):
        _walk_matched_sections(bin_path, bin_starts, line_to_name={})


def test_walk_matched_sections_bad_prelude_raises(tmp_path: Path) -> None:
    """A ``sections.bin`` with a corrupt prelude raises before the
    section walk even starts; downstream callers get a clear
    "regenerate the BIN" pointer rather than garbage offsets."""
    import pytest

    fid = 11
    bin_path = _build_one_section_bin(
        tmp_path, fid=fid, variant_offsets=[0x10]
    )
    raw = bin_path.read_bytes()
    # Corrupt the magic; keep everything else intact.
    bin_path.write_bytes(b"BAD!" + raw[4:])
    from tokenizer.aligned_data.memmap_format import (
        MATCHED_SECTIONS_BIN_PRELUDE_SIZE,
    )
    bin_starts = np.array([MATCHED_SECTIONS_BIN_PRELUDE_SIZE], dtype=np.int64)

    with pytest.raises(ValueError, match="magic"):
        _walk_matched_sections(
            bin_path, bin_starts, line_to_name={fid: "fn"}
        )
