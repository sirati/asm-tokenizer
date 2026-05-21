"""Per-variant metadata derived from the BIN section catalog.

Single concern: translate one ``(Section, VariantBlock)`` pair (as
parsed by :mod:`tokenizer.aligned_data.matched_sections_bin`) into the
``FunctionData.metadata`` dict the dataloader hands to callers. The
field shape pre-dates the BIN cutover -- it was previously derived
from a matched-section CSV row -- and we preserve it so consumers
that read the dict by key keep working.

``parse_inlining_data`` stays as a CSV cell parser for any
out-of-band consumer (e.g. the validator's CSV cross-check); it is
not on the loader's hot path post-cutover.
"""
from typing import Any, Dict, List

from .matched_sections_bin import Section, VariantBlock


def parse_inlining_data(inlining_str: str) -> List[List[int]]:
    """Parse a legacy inlining-data CSV cell.

    Format: ``"idx,hex_offset,hex_length,is_matched;..."`` for the
    matched arm; the unmatched arm prepends ``-comp_set`` to ``idx``
    (comp_set is discarded). Returns ``[[idx, offset, length, is_matched]]``.

    Out-of-band consumer (validator CSV cross-check). The loader hot
    path consumes the BIN-derived equivalent via
    :func:`extract_metadata_from_variant_block`.
    """
    if not inlining_str:
        return []
    result = []
    for entry in inlining_str.split(";"):
        if entry:
            parts = entry.split(",")
            if len(parts) == 4:
                idx_part = parts[0].split("-")[0]
                result.append([int(idx_part), int(parts[1], 16), int(parts[2], 16), int(parts[3])])
    return result


def extract_metadata_from_variant_block(
    section: Section, variant: VariantBlock
) -> Dict[str, Any]:
    """Build the per-variant metadata dict from a parsed BIN variant.

    Output shape (preserved across the Phase 4 CSV→BIN cutover so
    consumers reading the dict by key keep working):

    * ``variant_ref`` -- hex string (no ``0x`` prefix) of the variant
      block's ``variant_ref_offset`` u32. Matches the legacy CSV cell
      so :class:`BinarySession.get_variant_by_ref` round-trips through
      ``int(ref, 16)`` unchanged.
    * ``inlining_data`` -- ``[[called_idx, function_section_ptr,
      section_variant_index, is_matched_int]]`` derived from the
      variant's ``per_call_entries``. Per-call entry's ``called_idx``
      indexes into ``section.call_targets``; the call_target's
      ``function_section_ptr`` and ``is_matched`` flag are surfaced
      alongside the ``section_variant_index`` so consumers retain a
      structurally-equivalent 4-tuple view. Phase 4.1 will rename this
      key + slim the cell shape to match the new semantics.
    * ``data_offset`` -- real byte offset into ``_data.bin`` (matched)
      or ``_unmatched_data.bin`` (unmatched), recovered from the
      variant's ``data_offset_shifted << 4``.
    """
    inlining_data: List[List[int]] = []
    for called_idx, section_variant_index in variant.per_call_entries:
        call_target = section.call_targets[called_idx]
        inlining_data.append(
            [
                called_idx,
                call_target.function_section_ptr,
                section_variant_index,
                1 if call_target.is_matched else 0,
            ]
        )
    return {
        "variant_ref": f"{variant.variant_ref_offset:x}",
        "inlining_data": inlining_data,
        "data_offset": variant.data_offset_shifted << 4,
    }
