"""Per-variant metadata derived from the BIN section catalog.

Single concern: translate one ``(Section, VariantBlock)`` pair (as
parsed by :mod:`tokenizer.aligned_data.matched_sections_bin`) into the
``FunctionData.metadata`` dict the dataloader hands to callers. The
field shape pre-dates the BIN cutover -- it was previously derived
from a matched-section CSV row -- and we preserve it so consumers
that read the dict by key keep working.

``parse_call_targets`` + ``parse_called_line_nos_typed`` stay as CSV
cell parsers for any out-of-band consumer (e.g. the validator's CSV
cross-check); they are not on the loader's hot path post-cutover.
"""
from typing import Any, Dict, List, Tuple

from .call_target_type import CallTargetType
from .matched_sections_bin import Section, VariantBlock

# Inverse of :data:`tokenizer.aligned_data.csv_format._CALL_TARGET_TYPE_CHAR`.
# Single source of truth for the per-call-target type tag rides on the
# encoder side; this mirror decodes the on-disk character back into the
# typed enum so callers never re-implement the mapping.
_TYPE_CHAR_TO_CALL_TARGET: "dict[str, CallTargetType]" = {
    "L": CallTargetType.LOCAL,
    "P": CallTargetType.PLT,
    "E": CallTargetType.EXTERN,
}


def parse_call_targets(call_targets_str: str) -> List[List[int]]:
    """Parse a section-CSV call-targets cell.

    Wire form: ``"idx,hex_offset,is_matched;..."`` for the matched
    arm; the unmatched arm prepends ``-comp_set`` to ``idx`` (the
    comp_set fragment is discarded — it's a writer-side grouping
    artefact, not part of the call-target identity). Returns
    ``[[idx, offset, is_matched]]`` triples. The ``hex_length`` field
    is gone post-Phase 4.1; ``_data.bin`` records are self-describing
    (see :func:`tokenizer.aligned_data.binary_format.parse_binary_header`).

    Out-of-band consumer (validator CSV cross-check). The loader hot
    path consumes the BIN-derived equivalent via
    :func:`extract_metadata_from_variant_block`.
    """
    if not call_targets_str:
        return []
    result = []
    for entry in call_targets_str.split(";"):
        if entry:
            parts = entry.split(",")
            if len(parts) == 3:
                idx_part = parts[0].split("-")[0]
                result.append([int(idx_part), int(parts[1], 16), int(parts[2])])
    return result


def parse_called_line_nos_typed(
    cell: str,
) -> List[Tuple[int, CallTargetType]]:
    """Inverse of
    :func:`tokenizer.aligned_data.csv_format.format_called_line_nos_typed`.

    Decodes the comma-joined ``<base64_line_no>:<type_char>`` form back
    into ``(line_no, CallTargetType)`` tuples. Empty cell -> ``[]``.
    Unknown type characters raise :class:`ValueError` rather than
    silently coalescing — the type tag is a correctness signal (the
    Phase-3 fix to the v2 metadata category coalesce bug), so a
    malformed cell is a builder bug worth surfacing.
    """
    if not cell:
        return []
    # Local import keeps the module-level import graph one-directional
    # (csv_format consumes call_target_type; metadata.py decodes back
    # through line_no_codec without re-importing csv_format).
    from .line_no_codec import decode_line_no

    out: List[Tuple[int, CallTargetType]] = []
    for entry in cell.split(","):
        b64, _, type_char = entry.rpartition(":")
        if not b64 or not type_char:
            raise ValueError(
                f"call-targets cell entry missing type tag: {entry!r}"
            )
        call_type = _TYPE_CHAR_TO_CALL_TARGET.get(type_char)
        if call_type is None:
            raise ValueError(
                f"call-targets cell entry has unknown type char "
                f"{type_char!r} in {entry!r}"
            )
        out.append((decode_line_no(b64), call_type))
    return out


def extract_metadata_from_variant_block(
    section: Section, variant: VariantBlock
) -> Dict[str, Any]:
    """Build the per-variant metadata dict from a parsed BIN variant.

    * ``variant_ref`` -- hex string (no ``0x`` prefix) of the variant
      block's ``variant_ref_offset`` u32. Matches the legacy CSV cell
      so :class:`BinarySession.get_variant_by_ref` round-trips through
      ``int(ref, 16)`` unchanged.
    * ``call_targets`` -- ``[[called_idx, function_section_ptr,
      section_variant_index, is_matched_int]]`` derived from the
      variant's ``per_call_entries``. Per-call entry's ``called_idx``
      indexes into ``section.call_targets``; the call_target's
      ``function_section_ptr`` and ``is_matched`` flag are surfaced
      alongside the ``section_variant_index`` so consumers retain a
      structurally-equivalent 4-tuple view.
    * ``data_offset`` -- real byte offset into ``_data.bin`` (matched)
      or ``_unmatched_data.bin`` (unmatched), recovered from the
      variant's ``data_offset_shifted << 4``.
    """
    call_targets: List[List[int]] = []
    for called_idx, section_variant_index in variant.per_call_entries:
        call_target = section.call_targets[called_idx]
        call_targets.append(
            [
                called_idx,
                call_target.function_section_ptr,
                section_variant_index,
                1 if call_target.is_matched else 0,
            ]
        )
    return {
        "variant_ref": f"{variant.variant_ref_offset:x}",
        "call_targets": call_targets,
        "data_offset": variant.data_offset_shifted << 4,
    }
