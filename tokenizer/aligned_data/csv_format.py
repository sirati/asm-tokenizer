from typing import List, Sequence, Tuple

from .call_target_type import CallTargetType
from .line_no_codec import (
    decode_line_no,
    decode_line_nos_csv,
    encode_line_no,
    encode_line_nos_csv,
)
from .memmap_format import MEMMAP_FORMAT_VERSION


def write_csv_prelude(handle) -> None:
    """Emit the comment-line CSV prelude shared by sections + slim variants CSVs.

    Layout is exactly ``# format=<MEMMAP_FORMAT_VERSION>\\n`` written as the
    file's first line, before any other content (including the
    ``csv.writer`` header row). The leading ``#`` makes third-party CSV
    viewers ignore the line while our readers parse the integer back out
    via the same constant. Centralising the format string here keeps the
    on-wire prelude one definition.
    """
    handle.write(f"# format={MEMMAP_FORMAT_VERSION}\n")


# Per-call-target type tag used in the section-CSV header cell. The
# letter is the single source of truth for the type discriminator that
# rides alongside each base64 line number in the comma-joined
# ``call_targets`` cell; the BIN-side ``CallTargetType`` enum stays the
# canonical typed form on every other layer.
_CALL_TARGET_TYPE_CHAR: "dict[CallTargetType, str]" = {
    CallTargetType.LOCAL: "L",
    CallTargetType.PLT: "P",
    CallTargetType.EXTERN: "E",
}


def format_call_targets_dict(call_targets_list: List) -> str:
    """Format the matched-section variant row's call-targets cell.

    Wire form: semicolon-joined ``idx,hex_offset,is_matched`` triples
    (one per per-variant call-target reference). ``hex_length`` was
    dropped post-Phase 4.1 — records in ``_data.bin`` are
    self-describing via :func:`parse_binary_header` +
    :func:`record_total_size`, so the CSV cell no longer mirrors the
    callee record length. The CSV is debug-only since Phase 4.2; the
    loader consumes ``<binary>_sections.bin`` directly.
    """
    if not call_targets_list:
        return ""
    parts = []
    for idx, start, is_matched in call_targets_list:
        hex_start = f"{start:x}"
        parts.append(f"{idx},{hex_start},{is_matched}")
    return ";".join(parts)


def format_variant_refs(variant_refs: List[str]) -> str:
    """Encode an ordered list of ``0x<hex>`` variant refs into a single
    section-CSV cell.

    Each entry is already the ``0x<hex>`` form produced by
    ``VariantRegistry.ref`` — this function only chooses the
    separator (``;``). Empty list yields an empty cell.
    """
    if not variant_refs:
        return ""
    return ";".join(variant_refs)


def format_unique_called(unique_called: List[str]) -> str:
    """Format list of function names, comma-separated with escaped commas"""
    escaped = [name.replace(",", "\\,") for name in unique_called]
    return ",".join(escaped)


def parse_escaped_function_names(called_str: str) -> List[str]:
    """Parse comma-separated function names, handling escaped commas."""
    if not called_str:
        return []
    parts = []
    current = []
    i = 0
    while i < len(called_str):
        if called_str[i] == "\\" and i + 1 < len(called_str) and called_str[i + 1] == ",":
            current.append(",")
            i += 2
        elif called_str[i] == ",":
            parts.append("".join(current))
            current = []
            i += 1
        else:
            current.append(called_str[i])
            i += 1
    if current:
        parts.append("".join(current))
    return parts


def format_function_line_no(line_no: int) -> str:
    """Encode a function's sidecar line no for a section-CSV name cell.

    Thin wrapper over :func:`tokenizer.aligned_data.line_no_codec.encode_line_no`
    that exists so builder + loader call sites read as the domain
    operation ("format the line no for this function name") instead
    of the lower-level codec call. Single source of truth for the
    base64 byte representation still lives in ``line_no_codec``.
    """
    return encode_line_no(line_no)


def parse_function_line_no(s: str) -> int:
    """Inverse of :func:`format_function_line_no`."""
    return decode_line_no(s)


def format_function_line_nos_csv(line_nos: Sequence[int]) -> str:
    """Comma-joined :func:`format_function_line_no` for a sequence
    of sidecar line numbers.

    Name-only encoder kept for any out-of-band consumer that still
    handles the legacy untyped form (no per-call-site type tag).
    The matched-/unmatched-section header cell now carries the typed
    form produced by :func:`format_called_line_nos_typed` — every new
    on-disk caller routes through that helper.
    """
    return encode_line_nos_csv(line_nos)


def parse_function_line_nos_csv(s: str) -> List[int]:
    """Inverse of :func:`format_function_line_nos_csv`."""
    return decode_line_nos_csv(s)


def format_called_line_no_with_type(line_no: int, call_type: CallTargetType) -> str:
    """Encode one ``<base64_line_no>:<type_char>`` cell entry.

    ``type_char`` is ``L``/``P``/``E`` for
    :data:`CallTargetType.LOCAL`/:data:`CallTargetType.PLT`/:data:`CallTargetType.EXTERN`.
    The type tag preserves the call-site classification the BIN already
    captures (see ``matched_sections_bin.py``); the CSV cell exposes it
    so a human reading the debug file sees the same ``(name, type)``
    pairing the loader sees off the BIN.

    Two call sites in the same caller variant referencing the same
    callee name under different types yield TWO distinct entries — the
    type tag is what keeps them from being silently coalesced.
    """
    type_char = _CALL_TARGET_TYPE_CHAR[call_type]
    return f"{encode_line_no(line_no)}:{type_char}"


def format_called_line_nos_typed(
    typed: Sequence[Tuple[int, CallTargetType]],
) -> str:
    """Comma-joined :func:`format_called_line_no_with_type` for a
    sequence of ``(line_no, call_type)`` pairs.

    Used for the called-funcs cell in BOTH section CSVs (matched
    header's second cell + unmatched header's third cell). Empty
    sequence -> empty string. Order is preserved from the input — the
    callers in :mod:`tokenizer.memmap_builder._pass2` sort their typed
    callee lists once at pass-2 entry so the on-disk order is
    deterministic.
    """
    return ",".join(
        format_called_line_no_with_type(line_no, call_type)
        for line_no, call_type in typed
    )


def parse_variant_refs(variant_refs_str: str) -> List[str]:
    """Inverse of ``format_variant_refs``: split a section-CSV variant-ref
    cell back into the ordered list of ``0x<hex>`` refs.

    Returns the raw string entries (no integer conversion) so consumers
    can decide whether to keep the hex form or resolve into the
    sidecar variants CSV. Empty cell yields ``[]``.
    """
    if not variant_refs_str:
        return []
    return [part for part in variant_refs_str.split(";") if part]
