from typing import List, Sequence

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


def format_inlining_dict(inlining_list: List) -> str:
    """Format inlining data as semicolon-separated: idx,hex_offset,hex_length,is_matched;..."""
    if not inlining_list:
        return ""
    parts = []
    for idx, start, length, is_matched in inlining_list:
        hex_start = f"{start:x}"
        hex_length = f"{length:x}"
        parts.append(f"{idx},{hex_start},{hex_length},{is_matched}")
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

    Used for the called-funcs cell in both section CSVs: every function
    name there is replaced by its sidecar line no. Empty sequence ->
    empty string.
    """
    return encode_line_nos_csv(line_nos)


def parse_function_line_nos_csv(s: str) -> List[int]:
    """Inverse of :func:`format_function_line_nos_csv`."""
    return decode_line_nos_csv(s)


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
