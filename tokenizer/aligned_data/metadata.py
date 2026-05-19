from typing import Any, Dict, List

from .inline_indexer import decode_inline_indexer


def parse_inlining_data(inlining_str: str) -> List[List[int]]:
    """
    Parse inlining data string format: "idx,hex_offset,hex_length,is_matched;idx,hex_offset,hex_length,is_matched;..."
    For unmatched format: "idx-comp_set,hex_offset,hex_length,is_matched;..."
    Returns list of [idx, offset, length, is_matched]
    Note: comp_set information is discarded for unmatched functions
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


def extract_metadata_from_section_row(row: List[str], header: List[str]) -> Dict[str, Any]:
    """Given a matched-section variant row + its header, extract metadata.

    Post matched-arm restructuring the variant row is 3 cells
    ``[variant_ref, inlining_data, indexer_hex]`` (the previous
    ``data_offset, data_len`` pair was replaced by the single
    ``indexer_hex`` cell -- the 16-hex-char inline encoding of the v1
    index entry). This helper decodes ``indexer_hex`` through
    :func:`tokenizer.aligned_data.inline_indexer.decode_inline_indexer`
    so the layout knowledge lives in one place and the returned dict
    keeps the same downstream keys (``data_offset``, ``data_len``,
    ``is_overlong``).

    Returns: dict with keys ``variant_ref``, ``inlining_data``,
    ``data_offset``, ``data_len``, ``is_overlong``. The variant ref is
    the ``0x<hex>`` row index into the per-group
    ``<binary>_variants.csv`` sidecar -- resolution to the canonical-4
    axes / extra-metadata is a separate consumer-side concern and not
    performed here.
    """
    idx = {k: i for i, k in enumerate(header)}
    data_offset, data_len, is_overlong = decode_inline_indexer(
        row[idx["indexer_hex"]]
    )
    return {
        "variant_ref": row[idx["variant_ref"]],
        "inlining_data": parse_inlining_data(row[idx["inlining_data"]]),
        "data_offset": data_offset,
        "data_len": data_len,
        "is_overlong": is_overlong,
    }


def extract_all_metadata_from_section_rows(rows: List[List[str]], header: List[str]) -> List[Dict[str, Any]]:
    return [extract_metadata_from_section_row(row, header) for row in rows]
