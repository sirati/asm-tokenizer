from typing import Any, Dict, List


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
    """Given a row from the section CSV and its header, extract metadata as a dict.

    Returns: dict with keys ``variant_ref``, ``inlining_data``,
    ``data_offset``, ``data_len``. The variant ref is the
    ``0x<hex>`` row index into the per-group ``<binary>_variants.csv``
    sidecar — resolution to the canonical-4 axes / extra-metadata is
    a separate consumer-side concern and not performed here.
    """
    idx = {k: i for i, k in enumerate(header)}
    return {
        "variant_ref": row[idx["variant_ref"]],
        "inlining_data": parse_inlining_data(row[idx["inlining_data"]]),
        "data_offset": int(row[idx["data_offset"]], 16),
        "data_len": int(row[idx["data_len"]], 16),
    }


def extract_all_metadata_from_section_rows(rows: List[List[str]], header: List[str]) -> List[Dict[str, Any]]:
    return [extract_metadata_from_section_row(row, header) for row in rows]
