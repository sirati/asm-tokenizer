import json
from typing import Any, Dict, List


def parse_inlining_data(inlining_str: str) -> List[List[int]]:
    """
    Parse inlining data string format: "idx,hex_offset,hex_length,is_matched;idx,hex_offset,hex_length,is_matched;..."
    Returns list of [idx, offset, length, is_matched]
    """
    if not inlining_str:
        return []
    result = []
    for entry in inlining_str.split(";"):
        if entry:
            parts = entry.split(",")
            if len(parts) == 4:
                result.append([int(parts[0]), int(parts[1], 16), int(parts[2], 16), int(parts[3])])
    return result


def extract_metadata_from_section_row(row: List[str], header: List[str]) -> Dict[str, Any]:
    """
    Given a row from the section CSV and its header, extract metadata fields as a dict.
    Returns: dict with keys: arch, compiler, compilerversion, opt, inlining_data, data_offset, data_len
    """
    idx = {k: i for i, k in enumerate(header)}
    return {
        "arch": row[idx["arch"]],
        "compiler": row[idx["compiler"]],
        "compilerversion": row[idx["compilerversion"]],
        "opt": row[idx["opt"]],
        "inlining_data": parse_inlining_data(row[idx["inlining_data"]]),
        "data_offset": int(row[idx["data_offset"]], 16),
        "data_len": int(row[idx["data_len"]], 16),
    }


def extract_all_metadata_from_section_rows(rows: List[List[str]], header: List[str]) -> List[Dict[str, Any]]:
    return [extract_metadata_from_section_row(row, header) for row in rows]
