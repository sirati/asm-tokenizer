import json
from typing import Dict, Any, List

def extract_metadata_from_section_row(row: List[str], header: List[str]) -> Dict[str, Any]:
    """
    Given a row from the section CSV and its header, extract metadata fields as a dict.
    Returns: dict with keys: arch, compiler, compilerversion, opt, called, inlining_map, data_offset, data_len
    """
    idx = {k: i for i, k in enumerate(header)}
    return {
        'arch': row[idx['arch']],
        'compiler': row[idx['compiler']],
        'compilerversion': row[idx['compilerversion']],
        'opt': row[idx['opt']],
        'called': json.loads(row[idx['called']]),
        'inlining_map': json.loads(row[idx['inlining_map']]),
        'data_offset': int(row[idx['data_offset']]),
        'data_len': int(row[idx['data_len']]),
    }

def extract_all_metadata_from_section_rows(rows: List[List[str]], header: List[str]) -> List[Dict[str, Any]]:
    return [extract_metadata_from_section_row(row, header) for row in rows]

