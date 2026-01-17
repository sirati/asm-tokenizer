from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec

from ..aligned_data.io import (
    decode_and_translate_tokens,
    decode_runlengths,
    write_function_binary_data,
)


@dataclass
class FunctionBinaryData:
    data_offset: int
    data_len: int
    token_len: int


@dataclass
class InliningEntry:
    idx: int
    offset: int
    length: int
    is_matched: int


def should_skip_function(func_name: str, row: Optional[dict] = None) -> bool:
    """Check if function should be skipped based on name and optional row data."""
    if func_name.startswith(".L"):
        return True
    if row is not None and "block_runlength_base64" in row:
        try:
            block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
            if block_runlength.sum() >= 4096:
                return True
        except Exception:
            return True
    return False


def should_skip_function_for_matched(rows: List[Optional[dict]]) -> bool:
    """Check if any row in matched function set should cause skipping."""
    for row in rows:
        if row is not None and "block_runlength_base64" in row:
            try:
                block_runlength = base64_to_ndarray_vec(row["block_runlength_base64"])
                if block_runlength.sum() >= 4096:
                    return True
            except Exception:
                return True
    return False


def should_skip_function_for_unmatched(row: dict) -> bool:
    """Check if unmatched function should be skipped (note: different logic than matched)."""
    if "block_runlength" in row:
        try:
            block_runlength = base64_to_ndarray_vec(row["block_runlength"])
            if block_runlength.sum() < 4096:
                return True
        except Exception:
            pass
    return False


def process_function_binary_data(
    row: dict,
    mapping: Optional[np.ndarray],
    data_file,
    dedup_cache: dict,
) -> FunctionBinaryData:
    """Decode, translate, and write binary data for a function. Returns offset, length, token count."""
    tokens = decode_and_translate_tokens(row, mapping)
    block_runlength, insn_runlength = decode_runlengths(row)
    data_offset, data_len = write_function_binary_data(data_file, tokens, block_runlength, insn_runlength, dedup_cache)
    return FunctionBinaryData(data_offset=data_offset, data_len=data_len, token_len=len(tokens))


def get_called_functions_from_row(row: dict) -> List[str]:
    """Extract called function names from opaque_metadata field."""
    opaque_metadata = row.get("opaque_metadata", "")
    try:
        import ast

        meta = ast.literal_eval(opaque_metadata)
        called = set()
        for entry in meta:
            if isinstance(entry, tuple) and len(entry) >= 5:
                name = entry[2]
                type_field = entry[3]
                if type_field == "local_function":
                    called.add(name)
        return sorted(called)
    except Exception:
        return []


def collect_unique_called_functions(all_called_by_key: Dict, version_keys: List) -> List[str]:
    """Collect and sort unique called function names across all versions."""
    unique_called = sorted(set(fn for called_list in all_called_by_key.values() for fn in called_list))
    return unique_called


def build_inlining_data(
    called_functions: List[str],
    unique_called: List[str],
    vkey,
    function_lookup: dict,
    warn_log,
    func_name: str,
) -> List[InliningEntry]:
    """Build inlining data list with lookups, logging warnings for missing entries."""
    inlining_data = {}
    for called_func in called_functions:
        called_idx = unique_called.index(called_func)
        lookup_key = (called_func, vkey)
        if lookup_key in function_lookup:
            func_offset, func_len, is_matched = function_lookup[lookup_key]
            inlining_data[called_idx] = (func_offset, func_len, is_matched)
        else:
            warn_log.write(f"{func_name},{vkey.arch},{vkey.compiler},{vkey.compilerversion},{vkey.opt},{called_func}\n")

    inlining_list = [
        InliningEntry(idx=idx, offset=start, length=length, is_matched=is_matched)
        for idx, (start, length, is_matched) in sorted(inlining_data.items())
    ]
    return inlining_list


def format_inlining_list(inlining_list: List[InliningEntry]) -> List[List]:
    """Convert InliningEntry list to format expected by CSV writer."""
    return [[entry.idx, entry.offset, entry.length, entry.is_matched] for entry in inlining_list]
