from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from tokenizer.compact_base64_utils import base64_to_ndarray_vec

from ..aligned_data.io import (
    decode_and_translate_tokens,
    decode_runlengths,
    write_function_binary_data,
)
from .variants import VariantRegistry, write_warn_log_entry


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
    """Extract called function names from the per-function metadata column.

    Schema dispatch is driven by which column key the row carries — that
    key is the one the writer chose based on ``vocab_manager.format_version``
    (see ``tokenizer/main_loop.py``: ``metadata`` for v2, ``opaque_metadata``
    for v1) and the column-mapping step in ``lockstep_function_match``
    propagates the original column name into the row dict verbatim, so the
    presence of one key vs the other unambiguously identifies the wire
    format without any caller-side plumbing.

    v2 (``metadata`` key): the column is a JSON object with the shape
    documented in ``main_loop._build_v2_metadata_json`` — categories
    ``local_funcs``, ``plt_funcs``, ``ext_funcs`` each hold a list of
    ``{"name": ..., ...}`` entries; the per-function callee set is the
    union of names across all three (callees regardless of binding kind
    are call targets for inlining-lookup purposes in v2).

    v1 (``opaque_metadata`` key): the column is the Python ``repr`` of a
    list of tuples; only entries whose 4th field equals ``"local_function"``
    are callees. Preserved verbatim from the pre-v2 implementation.
    """
    if "metadata" in row:
        return _called_from_v2_metadata(row["metadata"])
    return _called_from_v1_opaque_metadata(row.get("opaque_metadata", ""))


def _called_from_v2_metadata(metadata_cell: str) -> List[str]:
    if not metadata_cell:
        return []
    try:
        import json

        meta = json.loads(metadata_cell)
    except Exception:
        return []
    if not isinstance(meta, dict):
        return []
    called = set()
    for category_key in ("local_funcs", "plt_funcs", "ext_funcs"):
        for entry in meta.get(category_key, ()) or ():
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str):
                    called.add(name)
    return sorted(called)


def _called_from_v1_opaque_metadata(opaque_metadata: str) -> List[str]:
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
    variants: VariantRegistry,
) -> List[InliningEntry]:
    """Build inlining data list with lookups, logging warnings for missing entries."""
    variant_ref = variants.ref(vkey)
    inlining_data = {}
    for called_func in called_functions:
        called_idx = unique_called.index(called_func)
        lookup_key = (called_func, vkey)
        if lookup_key in function_lookup:
            func_offset, func_len, is_matched = function_lookup[lookup_key]
            inlining_data[called_idx] = (func_offset, func_len, is_matched)
        else:
            write_warn_log_entry(warn_log, func_name, variant_ref, called_func)

    inlining_list = [
        InliningEntry(idx=idx, offset=start, length=length, is_matched=is_matched)
        for idx, (start, length, is_matched) in sorted(inlining_data.items())
    ]
    return inlining_list


def format_inlining_list(inlining_list: List[InliningEntry]) -> List[List]:
    """Convert InliningEntry list to format expected by CSV writer."""
    return [[entry.idx, entry.offset, entry.length, entry.is_matched] for entry in inlining_list]
