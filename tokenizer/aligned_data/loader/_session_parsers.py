"""Stateless parsers used by ``BinarySession``.

Split out so ``session.py`` stays focused on lifecycle: section-row
parsing into ``FunctionData`` / ``MatchedFunction`` is pure on its
inputs (raw bytes/rows + a resolver callable) and has no business
touching the lazy-open machinery. Tests can exercise these helpers
without standing up a full session.
"""

from __future__ import annotations

import csv
from typing import Any, Callable, List, Optional

from ..metadata import extract_metadata_from_section_row, parse_inlining_data
from .function_data import FunctionData
from .matched_function import MatchedFunction


def arm_arrays(arm: Any, kind: str, binary_name: str):
    if arm is None:
        raise IndexError(f"{kind} arm not loaded for binary {binary_name}")
    starts = getattr(arm, "starts", None)
    lengths = getattr(arm, "lengths", None)
    if starts is None or lengths is None:
        raise IndexError(
            f"{kind} arm has no starts/lengths for binary {binary_name}"
        )
    return starts, lengths


def parse_matched_section(
    section_data,
    *,
    func_name_override: Optional[str] = None,
    data_slice: Callable,
    resolve_ref: Callable,
) -> MatchedFunction:
    """Parse a matched-section blob into a ``MatchedFunction``.

    ``data_slice(offset, length)`` returns ``(insn_rl, block_rl, tokens)``;
    ``resolve_ref(ref_str)`` returns the variant dict (or ``None``).
    Both injected so this helper does not import the session's lazy
    openers.
    """
    text = section_data.strip() if isinstance(section_data, str) else section_data
    lines = text.split("\n")
    func_name = func_name_override or list(csv.reader([lines[0]]))[0][0]

    versions: List[FunctionData] = []
    header = ["variant_ref", "inlining_data", "data_offset", "data_len"]
    for row in csv.reader(lines[1:]):
        if not row:
            continue
        metadata = extract_metadata_from_section_row(row, header)
        variant_row = resolve_ref(metadata["variant_ref"])
        if variant_row is not None:
            for k, v in variant_row.items():
                metadata.setdefault(k, v)
        insn_rl, block_rl, tokens = data_slice(
            metadata["data_offset"], metadata["data_len"]
        )
        versions.append(
            FunctionData(func_name, metadata, tokens, insn_rl, block_rl)
        )
    return MatchedFunction(func_name, versions)


def build_unmatched_function_data(
    row: Optional[List[str]],
    idx: int,
    start: int,
    length: int,
    tokens,
    insn_rl,
    block_rl,
    *,
    resolve_ref: Callable,
) -> FunctionData:
    """Assemble an unmatched ``FunctionData`` from its CSV row + bytes.

    ``row`` may be ``None`` (CSV row not recoverable) — caller passes
    the bytes-derived ``start``/``length`` as fallback offset/length.
    """
    func_name_default = f"unmatched_{idx}"
    if row is None:
        func_name = func_name_default
        platform_info, called, inlining_data = "unknown", [], []
        data_offset_csv, data_len_csv = start, length
    else:
        func_name = row[0] or func_name_default
        platform_info = row[1]
        called = (
            [name.replace("\\,", ",") for name in row[2].split(",") if name]
            if row[2] else []
        )
        inlining_data = parse_inlining_data(row[3])
        try:
            data_offset_csv = int(row[4], 16)
            data_len_csv = int(row[5], 16)
        except ValueError:
            data_offset_csv, data_len_csv = start, length

    variant_refs = (
        [r for r in platform_info.split(";") if r] if platform_info else []
    )
    variants = [v for v in (resolve_ref(r) for r in variant_refs) if v is not None]

    metadata = {
        "arch": "unknown",
        "compiler": "unknown",
        "compilerversion": "unknown",
        "opt": "unknown",
        "platform_info": platform_info,
        "variant_refs": variant_refs,
        "variants": variants,
        "called": called,
        "inlining_data": inlining_data,
        "data_offset": data_offset_csv,
        "data_len": data_len_csv,
    }
    return FunctionData(func_name, metadata, tokens, insn_rl, block_rl)
