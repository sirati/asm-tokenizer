"""Stateless parsers used by ``BinarySession``.

Split out so ``session.py`` stays focused on lifecycle: section-row
parsing into ``FunctionData`` / ``MatchedFunction`` is pure on its
inputs (raw bytes/rows + a resolver callable) and has no business
touching the lazy-open machinery. Tests can exercise these helpers
without standing up a full session.
"""

from __future__ import annotations

import csv
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..metadata import extract_metadata_from_section_row, parse_inlining_data
from .function_data import FunctionData
from .matched_function import MatchedFunction


# Empty uint16 buffer reused when a variant ref cannot be resolved. Sharing one
# instance avoids per-call allocation; consumers MUST treat it as read-only
# (matches the rest of the loader's lazy-view discipline).
_EMPTY_VARIANT_TOKENS: np.ndarray = np.zeros(0, dtype=np.uint16)


def _variant_tokens_from_row(variant_row: Optional[Dict[str, Any]]) -> np.ndarray:
    """Pull the resolver's ``variant_tokens`` ndarray off a variant dict.

    Empty / missing variant row -> the shared empty uint16 buffer so
    ``FunctionData.variant_tokens`` is always a valid uint16 ndarray
    (zero-length signals "no variant resolved", matching the plan's
    "zero-length only on a corrupt dataset" contract).
    """
    if not variant_row:
        return _EMPTY_VARIANT_TOKENS
    tokens = variant_row.get("variant_tokens")
    if tokens is None:
        return _EMPTY_VARIANT_TOKENS
    return tokens


def arm_arrays(arm: Any, kind: str, binary_name: str):
    """Per-function arrays the session uses to slice for ``kind``.

    Matched ``load(idx)`` slices the section CSV (per-function), so it
    needs the CSV bounds from ``csv_starts``/``csv_lengths``. Unmatched
    ``load(idx)`` slices ``_data.bin`` directly (the unmatched index is
    per-function 1:1), so it needs ``starts``/``lengths``.
    """
    if arm is None:
        raise IndexError(f"{kind} arm not loaded for binary {binary_name}")
    if kind == "matched":
        starts = getattr(arm, "csv_starts", None)
        lengths = getattr(arm, "csv_lengths", None)
        attr_pair = "csv_starts/csv_lengths"
    else:
        starts = getattr(arm, "starts", None)
        lengths = getattr(arm, "lengths", None)
        attr_pair = "starts/lengths"
    if starts is None or lengths is None:
        raise IndexError(
            f"{kind} arm has no {attr_pair} for binary {binary_name}"
        )
    return starts, lengths


# Matched-section variant-row schema after the matched-arm restructuring.
# Pinned here so the row decoder + the helper that walks the row both see
# the same column order (a future schema bump touches this one definition).
_MATCHED_VARIANT_HEADER: List[str] = ["variant_ref", "inlining_data", "indexer_hex"]


def parse_matched_section(
    section_data,
    *,
    func_name_override: Optional[str] = None,
    data_slice: Callable,
    resolve_ref: Callable,
) -> MatchedFunction:
    """Parse a matched-section blob into a ``MatchedFunction``.

    ``data_slice(offset, length, is_overlong)`` returns
    ``(insn_rl, block_rl, tokens)``; ``resolve_ref(ref_str)`` returns
    the variant dict (or ``None``). Both injected so this helper does
    not import the session's lazy openers.

    Per the post-restructuring layout each variant row is 3 cells
    ``[variant_ref, inlining_data, indexer_hex]``; the inline indexer
    decode populates ``data_offset`` / ``data_len`` / ``is_overlong`` on
    the metadata dict (and is the only authoritative source for the
    record's overlong flag on the matched arm, since no separate index
    file tracks per-variant data positions).
    """
    text = section_data.strip() if isinstance(section_data, str) else section_data
    lines = text.split("\n")
    func_name = func_name_override or list(csv.reader([lines[0]]))[0][0]

    versions: List[FunctionData] = []
    for row in csv.reader(lines[1:]):
        if not row:
            continue
        metadata = extract_metadata_from_section_row(row, _MATCHED_VARIANT_HEADER)
        variant_row = resolve_ref(metadata["variant_ref"])
        if variant_row is not None:
            for k, v in variant_row.items():
                metadata.setdefault(k, v)
        insn_rl, block_rl, tokens = data_slice(
            metadata["data_offset"],
            metadata["data_len"],
            metadata["is_overlong"],
        )
        versions.append(
            FunctionData(
                func_name, metadata, tokens, insn_rl, block_rl,
                variant_tokens=_variant_tokens_from_row(variant_row),
            )
        )
    return MatchedFunction(func_name, versions)


def parse_unmatched_row(row: List[str]) -> Tuple[List[str], List[str], List[List[int]]]:
    """Parse a 5-cell unmatched section row into its semantic fields.

    Post matched-arm restructuring the unmatched row layout is
    ``[line_no_b64, variant_refs, called_b64, inlining, indexer_hex]``:

    * ``line_no_b64`` -- caller resolves to the function name via the
      function-names sidecar; not consumed here (the session passes the
      resolved name in separately).
    * ``variant_refs`` -- semicolon-joined ``0x<hex>`` byte offsets into
      ``_variants.bin``.
    * ``called_b64`` -- comma-joined base64 line numbers; this helper
      surfaces them as the raw base64 strings so the session layer can
      resolve them through the sidecar (this module stays
      indirection-agnostic).
    * ``inlining`` -- comma-joined per-version inlining tuples (parsed
      via :func:`parse_inlining_data`).
    * ``indexer_hex`` -- ignored on the unmatched arm because the
      index file (``unmatched_index.bin``) is the authoritative source
      for per-version data positions; the cell is kept inline for
      validator cross-checks and round-trip parity with the matched arm.
    """
    variant_refs = [r for r in row[1].split(";") if r] if row[1] else []
    called_b64s = (
        [name.replace("\\,", ",") for name in row[2].split(",") if name]
        if row[2] else []
    )
    inlining_data = parse_inlining_data(row[3])
    return variant_refs, called_b64s, inlining_data


def build_unmatched_function_data(
    row: Optional[List[str]],
    idx: int,
    func_name: str,
    start: int,
    length: int,
    is_overlong: bool,
    tokens,
    insn_rl,
    block_rl,
    *,
    resolve_ref: Callable,
) -> FunctionData:
    """Assemble an unmatched ``FunctionData`` from its CSV row + bytes.

    ``row`` may be ``None`` (CSV row not recoverable) -- callers supply
    the index-derived ``start`` / ``length`` / ``is_overlong`` directly
    and the metadata dict falls back to "unknown" placeholders. The
    function name is resolved by the caller (session layer) via the
    function-names sidecar; this helper never decodes ``line_no_b64``.
    """
    if row is None:
        variant_refs, called_b64s, inlining_data = [], [], []
    else:
        variant_refs, called_b64s, inlining_data = parse_unmatched_row(row)

    variants = [v for v in (resolve_ref(r) for r in variant_refs) if v is not None]

    metadata = {
        "arch": "unknown",
        "compiler": "unknown",
        "compilerversion": "unknown",
        "opt": "unknown",
        "variant_refs": variant_refs,
        "variants": variants,
        # ``called`` is exposed as the raw base64 line numbers because
        # this module does not own the function-names sidecar; the
        # AlignedDataLoader layer resolves them when callers ask for
        # human-readable names.
        "called": called_b64s,
        "inlining_data": inlining_data,
        "data_offset": start,
        "data_len": length,
        "is_overlong": is_overlong,
    }
    # Unmatched functions span multiple variants; ``FunctionData.variant_tokens``
    # carries the first resolved variant's prefix (deterministic by section-row
    # order). Consumers wanting the full per-variant axis token streams find
    # them on ``metadata["variants"][i]["variant_tokens"]``.
    primary_variant = variants[0] if variants else None
    return FunctionData(
        func_name, metadata, tokens, insn_rl, block_rl,
        variant_tokens=_variant_tokens_from_row(primary_variant),
    )
