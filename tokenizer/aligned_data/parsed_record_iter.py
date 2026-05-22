"""Per-CSV parsed-record iterator + typed-sum lockstep merge.

A :class:`ParsedRecord` carries everything pass-1 needs to call into the
dedup helper for one variant of one function: decoded ndarrays, the
called-functions list, and the 64-bit xxh3 of the canonical body bytes
(``insn || block || tokens``, post-mapping). Hashing happens once per
record at the iterator boundary so the same value flows through the
dedup primary map without rehashing on every collision lookup.

Two concerns are pushed down here from the legacy
``process_function_binary_data`` path:

* base64 decode of tokens + runlengths (was in ``aligned_data/io.py``);
* called-function extraction from the v1/v2 metadata column (was in
  the now-removed ``memmap_builder/helpers.py``).

The lockstep merge :func:`lockstep_records` consumes N per-CSV
iterators (one per build variant) and yields a tagged union per
function name:

* :class:`Matched` when the function name appears in ≥ 2 input CSVs.
  ``records`` is a dict keyed by the variant index (the position in
  the ``per_csv_iters`` list), mapping to the parsed record from that
  CSV.
* :class:`Unmatched` when the function name appears in exactly one
  input CSV. ``record`` is the single parsed record; ``variant_index``
  is its position in ``per_csv_iters``.

Both arms of the memmap builder (and the validator) dispatch on this
typed sum instead of the pre-refactor ``count``-based dict yield.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Union

import numpy as np
import xxhash

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.compact_base64_utils import base64_to_ndarray_vec

from .match import (
    PositionTrackingWrapper,
    open_csv_skip_vocab,
)


@dataclass
class ParsedRecord:
    """One function row from one input CSV, fully parsed + hashed.

    ``extern_libraries`` maps EXTERN-callee names to the library string
    reported by the disassembler. v1 rows (which never carry library
    info) always populate an empty dict; v2 rows populate one entry per
    ``ext_funcs`` member whose ``library`` is not ``None``.
    """

    func_name: str
    insn_runlength: np.ndarray
    block_runlength: np.ndarray
    tokens: np.ndarray
    # Encoder-allocation order preserved per category, concatenated in
    # LOCAL -> PLT -> EXTERN order. The K-th entry of category C in this
    # list equals the function whose encoder-assigned identity for C is K.
    called_funcs: list[tuple[str, CallTargetType]]
    extern_libraries: dict[str, str]
    content_hash: int


@dataclass
class Matched:
    """Function with rows from ≥ 2 variants. ``records`` is keyed by variant index."""

    func_name: str
    records: Dict[int, ParsedRecord]


@dataclass
class Unmatched:
    """Function with a row from exactly one variant."""

    func_name: str
    record: ParsedRecord
    variant_index: int


LockstepYield = Union[Matched, Unmatched]


def open_parsed_record_iter(
    csv_path: str,
    mapping: Optional[np.ndarray] = None,
) -> "tuple[PositionTrackingWrapper, Iterator[ParsedRecord], List[str]]":
    """Open ``csv_path`` and return ``(wrapper, iterator, header)``.

    ``wrapper`` is the position-tracking handle the lockstep merge
    polls for progress reporting (``wrapper.get_position()`` ⇒ bytes
    read). The iterator yields one :class:`ParsedRecord` per function
    row, with the v2 ``version=`` prelude + vocab rows already
    filtered out.

    ``mapping`` is the per-CSV local-ID → unified-ID lookup loaded by
    the builder from ``<base>.mapping.b64c``. Threaded into every
    token decode so the hash is computed on unified-vocab bytes.
    """
    wrapper, raw_rows, header = open_csv_skip_vocab(csv_path)
    column_index = {field: idx for idx, field in enumerate(header)}

    def iterator() -> Iterator[ParsedRecord]:
        for row in raw_rows:
            yield _parse_row(row, column_index, mapping)

    return wrapper, iterator(), header


def lockstep_records(
    per_csv_iters: List[Iterator[ParsedRecord]],
    wrappers: Optional[List[PositionTrackingWrapper]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Iterator[LockstepYield]:
    """Merge N per-CSV ParsedRecord iterators into a typed-sum stream.

    Inputs must be sorted by ``func_name`` (every per-CSV CSV is
    already sorted; the memmap-output chain guarantees it).
    """
    current: List[Optional[ParsedRecord]] = []
    for it in per_csv_iters:
        try:
            current.append(next(it))
        except StopIteration:
            current.append(None)

    iteration_count = 0
    while True:
        iteration_count += 1
        names = [r.func_name if r is not None else None for r in current]
        if all(n is None for n in names):
            break
        min_name = min(n for n in names if n is not None)
        matching_indices = [i for i, n in enumerate(names) if n == min_name]

        if len(matching_indices) >= 2:
            yield Matched(
                func_name=min_name,
                records={i: current[i] for i in matching_indices},  # type: ignore[misc]
            )
        else:
            (i,) = matching_indices
            yield Unmatched(
                func_name=min_name,
                record=current[i],  # type: ignore[arg-type]
                variant_index=i,
            )

        for i in matching_indices:
            try:
                current[i] = next(per_csv_iters[i])
            except StopIteration:
                current[i] = None

        if (
            progress_callback is not None
            and wrappers is not None
            and iteration_count % 100 == 0
        ):
            progress_callback(sum(w.get_position() for w in wrappers))

    if progress_callback is not None and wrappers is not None:
        progress_callback(sum(w.get_position() for w in wrappers))


def _parse_row(
    row: List[str],
    column_index: "dict[str, int]",
    mapping: Optional[np.ndarray],
) -> ParsedRecord:
    func_name = row[0]
    tokens = _decode_tokens(row, column_index, mapping)
    block_runlength = base64_to_ndarray_vec(
        row[column_index["block_runlength_base64"]]
    )
    insn_runlength = base64_to_ndarray_vec(
        row[column_index["instruction_runlength_base64"]]
    )
    called, extern_libraries = _extract_called_funcs(row, column_index)
    content_hash = _hash_record_body(insn_runlength, block_runlength, tokens)
    return ParsedRecord(
        func_name=func_name,
        insn_runlength=insn_runlength,
        block_runlength=block_runlength,
        tokens=tokens,
        called_funcs=called,
        extern_libraries=extern_libraries,
        content_hash=content_hash,
    )


def _decode_tokens(
    row: List[str],
    column_index: "dict[str, int]",
    mapping: Optional[np.ndarray],
) -> np.ndarray:
    tokens = base64_to_ndarray_vec(row[column_index["tokens_base64"]])
    if mapping is not None:
        tokens = mapping[tokens]
    return tokens.astype(np.uint16)


_V2_CATEGORY_TYPES: "tuple[tuple[str, CallTargetType], ...]" = (
    ("local_funcs", CallTargetType.LOCAL),
    ("plt_funcs", CallTargetType.PLT),
    ("ext_funcs", CallTargetType.EXTERN),
)


def _extract_called_funcs(
    row: List[str],
    column_index: "dict[str, int]",
) -> "tuple[list[tuple[str, CallTargetType]], dict[str, str]]":
    """Return ``(typed_called, extern_libraries)`` for one CSV row.

    Schema dispatch mirrors the pre-refactor
    ``helpers.get_called_functions_from_row`` — v2 carries a ``metadata``
    JSON column; v1 carries the Python-repr ``opaque_metadata`` column.
    The library dict is only populated by v2; v1 always returns ``{}``.
    """
    if "metadata" in column_index:
        return _called_from_v2_metadata(row[column_index["metadata"]])
    if "opaque_metadata" in column_index:
        return _called_from_v1_opaque_metadata(row[column_index["opaque_metadata"]])
    return [], {}


def _called_from_v2_metadata(
    metadata_cell: str,
) -> "tuple[list[tuple[str, CallTargetType]], dict[str, str]]":
    if not metadata_cell:
        return [], {}
    try:
        meta = json.loads(metadata_cell)
    except Exception:
        return [], {}
    if not isinstance(meta, dict):
        return [], {}
    # Preserve the encoder's per-category allocation order: each
    # ``_V2_CATEGORY_TYPES`` array is the CSV's identity-indexed
    # metadata cell, so the K-th name in category C is the function
    # whose encoder-assigned identity for C is K. Dedupe inside one
    # category via ``dict.fromkeys`` (order-preserving primitive); the
    # same name appearing in two categories (e.g. PLT ``foo`` + EXTERN
    # ``foo``) still surfaces as two distinct ``(name, type)`` entries.
    # Categories are concatenated in LOCAL -> PLT -> EXTERN order.
    called: list[tuple[str, CallTargetType]] = []
    extern_libraries: dict[str, str] = {}
    for category_key, category_type in _V2_CATEGORY_TYPES:
        names_in_order: list[str] = []
        for entry in meta.get(category_key, ()) or ():
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str):
                    names_in_order.append(name)
                    if category_type is CallTargetType.EXTERN:
                        library = entry.get("library")
                        if isinstance(library, str):
                            extern_libraries[name] = library
        for unique_name in dict.fromkeys(names_in_order):
            called.append((unique_name, category_type))
    return called, extern_libraries


def _called_from_v1_opaque_metadata(
    opaque_metadata: str,
) -> "tuple[list[tuple[str, CallTargetType]], dict[str, str]]":
    # v1 carries only ``local_function`` callees, in disassembler
    # encounter order. Mirror the v2 path: order-preserving dedupe via
    # ``dict.fromkeys`` instead of set + alphabetical sort, so the K-th
    # surviving name equals the encoder's K-th LOCAL allocation.
    try:
        meta = ast.literal_eval(opaque_metadata)
        names_in_order: list[str] = []
        for entry in meta:
            if isinstance(entry, tuple) and len(entry) >= 5:
                name = entry[2]
                type_field = entry[3]
                if type_field == "local_function":
                    names_in_order.append(name)
        return (
            [(name, CallTargetType.LOCAL) for name in dict.fromkeys(names_in_order)],
            {},
        )
    except Exception:
        return [], {}


def _hash_record_body(
    insn_runlength: np.ndarray,
    block_runlength: np.ndarray,
    tokens: np.ndarray,
) -> int:
    """xxh3-64 of the canonical body byte concatenation."""
    hasher = xxhash.xxh3_64()
    hasher.update(insn_runlength.astype(np.uint8).tobytes())
    hasher.update(block_runlength.tobytes())
    hasher.update(tokens.astype(np.uint16).tobytes())
    return hasher.intdigest()
