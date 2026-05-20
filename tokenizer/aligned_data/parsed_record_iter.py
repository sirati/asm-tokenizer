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

from tokenizer.compact_base64_utils import base64_to_ndarray_vec

from .match import (
    PositionTrackingWrapper,
    open_csv_skip_vocab,
)


@dataclass
class ParsedRecord:
    """One function row from one input CSV, fully parsed + hashed."""

    func_name: str
    insn_runlength: np.ndarray
    block_runlength: np.ndarray
    tokens: np.ndarray
    called_funcs: List[str]
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
    called = _extract_called_funcs(row, column_index)
    content_hash = _hash_record_body(insn_runlength, block_runlength, tokens)
    return ParsedRecord(
        func_name=func_name,
        insn_runlength=insn_runlength,
        block_runlength=block_runlength,
        tokens=tokens,
        called_funcs=called,
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


def _extract_called_funcs(
    row: List[str],
    column_index: "dict[str, int]",
) -> List[str]:
    # Schema dispatch mirrors the pre-refactor
    # ``helpers.get_called_functions_from_row`` — v2 carries a
    # ``metadata`` JSON column; v1 carries the Python-repr
    # ``opaque_metadata`` column.
    if "metadata" in column_index:
        return _called_from_v2_metadata(row[column_index["metadata"]])
    if "opaque_metadata" in column_index:
        return _called_from_v1_opaque_metadata(row[column_index["opaque_metadata"]])
    return []


def _called_from_v2_metadata(metadata_cell: str) -> List[str]:
    if not metadata_cell:
        return []
    try:
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
