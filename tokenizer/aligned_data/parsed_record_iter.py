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
from dataclasses import dataclass, field
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
    # Per-binary-CSV occurrence ordinal for this ``func_name``. The body-
    # divergence deduper (``tokenizer/main_loop.py``) keeps every
    # genuinely-distinct function that happens to share a canonical name
    # (per-TU static initializers, thunks, anon-namespace collisions) and
    # bumps this column: 0 for the first body, 1 for the second, and so
    # on. A name whose stream tops out at occurrence 0 is unique; any name
    # that reaches occurrence >= 1 is DUPLICATED. v1 ``opaque_metadata``
    # rows predate the column and default to 0.
    occurrence: int
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
    """Function with rows from ≥ 2 variants. ``records`` is keyed by variant index.

    Only ever produced for a name that is UNIQUE in every input stream
    (occurrence-0-only). A name that is duplicated in any single stream
    (occurrence reaches >= 1) is structurally barred from this arm by
    :func:`lockstep_records` — it cannot be aligned variant-against-variant
    because the name no longer identifies a single function.
    """

    func_name: str
    records: Dict[int, ParsedRecord]


@dataclass
class Unmatched:
    """Function with a row from exactly one variant.

    Also the destination for every row of a DUPLICATED name (see
    :class:`Matched`): each duplicate body is emitted as its own
    ``Unmatched`` carrying the originating stream's ``variant_index``,
    so the unmatched arm groups them into one section's variant blocks
    rather than same-FID sibling sections.
    """

    func_name: str
    record: ParsedRecord
    variant_index: int


LockstepYield = Union[Matched, Unmatched]


@dataclass
class DuplicateNameClassifier:
    """One-boundary record of which function names are DUPLICATED in a binary.

    The duplication question ("does this canonical name map to more than
    one distinct function within this binary?") is answered exactly once,
    by :func:`lockstep_records`, while it holds the only complete view of
    every stream's rows for a name. The answer is consumed by two
    downstream concerns that must agree:

    * def-routing — handled implicitly by :func:`lockstep_records` itself
      emitting duplicated names down the :class:`Unmatched` arm;
    * call-side J stamping — pass 2 reads :attr:`duplicated_names` to
      decide that a call edge into a duplicated callee is RECORDED but
      its per-call ``section_variant_index`` is unresolvable.

    The set is populated as the generator runs and is complete once the
    caller has fully drained the :func:`lockstep_records` stream.
    """

    duplicated_names: set = field(default_factory=set)


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


def _drain_same_name_group(
    it: Iterator[ParsedRecord],
    first: ParsedRecord,
) -> "tuple[list[ParsedRecord], Optional[ParsedRecord]]":
    """Consume ``first`` plus every consecutive same-name row from ``it``.

    The per-CSV CSV is sorted by ``(func_name, occurrence)``, so all rows
    for one name are contiguous in a single stream. Returns ``(group,
    next_lookahead)`` where ``group`` is the (≥ 1)-length run sharing
    ``first.func_name`` and ``next_lookahead`` is the first row of the
    NEXT name (or ``None`` at end-of-stream) — the value that re-seeds the
    stream's one-record lookahead. ``len(group) >= 2`` is the in-stream
    duplication signal.
    """
    group = [first]
    name = first.func_name
    while True:
        try:
            nxt = next(it)
        except StopIteration:
            return group, None
        if nxt.func_name != name:
            return group, nxt
        group.append(nxt)


def lockstep_records(
    per_csv_iters: List[Iterator[ParsedRecord]],
    wrappers: Optional[List[PositionTrackingWrapper]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    classifier: Optional[DuplicateNameClassifier] = None,
) -> Iterator[LockstepYield]:
    """Merge N per-CSV ParsedRecord iterators into a typed-sum stream.

    Inputs must be sorted by ``(func_name, occurrence)`` (every per-CSV
    CSV is already sorted; the memmap-output chain guarantees it).

    A canonical function name can map to MULTIPLE distinct functions
    inside one binary (per-TU static initializers, thunks, anon-namespace
    collisions) — the body-divergence deduper keeps each body and bumps
    the per-stream ``occurrence`` ordinal. Such a name is **duplicated**
    and CANNOT be matched: there is no single function for the matched
    arm to align variant-against-variant. This merge drains each stream's
    full consecutive same-name run, classifies the name (unique iff every
    contributing stream yielded exactly one row), and:

    * unique  ⇒ :class:`Matched` (≥ 2 streams) / :class:`Unmatched`
      (1 stream), exactly as before;
    * duplicated ⇒ every drained row down the :class:`Unmatched` arm,
      one item per row, so the unmatched grouper folds them into a single
      section's variant blocks instead of same-FID sibling sections.

    When ``classifier`` is supplied, each duplicated name is recorded in
    :attr:`DuplicateNameClassifier.duplicated_names` so pass 2 can stamp
    the unresolvable per-call sentinel on edges into those callees. The
    set is complete once the caller has fully drained this generator.
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

        # Drain the full consecutive same-name run from every matching
        # stream, re-seeding each stream's one-record lookahead with the
        # first row of its NEXT name. A name is DUPLICATED iff any single
        # stream contributed more than one row.
        groups: Dict[int, List[ParsedRecord]] = {}
        for i in matching_indices:
            group, current[i] = _drain_same_name_group(
                per_csv_iters[i], current[i]  # type: ignore[arg-type]
            )
            groups[i] = group

        is_duplicated = any(len(group) >= 2 for group in groups.values())

        if is_duplicated:
            if classifier is not None:
                classifier.duplicated_names.add(min_name)
            # Duplicated ⇒ unmatched no matter what: emit every body of
            # every contributing stream as its own Unmatched item.
            for i in matching_indices:
                for record in groups[i]:
                    yield Unmatched(
                        func_name=min_name,
                        record=record,
                        variant_index=i,
                    )
        elif len(matching_indices) >= 2:
            # Unique in every stream, present in ≥ 2 streams ⇒ matched.
            yield Matched(
                func_name=min_name,
                records={i: groups[i][0] for i in matching_indices},
            )
        else:
            (i,) = matching_indices
            yield Unmatched(
                func_name=min_name,
                record=groups[i][0],
                variant_index=i,
            )

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
    occurrence = _parse_occurrence(row, column_index)
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
        occurrence=occurrence,
        insn_runlength=insn_runlength,
        block_runlength=block_runlength,
        tokens=tokens,
        called_funcs=called,
        extern_libraries=extern_libraries,
        content_hash=content_hash,
    )


def _parse_occurrence(
    row: List[str],
    column_index: "dict[str, int]",
) -> int:
    """Read the integer ``occurrence`` ordinal for one CSV row.

    v2 carries a dedicated ``occurrence`` column (0 for the first body
    of a canonical name, 1+ for divergent same-name bodies). v1
    ``opaque_metadata`` rows predate the column entirely; they always
    represent the first-and-only body of their name and so default to 0.
    """
    idx = column_index.get("occurrence")
    if idx is None:
        return 0
    return int(row[idx])


def _decode_tokens(
    row: List[str],
    column_index: "dict[str, int]",
    mapping: Optional[np.ndarray],
) -> np.ndarray:
    tokens = base64_to_ndarray_vec(row[column_index["tokens_base64"]])
    if mapping is not None:
        tokens = mapping[tokens]
    return tokens.astype(np.uint16)


V2_CATEGORY_TYPES: "tuple[tuple[str, CallTargetType], ...]" = (
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
        return called_from_v2_metadata(row[column_index["metadata"]])
    if "opaque_metadata" in column_index:
        return _called_from_v1_opaque_metadata(row[column_index["opaque_metadata"]])
    return [], {}


def called_from_v2_metadata(
    metadata_cell: str,
) -> "tuple[list[tuple[str, CallTargetType]], dict[str, str]]":
    if not metadata_cell:
        return [], {}
    # Raise loud on malformed metadata (F-MED-11 / plan decision #5):
    # a corrupted CSV is a data-integrity violation that should crash
    # the consumer rather than silently emit zero callees + collapse
    # every EXTERN row to the same provider.
    meta = json.loads(metadata_cell)
    if not isinstance(meta, dict):
        return [], {}
    # Preserve the encoder's per-category allocation order: each
    # ``V2_CATEGORY_TYPES`` array is the CSV's identity-indexed
    # metadata cell, so the K-th name in category C is the function
    # whose encoder-assigned identity for C is K. Dedupe inside one
    # category via ``dict.fromkeys`` (order-preserving primitive); the
    # same name appearing in two categories (e.g. PLT ``foo`` + EXTERN
    # ``foo``) still surfaces as two distinct ``(name, type)`` entries.
    # Categories are concatenated in LOCAL -> PLT -> EXTERN order.
    called: list[tuple[str, CallTargetType]] = []
    extern_libraries: dict[str, str] = {}
    for category_key, category_type in V2_CATEGORY_TYPES:
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
