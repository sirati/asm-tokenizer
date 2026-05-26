"""Per-binary FTL discovery + parse-cache.

Single concern: own the per-binary state shared across every
:class:`FtlBackend` instance for that binary -- the surviving CSV
paths, the per-CSV :class:`VocabularyManager` cache, and the
content-hash-keyed :class:`ParsedRecord` cache produced by
:func:`lockstep_records` over those CSVs.

Plan v2 ``F-CRIT-1`` mandates :func:`lockstep_records` as the
discovery primitive (not a bespoke ``(name, occurrence)`` union scan).
``F-MED-9`` mandates :class:`CsvIndex` ownership of the vocab cache
and parsed-record cache: backends never reach into the loader; this
class is the single facade.

``F-MED-12`` filters empty CSVs at construction with a logged warning;
otherwise feeding an empty iterator into :func:`lockstep_records`
would underflow downstream lookups.

Plan v2 also locks-in (decision #25) per-binary vocab binding: each
CSV's :class:`VocabularyManager` is loaded lazily and cached for the
factory's lifetime; a per-binary FTL parse never crosses vocabs.
"""

from __future__ import annotations

import logging
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from tokenizer.aligned_data.parsed_record_iter import (
    LockstepYield,
    Matched,
    ParsedRecord,
    Unmatched,
    lockstep_records,
    open_parsed_record_iter,
)
from tokenizer.function_deduper import logical_function_name
from tokenizer.token_manager import VocabularyManager
from tokenizer.variant_info import VariantInfo
from tokenizer.vocab_unifier.loader import load_vocab_manager


__all__ = [
    "CsvIndex",
    "FunctionKey",
]


logger = logging.getLogger(__name__)


# A function-key identifies one logical-name group by ``(logical_name,
# group_content_hash)``. The logical name is the canonical name with
# the per-binary ``@thunk:<digits>`` suffix stripped via
# :func:`logical_function_name`, so PLT trampolines whose Ghidra-supplied
# resolved-extern offset varies across ELF builds still collapse into
# one inspector function-list entry. Two unrelated functions sharing a
# display name still appear as distinct handles via the disambiguating
# content hash.
FunctionKey = tuple[str, int]


def _peek_iterator(
    it: Iterator[ParsedRecord],
) -> tuple[Optional[ParsedRecord], Iterator[ParsedRecord]]:
    """Return ``(first_or_None, replayed_iterator)``.

    Pull one record to detect an empty CSV; if non-empty, prepend the
    pulled record back via :func:`itertools.chain` so the consumer
    sees the original stream.
    """
    try:
        first = next(it)
    except StopIteration:
        return None, iter(())
    return first, chain([first], it)


def _content_hash_for_record(record: LockstepYield) -> int:
    """Stable content-hash key for one lockstep-yield record.

    For :class:`Matched`, use the smallest-variant-index record's hash
    -- deterministic across runs since the lockstep yields are keyed
    by ``func_name`` and the variant order is fixed by the caller's
    iterator list.
    """
    if isinstance(record, Matched):
        first_key = min(record.records.keys())
        return record.records[first_key].content_hash
    if isinstance(record, Unmatched):
        return record.record.content_hash
    raise TypeError(f"unexpected LockstepYield arm: {type(record).__name__}")


class CsvIndex:
    """Per-binary discovery + vocab cache + parsed-record cache.

    Constructed once per binary by the factory (in the Wave-5
    ``_backend_factory`` package; this class is the boundary the
    factory and the per-function :class:`FtlBackend` instances both
    cross).

    Attributes:
        csv_dir: Root directory the caller asked us to discover from.
        binary_name: ``pkg`` filter -- only CSVs whose
            :func:`VariantInfo.from_csv` reports a matching ``pkg``
            survive.
        csv_paths: Surviving + sorted list of per-variant CSVs.
            Variant indices throughout the FtlBackend stack are
            positions in this list.
    """

    def __init__(self, csv_dir: Path, binary_name: str) -> None:
        self._csv_dir = csv_dir
        self._binary_name = binary_name
        # Per-CSV vocab + parsed-record caches.
        self._vocab_by_csv: Dict[Path, Optional[VocabularyManager]] = {}
        # Discovery: per-variant CSV walk + empty-CSV filter.
        self._csv_paths: List[Path] = _discover_csv_paths(csv_dir, binary_name)
        # Lazy: built on first ``function_keys()`` call. ``_records`` is
        # the raw stream from :func:`lockstep_records` (one entry per
        # canonical-name yield); ``_groups`` is the post-pass that
        # collapses canonical names sharing a logical name (see
        # :func:`logical_function_name`) into one inspector function-list
        # entry. ``handle.idx`` indexes into ``_groups``.
        self._records: Optional[List[LockstepYield]] = None
        self._groups: Optional[List[List[int]]] = None
        self._group_logical_names: Optional[List[str]] = None
        self._closed = False

    @property
    def csv_dir(self) -> Path:
        return self._csv_dir

    @property
    def binary_name(self) -> str:
        return self._binary_name

    @property
    def csv_paths(self) -> List[Path]:
        return self._csv_paths

    def function_keys(self) -> List[FunctionKey]:
        """Dense list of ``(logical_name, content_hash)`` keys for the binary.

        Streams :func:`lockstep_records` over the surviving CSVs once
        (cached for the index's lifetime), then groups same-logical-name
        canonical yields so PLT thunks whose ``@thunk:<offset>`` varies
        across ELF builds collapse into one inspector function. Position
        in the returned list is the canonical ``handle.idx`` for backend
        construction.
        """
        self._ensure_open()
        self._ensure_records_loaded()
        assert self._records is not None
        assert self._groups is not None
        assert self._group_logical_names is not None
        return [
            (
                self._group_logical_names[group_idx],
                _content_hash_for_group(self._records, member_indices),
            )
            for group_idx, member_indices in enumerate(self._groups)
        ]

    def parsed_record_for(
        self, idx: int, variant_idx: int
    ) -> Optional[ParsedRecord]:
        """Look up the parsed record for ``(handle.idx, variant_idx)``.

        Walks every canonical-name yield in the logical-name group at
        ``idx`` and returns the first record whose variant slot matches.
        Returns ``None`` when no yield in the group covers
        ``variant_idx`` -- the variant CSV either didn't include any
        same-logical-name function (group-wide Matched gap) or this
        single-variant function is reported only against a different
        slot.

        When more than one yield in the group has data for the same
        variant slot (e.g. a binary that carries both a local ``foo``
        AND a PLT thunk ``foo@thunk:N``), the first-yielded record wins
        and the rest are dropped with a logged warning. Per the task:
        the inspector function list collapses by logical name, which is
        a known information-loss trade-off in that rare edge case.
        """
        self._ensure_open()
        self._ensure_records_loaded()
        assert self._records is not None
        assert self._groups is not None
        first: Optional[ParsedRecord] = None
        for record_idx in self._groups[idx]:
            record = self._records[record_idx]
            parsed = _variant_lookup(record, variant_idx)
            if parsed is None:
                continue
            if first is None:
                first = parsed
                continue
            logger.warning(
                "logical-name group %r has multiple records in variant_idx=%d "
                "(canonical names collide post-suffix-strip); using the first "
                "(%r) and dropping %r",
                self._group_logical_names[idx]
                if self._group_logical_names is not None
                else "?",
                variant_idx,
                first.func_name,
                parsed.func_name,
            )
        return first

    def has_variant(self, idx: int, variant_idx: int) -> bool:
        """``True`` iff the function appears in this variant slot."""
        return self.parsed_record_for(idx, variant_idx) is not None

    def vocab_for(self, csv_path: Path) -> Optional[VocabularyManager]:
        """Lazy-load + cache the per-CSV :class:`VocabularyManager`."""
        self._ensure_open()
        if csv_path not in self._vocab_by_csv:
            self._vocab_by_csv[csv_path] = load_vocab_manager(csv_path)
        return self._vocab_by_csv[csv_path]

    def close(self) -> None:
        """Drop caches. Idempotent."""
        if self._closed:
            return
        self._records = None
        self._groups = None
        self._group_logical_names = None
        self._vocab_by_csv.clear()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CsvIndex closed")

    def _ensure_records_loaded(self) -> None:
        if self._records is not None:
            return
        # Open one iterator per surviving CSV. ``mapping=None`` per
        # plan decision 25 -- FTL backend renders in per-binary vocab
        # space, not unified.
        wrappers = []
        iters: List[Iterator[ParsedRecord]] = []
        try:
            for p in self._csv_paths:
                wrapper, it, _header = open_parsed_record_iter(str(p))
                wrappers.append(wrapper)
                iters.append(it)
            records = list(lockstep_records(iters))
        finally:
            for w in wrappers:
                try:
                    w.close()
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass
        self._records = records
        self._groups, self._group_logical_names = _build_logical_groups(records)


def _build_logical_groups(
    records: List[LockstepYield],
) -> "tuple[List[List[int]], List[str]]":
    """Collapse same-logical-name canonical yields into one group each.

    Preserves first-occurrence ordering so the inspector's function list
    keeps the lockstep-driven (alphabetical-ish) order rather than
    flipping to a Python-hash one. Returns ``(groups, logical_names)``
    where ``groups[k]`` is the list of indices into ``records`` whose
    canonical names map to ``logical_names[k]`` under
    :func:`logical_function_name`.
    """
    groups: List[List[int]] = []
    logical_names: List[str] = []
    name_to_group_idx: Dict[str, int] = {}
    for record_idx, record in enumerate(records):
        logical = logical_function_name(record.func_name)
        existing = name_to_group_idx.get(logical)
        if existing is None:
            name_to_group_idx[logical] = len(groups)
            groups.append([record_idx])
            logical_names.append(logical)
        else:
            groups[existing].append(record_idx)
    return groups, logical_names


def _variant_lookup(
    record: LockstepYield, variant_idx: int
) -> Optional[ParsedRecord]:
    """Per-(yield, variant_idx) record lookup. ``None`` on miss."""
    if isinstance(record, Matched):
        return record.records.get(variant_idx)
    if isinstance(record, Unmatched):
        if record.variant_index == variant_idx:
            return record.record
        return None
    raise TypeError(f"unexpected LockstepYield arm: {type(record).__name__}")


def _content_hash_for_group(
    records: List[LockstepYield], member_indices: List[int]
) -> int:
    """Stable content-hash for a logical-name group.

    The first member's :func:`_content_hash_for_record` is the group
    key (deterministic given the lockstep yield order is fixed). Two
    unrelated functions whose canonical names happen to share a logical
    name still appear as distinct handles because they land in distinct
    groups (each canonical-name lockstep yield is its own pre-grouping
    record); only same-logical-name yields collapse here.
    """
    first = records[member_indices[0]]
    return _content_hash_for_record(first)


def _discover_csv_paths(csv_dir: Path, binary_name: str) -> List[Path]:
    """Scan ``csv_dir`` for per-variant CSVs of ``binary_name``.

    Two layouts are accepted: flat (the canonical memmap-builder input
    layout, ``csv_dir/<variant>_output.csv``) and nested (the
    tokenize-worker output layout, ``csv_dir/<variant_dir>/<variant>_
    output.csv``). :func:`Path.rglob` covers both -- the per-CSV
    ``VariantInfo.from_csv(p).pkg`` filter keeps cross-binary CSVs
    out so descending into a non-binary subtree is safe.

    Empty CSVs (header-only, no function rows) are filtered with a
    warning per ``F-MED-12`` -- feeding an empty iterator into
    :func:`lockstep_records` underflows downstream lookups otherwise.
    """
    candidates = sorted(csv_dir.rglob("*_output.csv"))
    surviving: List[Path] = []
    for path in candidates:
        try:
            info = VariantInfo.from_csv(path)
        except ValueError as exc:
            logger.warning("skipping %s: VariantInfo.from_csv failed: %s", path, exc)
            continue
        if info.pkg != binary_name:
            continue
        if _csv_is_empty(path):
            logger.warning(
                "dropping empty CSV (no function rows): %s", path
            )
            continue
        surviving.append(path)
    return surviving


def _csv_is_empty(csv_path: Path) -> bool:
    """``True`` when the CSV yields zero :class:`ParsedRecord` rows.

    Peeks the iterator and discards the wrapper. The factory's
    ``_ensure_records_loaded`` re-opens the survivors fresh, so the
    peek-and-close pattern here doesn't leave dangling FDs.
    """
    wrapper, it, _header = open_parsed_record_iter(str(csv_path))
    try:
        first, _ = _peek_iterator(it)
        return first is None
    finally:
        try:
            wrapper.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


