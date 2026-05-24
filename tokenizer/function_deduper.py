"""Per-binary semantic-deduplication of functions emitted by the tokenizer.

Concern
-------
Decide whether a function about to be emitted is the SAME logical
function as one already emitted in this run. The decision is the
four-way AND of:

1. Same ``func_name`` — the obvious group key.
2. Same ``comment`` — the provider-supplied *context* string (typically
   the demangled C++ scoped signature, e.g.
   ``ARPHeader::reset(...)``). Two distinct C++ methods sharing an
   unqualified name (``ARPHeader::reset`` vs ``EthernetHeader::reset``)
   surface as same-``name`` Ghidra ``Function``s with *different*
   plate comments; the comment is the natural disambiguator.
3. Same ``identity_key`` — the provider-supplied "stronger-than-name"
   identity (today: the thunked-function entry-point offset for PLT
   thunks). ``None`` when the provider declines.
4. Same ``tokens_base64`` body — the emitted token-body string. When
   ``comment`` AND ``identity_key`` are both ``None`` we still merge
   same-named functions whose body matches (the LTO-clone case: same
   static helper inlined into many TUs surfaces with identical body
   under one shared name).

When all four match this is a duplicate emission and downstream
should fold it (skip the CSV row, skip the FunctionDataManager record,
do not increment the occurrence counter). When any axis differs this
is a fresh function and downstream emits a new row / allocates a new
slot.

The body-divergence warning case (same name + same comment + same
identity_key but different body) is a rare diagnostic surface: the
first-recorded body wins as the canonical for the identity tuple;
the second call is reported as ``body_divergence_warning=True`` and
is NOT folded (it becomes a fresh record that the caller may suffix
or warn-log as it sees fit).

Module boundary
---------------
This module owns ONE concern: the four-axis merge decision. It does
not know about Ghidra, about PLT thunks, about Capstone, the CSV
writer, or the FDM slot table. Providers compute ``comment`` +
``identity_key`` (they know what counts as same-logical-function in
their disassembly model). Callers ask :meth:`FunctionDeduper.resolve`
once per function and act on the returned :class:`DedupResolution`.

A folded duplicate's resolution carries the FIRST recorded call's
``slot_id``: an opaque token the caller can use to look up its own
side-table entries (FDM array index, CSV final-name, etc.). The
deduper itself never assigns names; it only allocates slot_ids and
tracks which body each slot's canonical entry recorded.

Producer pipelines (the tokenizer main loop, the FunctionDataManager
in the verification path) hold their own ``FunctionDeduper`` instance
per binary. The provider's ``iter_functions`` sort puts colliding
names consecutively, so the deduper's memory is tight in practice
(same-name entries arrive back-to-back), but the implementation keeps
no ordering assumption — the keyed map handles any iteration order
correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Tuple


@dataclass(frozen=True)
class DedupResolution:
    """Outcome of :meth:`FunctionDeduper.resolve` for one function.

    Fields:

    * ``slot_id`` — opaque per-binary identifier of the canonical record
      for this identity tuple. Two calls returning the same ``slot_id``
      refer to the same logical function (folded). Callers maintain
      their own ``slot_id -> downstream_state`` side-tables.
    * ``is_duplicate`` — ``True`` iff this call folded into a prior
      resolution with the same ``(name, comment, identity_key, body)``
      tuple. Callers that fold (skip the row, skip the FDM slot) gate
      on this flag.
    * ``body_divergence_warning`` — ``True`` iff this call has the
      same ``(name, comment, identity_key)`` as a prior call but a
      DIFFERENT body. The first-recorded body wins as canonical; this
      call is NOT folded (it spawns a fresh ``slot_id``). The flag is
      informational — callers that surface deduper diagnostics can
      log on it.
    """

    slot_id: int
    is_duplicate: bool
    body_divergence_warning: bool = False


class FunctionDeduper:
    """Per-binary stateful gate over the four-axis identity.

    Lifecycle: one instance per binary tokenization. ``resolve`` is
    called once per function in iteration order. The instance is not
    threadsafe (the tokenizer main loop is single-threaded).
    """

    __slots__ = ("_seen", "_next_slot_id")

    def __init__(self) -> None:
        # (name, comment, identity_key) -> (canonical_slot_id, body)
        # of the first accepted function with that identity tuple.
        # Subsequent matching-body calls fold into ``canonical_slot_id``
        # (return is_duplicate=True). Divergent-body calls surface a
        # warning and spawn a fresh ``slot_id`` (no fold).
        self._seen: Dict[
            Tuple[str, Optional[str], Optional[Hashable]],
            Tuple[int, str],
        ] = {}
        # Monotonic counter; slot_id allocation is the only ordering
        # signal the deduper exposes. Encounter-order across binaries
        # IS NOT a cross-ISA-stable property — callers that need a
        # cross-ISA-stable name should derive one from the SAME
        # ``(name, comment, identity_key)`` axes the deduper consumes
        # (the per-binary slot_id is correct ONLY within one binary).
        self._next_slot_id = 0

    def resolve(
        self,
        func_name: str,
        comment: Optional[str],
        identity_key: Optional[Hashable],
        tokens_base64: str,
    ) -> DedupResolution:
        """Resolve one function to its ``(slot_id, is_duplicate)`` pair.

        The four-way merge condition is documented at module level.
        Side effect on a first sighting: the identity tuple is recorded
        so a future matching call returns ``is_duplicate=True``.

        Args:
            func_name: The provider-reported function name.
            comment: The provider-reported plate comment (the C++
                scoped signature for C++ symbols; ``None`` otherwise).
                Empty strings are treated as ``None``.
            identity_key: The provider-reported stronger-than-name
                identity (e.g. thunked-offset for PLT thunks; ``None``
                otherwise).
            tokens_base64: The function's emitted token body, already
                serialised for the CSV row. Used verbatim as the body
                equality key.
        """
        # Normalise empty comment to None so callers can pass either
        # shape without semantic difference (the demangler may emit
        # ``""`` for some edge cases).
        if comment == "":
            comment = None

        key = (func_name, comment, identity_key)
        recorded = self._seen.get(key)
        if recorded is None:
            # First sighting under this identity tuple.
            slot_id = self._next_slot_id
            self._next_slot_id += 1
            self._seen[key] = (slot_id, tokens_base64)
            return DedupResolution(slot_id=slot_id, is_duplicate=False)

        recorded_slot, recorded_body = recorded
        if recorded_body == tokens_base64:
            # Four-way match — fold into the canonical record.
            return DedupResolution(slot_id=recorded_slot, is_duplicate=True)

        # Same identity tuple but DIFFERENT body — body divergence.
        # The first-recorded body wins as canonical; this call gets a
        # fresh slot_id but a warning flag the caller surfaces.
        # First-recorded body stays canonical for ANY future
        # identical-body fold attempt under this same tuple.
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        return DedupResolution(
            slot_id=slot_id,
            is_duplicate=False,
            body_divergence_warning=True,
        )


__all__ = ("DedupResolution", "FunctionDeduper")
