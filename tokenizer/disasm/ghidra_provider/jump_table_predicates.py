"""Semantic predicates that decide whether a Ghidra ``Data`` is a switch table.

Concern: structural Pointer-in-rodata-block is necessary but not sufficient
evidence that a piece of Data is a Ghidra-recovered switch table. Hardened
ELF builds (``-z relro -z now``) place the PLT GOT analogue in
``.data.rel.ro``; C++ RTTI puts typeinfo and vtable component pointers in
the same rodata-flavored sections. All trivially satisfy the structural
check.

The authoritative signal is the inverse: did Ghidra's analyser record an
inbound ``COMPUTED_JUMP`` reference into the Data's address range? If yes,
the Data is the target of a computed-jump dispatch site, which is the
definition of a switch table. If no, the structural match is coincidental
and must NOT be classified as ``JUMP_TABLE_SLOT``.

This module owns ONLY the back-reference walk; the predicate caller
(``GhidraMetadataLookup._is_jump_table_slot``) keeps ownership of the
structural check + the rodata-block-name gate. API surface:

    has_inbound_computed_jump(reference_manager, data) -> bool

Callers learn nothing about the ReferenceManager iterator protocol, the
RefType import path, or the address-range walk; they pass the resolved
ReferenceManager + Data and receive a boolean.
"""

from __future__ import annotations

from typing import Any


def has_inbound_computed_jump(reference_manager: Any, data: Any) -> bool:
    """Return True iff any address inside ``data``'s span has an inbound
    ``COMPUTED_JUMP`` reference.

    Walks every destination address in ``[data.getMinAddress(),
    data.getMaxAddress()]`` reachable via the ReferenceManager's destination
    iterator; for each, asks ``getReferencesTo`` and inspects the
    ``RefType``. A back-ref qualifies when ``isJump() and isComputed()``;
    we also accept an exact ``RefType.COMPUTED_JUMP`` identity match as a
    defensive fallback for older Ghidra builds where the predicate methods
    might be absent.

    For Array-of-Pointer Data, an inbound back-ref into ANY slot of the
    array qualifies the whole table — the dispatch site needs only to
    reference one slot (the first; subsequent slots are addressed by
    offset) for Ghidra to record the computed-jump back-ref.

    All Ghidra calls are wrapped in defensive ``try/except`` so transient
    JPype errors degrade to "no evidence" (False) rather than propagating
    upstream and corrupting the per-instruction tokenization loop.
    """
    if reference_manager is None or data is None:
        return False
    try:
        from ghidra.program.model.symbol import RefType
    except Exception:
        RefType = None  # noqa: N806 — best-effort fallback below
    try:
        min_addr = data.getMinAddress()
        max_addr = data.getMaxAddress()
    except Exception:
        return False
    if min_addr is None or max_addr is None:
        return False
    try:
        addr_iter = reference_manager.getReferenceDestinationIterator(min_addr, True)
    except Exception:
        return False
    while True:
        try:
            has_next = addr_iter.hasNext()
        except Exception:
            return False
        if not has_next:
            return False
        try:
            dest_addr = addr_iter.next()
        except Exception:
            return False
        # Iterator walks ALL ref-destination addresses from min_addr upward;
        # stop when we cross past max_addr so we don't scan the rest of
        # the program.
        try:
            if dest_addr.compareTo(max_addr) > 0:
                return False
        except Exception:
            return False
        try:
            refs_to = reference_manager.getReferencesTo(dest_addr)
        except Exception:
            continue
        for ref in refs_to or ():
            try:
                rt = ref.getReferenceType()
            except Exception:
                continue
            if rt is None:
                continue
            try:
                if rt.isJump() and rt.isComputed():
                    return True
            except Exception:
                pass
            if RefType is not None:
                try:
                    if rt == RefType.COMPUTED_JUMP:
                        return True
                except Exception:
                    pass
    # Unreachable; the loop exits via explicit returns above.
