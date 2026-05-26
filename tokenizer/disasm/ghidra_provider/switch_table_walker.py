"""Per-function computed-jump dispatch walk for switch-table recovery.

Concern: walking ONE Ghidra ``Function``'s body and yielding the
``(table_base_addr, [target_block_addrs])`` tuples for each computed-jump
dispatch instruction in it. Shared by two callers that previously
re-implemented this walk inline:

- ``GhidraDisassemblyProvider.iter_switch_tables`` (entry-point lookup +
  per-function delegate); the public, per-function API consumed by
  ``_emit_jump_table_footer``.
- ``GhidraMetadataLookup._ensure_switch_table_cache`` (walks every
  function and accumulates ``{table_base -> [targets]}``); built lazily
  for the ``JUMP_TABLE_SLOT`` slot_target resolution path.

Boundary
--------

API surface:

    walk_switch_tables_for_function(
        ghidra_function, listing, *, skip_thunks=True,
    ) -> Iterable[tuple[int, list[int]]]

Callers pass a Ghidra ``Function`` handle + the program's ``Listing``;
they receive an iterator of ``(table_addr, list_of_targets)``. The
helper owns:

- The thunk gate (skipping PLT trampolines whose reference graph
  structurally mimics a 1-target switch dispatch).
- Per-instruction dispatch filtering via ``FlowType.isJump() and
  FlowType.isComputed()`` (``FlowType`` is structurally distinct from
  ``RefType``; the helper inlines this check because the
  ``RefType``-only predicate in ``jump_table_predicates`` does NOT
  apply).
- Outbound-reference classification using
  ``is_computed_jump_reftype`` for target refs and a ``isData()+isRead()``
  test for the table-base READ ref.
- Operand-object fallback when no READ ref exists (some
  architectures encode the table base as a directly-typed Address
  operand rather than as an outbound reference).

It deliberately does NOT know:

- How callers resolve a ``FunctionView`` / address to a Ghidra
  ``Function`` (that's the provider's ``_funcs_by_entry`` cache).
- How callers aggregate or de-duplicate the yields (the provider passes
  them straight through; the metadata-lookup cache builder
  ``cache.setdefault``s them).
- The thunk-canonical-naming concern (lives in
  ``ghidra_views.unnamed_rename``).

Drift consolidation
-------------------

The two pre-existing inline copies had diverged:

- Provider's walk used both the operand-object fallback for table_addr
  AND the legacy ``RefType.COMPUTED_JUMP`` direct-equality fallback for
  target refs.
- Metadata-lookup's cache walk lacked both.

This helper adopts the FULLER recovery set (provider's superset), so the
cache builder now catches everything the canonical walk catches. Strictly
additive: no previously-recovered table is dropped; some previously-missed
tables (Ghidra-legacy + no-READ-ref operand-encoded) now feed the
``JUMP_TABLE_SLOT`` slot-resolution path.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from tokenizer.disasm.ghidra_provider.jump_table_predicates import (
    is_computed_jump_reftype,
)


def walk_switch_tables_for_function(
    ghidra_function: Any,
    listing: Any,
    *,
    skip_thunks: bool = True,
) -> Iterable[tuple[int, list[int]]]:
    """Yield ``(table_addr, [target_addrs])`` for each computed-jump
    dispatch in ``ghidra_function``'s body.

    When ``skip_thunks`` is True (default), functions that report
    ``isThunk() == True`` produce no yields. PLT trampolines look
    structurally identical to 1-target switch dispatches in Ghidra's
    reference graph (DATA-READ to the GOT slot + COMPUTED_JUMP to PLT0);
    skipping them is the authoritative way to avoid emitting bogus
    1-target "tables" for every thunk in the binary.

    The walk is defensive against malformed Ghidra handles: any
    individual ``Function`` / ``Instruction`` / ``Reference`` whose
    accessors raise is silently skipped, never propagating upstream and
    corrupting the per-function iteration loop.
    """
    if ghidra_function is None or listing is None:
        return
    if skip_thunks:
        try:
            if ghidra_function.isThunk():
                # PLT trampolines emit ``ldr pc, [GOT_slot]`` / ``jmp [GOT_slot]``
                # whose Ghidra reference graph is structurally identical to a
                # 1-target switch-table dispatch (computed-jump flow + DATA-READ
                # to the GOT slot + COMPUTED_JUMP to PLT0). Thunks don't carry
                # switch tables; their indirect-call semantics surface via the
                # thunk-target path, so skip them here.
                return
        except Exception:
            # Defensive: malformed Function handle. Treat as "no thunk info"
            # rather than crashing the iter loop.
            pass

    try:
        body = ghidra_function.getBody()
        insn_iter = listing.getInstructions(body, True)
    except Exception:
        return

    while True:
        try:
            if not insn_iter.hasNext():
                return
            insn = insn_iter.next()
        except Exception:
            return
        try:
            flow_type = insn.getFlowType()
            # ``FlowType`` predicate is structurally distinct from
            # the ``RefType`` predicate folded into
            # ``is_computed_jump_reftype`` — same method names, different
            # class hierarchy. Keep inlined here.
            if not (flow_type.isJump() and flow_type.isComputed()):
                continue
        except Exception:
            continue

        # Outbound references from the dispatch instruction. Computed
        # jumps surface their resolved targets as COMPUTED_JUMP refs.
        try:
            refs_from = list(insn.getReferencesFrom() or ())
        except Exception:
            refs_from = []

        table_addr: Optional[int] = None
        targets: list[int] = []
        for ref in refs_from:
            try:
                rtype = ref.getReferenceType()
            except Exception:
                continue
            # READ references typically point at the table base in rodata.
            try:
                if rtype.isData() and rtype.isRead():
                    to_addr = int(ref.getToAddress().getOffset())
                    if table_addr is None:
                        table_addr = to_addr
                    continue
            except Exception:
                pass
            # COMPUTED_JUMP / CONDITIONAL_COMPUTED_JUMP / COMPUTED_CALL
            # references list the resolved target blocks. The helper
            # absorbs the modern-predicate + legacy-equality fallback.
            if is_computed_jump_reftype(rtype):
                try:
                    to_addr = int(ref.getToAddress().getOffset())
                except Exception:
                    continue
                targets.append(to_addr)

        if not targets:
            continue

        # If we did not find a READ reference, fall back to scanning
        # the instruction's data-typed operand for a memory-base address.
        if table_addr is None:
            try:
                num_ops = insn.getNumOperands()
            except Exception:
                num_ops = 0
            for i in range(num_ops):
                try:
                    for obj in insn.getOpObjects(i) or ():
                        # ghidra.program.model.address.Address has getOffset()
                        if hasattr(obj, "getOffset"):
                            table_addr = int(obj.getOffset())
                            break
                except Exception:
                    continue
                if table_addr is not None:
                    break

        if table_addr is None:
            # No locatable table base; skip rather than emit a synthetic.
            continue

        yield table_addr, targets
