"""Address metadata lookup built from Ghidra's analysis results.

Owns ``GhidraMetadataLookup``: conforms to the ``MetadataLookup`` protocol
(``lookup(addr) -> AddressMetadataView``). Returns a typed
``_GhidraAddressMetadataView`` (defined in
``tokenizer.disasm.ghidra_provider.metadata_view``) reused across calls.

Slot-target resolution (vtable / code_ptr / jump-table slots) is bounded
by ``allow_slot_recursion`` per the task contract: outer ``lookup()``
mutates the per-lookup cursor view; the slot_target re-classification
builds a fresh standalone view with slot-detection signals suppressed.
"""

from __future__ import annotations

from typing import Any, Hashable, Optional

from tokenizer.disasm.ghidra_provider.metadata_view import _GhidraAddressMetadataView
from tokenizer.disasm.ghidra_provider.section_classify import (
    _encoding_from_string_datatype,
    _is_code_ptr_table_block_name,
    _is_plt_block_name,
    _is_rodata_block_name,
    _is_tls_block_name,
    _java_bytes_to_python,
    _looks_like_vtable_datatype_name,
    _looks_like_vtable_symbol_name,
    _section_type_from_block,
)
from tokenizer.disasm.ghidra_views.function import (
    _ghidra_function_comment,
    _ghidra_identity_key,
)
from tokenizer.disasm.metadata import (
    AddressKind,
    AddressMetadataView,
    address_kind_from_string,
    encoding_from_string,
    section_kind_from_type_string,
)
from tokenizer.function_deduper import canonical_function_name


class GhidraMetadataLookup:
    """Address metadata lookup built from Ghidra's analysis results.

    Conforms to the ``MetadataLookup`` protocol (``lookup(addr) ->
    AddressMetadataView``). The returned ``_GhidraAddressMetadataView`` is
    REUSED across ``lookup()`` calls (lifecycle per ``AddressMetadataView``
    docstring); valid only until the next ``lookup()``.

    Per ``lookup()`` the lookup walks Ghidra's symbol/function/memory
    state, derives every typed slot the view exposes, and stamps them in
    one ``_populate`` call. The string -> enum mapping happens here at
    population time, never on the read path.

    Typed slots populated per address:
        kind                -- ``AddressKind`` (precedence-aware: STRING >
                               vtable / code_ptr / jump_table-as-slot >
                               PLT > extern_synthetic > base type)
        section_kind        -- ``SectionKind`` derived from the containing
                               ``MemoryBlock``'s name + permission bits
        section_name        -- raw ELF section name (``.rodata`` etc.) of
                               the containing ``MemoryBlock`` (separate
                               from ``name``, which holds the function /
                               symbol label)
        name                -- function / symbol / section label
        start_addr/end_addr -- range bounds for the matched entity
        size                -- entity size in bytes
        library             -- non-None only for resolved extern targets
        string_encoding     -- ``Encoding`` enum from the Ghidra DataType
        string_bytes        -- raw string bytes when ``kind == STRING``
        is_vtable           -- C++ vtable slot (RTTI-analyzer output)
        tls                 -- TLS section (``.tdata`` / ``.tbss``)
        slot_target         -- lazily resolved on first read; uses
                               ``_slot_target_addr`` populated here
        jump_table_base_addr / jump_table_offset
                            -- populated for ``JUMP_TABLE_SLOT`` addresses

    Cross-provider parity: ``AngrMetadataLookup`` exposes the same typed
    slots; angr-only lookups leave Ghidra-only signals at their
    conservative defaults (``Encoding.UNKNOWN`` for non-ASCII strings,
    ``slot_target=None``, no jump-table info; see
    ``tokenizer/disasm/angr_limitations.md``).
    """

    def __init__(self, program: Any, function_manager: Any) -> None:
        self._program = program
        self._fm = function_manager
        self._memory = program.getMemory()
        self._symbol_table = program.getSymbolTable()
        self._listing = program.getListing()
        # Lazy switch-table cache: built on first JUMP_TABLE_SLOT slot_target
        # access. Maps table-base-addr -> list of resolved target block
        # addresses (slot order). Populated by walking every function once
        # via ``GhidraDisassemblyProvider.iter_switch_tables``.
        self._switch_table_cache: Optional[dict[int, list[int]]] = None
        # REUSED view wrapper - per-`lookup()` call we mutate its state
        # via `_populate` and return the same instance.
        self._view = _GhidraAddressMetadataView(self)

    # -- Enrichment helpers (one concern each) ------------------------------
    def _block_at(self, addr_obj: Any) -> Any:
        """Return the ``MemoryBlock`` containing ``addr_obj`` or None."""
        try:
            return self._memory.getBlock(addr_obj)
        except Exception:
            return None

    def _data_at(self, addr_obj: Any) -> Any:
        """Return the ``Data`` defined at ``addr_obj`` (exact start) or None.

        ``getDataAt`` returns None when the address is not the start of a
        Data unit; ``getDataContaining`` returns the Data whose body covers
        the address. We try exact first (matches v2's "addr is a slot start"
        semantics for jump tables / vtables); callers needing containment
        can ask separately.
        """
        try:
            return self._listing.getDataAt(addr_obj)
        except Exception:
            return None

    def _data_containing(self, addr_obj: Any) -> Any:
        """Return the ``Data`` whose body covers ``addr_obj`` or None."""
        try:
            return self._listing.getDataContaining(addr_obj)
        except Exception:
            return None

    def _classify_string(self, addr_obj: Any) -> tuple[bool, Optional[str], Optional[bytes]]:
        """Decide whether ``addr_obj`` starts a typed string Data.

        Returns ``(is_string, encoding, bytes)``. When the address is in
        the middle of a string Data (substring access) we still report
        ``is_string=True`` and return the *containing* string's bytes;
        the classifier consumes ``start_addr`` from the Data's min-address
        if it needs the offset.
        """
        data = self._data_at(addr_obj) or self._data_containing(addr_obj)
        if data is None:
            return False, None, None
        try:
            if not data.hasStringValue():
                return False, None, None
        except Exception:
            return False, None, None
        try:
            dt = data.getDataType()
        except Exception:
            dt = None
        encoding = _encoding_from_string_datatype(dt)
        try:
            raw = data.getBytes()
        except Exception:
            raw = None
        return True, encoding, _java_bytes_to_python(raw)

    def _is_vtable(self, addr_obj: Any, block: Any) -> bool:
        """Decide whether ``addr_obj`` is a C++ vtable slot.

        Two signals (either triggers): a symbol at the address whose
        name contains ``vftable``/``vtable``, OR a Data object whose
        DataType name contains the same substring. Only meaningful for
        rodata-flavored sections (``.rodata`` / ``.data.rel.ro``).
        """
        if block is None:
            return False
        if not _is_rodata_block_name(str(block.getName())):
            return False
        # Symbol-name signal — exact match at this address.
        try:
            symbols = self._symbol_table.getSymbols(addr_obj)
        except Exception:
            symbols = ()
        for sym in symbols or ():
            try:
                if _looks_like_vtable_symbol_name(str(sym.getName())):
                    return True
            except Exception:
                continue
        # DataType-name signal — Ghidra's GccRttiAnalyzer stamps the Data.
        data = self._data_containing(addr_obj)
        if data is not None:
            try:
                dt = data.getDataType()
                if dt is not None and _looks_like_vtable_datatype_name(str(dt.getName())):
                    return True
            except Exception:
                pass
        return False

    def _is_jump_table_slot(self, addr_obj: Any, block: Any) -> bool:
        """Decide whether ``addr_obj`` is a slot in a Ghidra-recovered switch table.

        First-pass implementation: check that the Data at/around the
        address is a pointer-array element AND the containing block
        is rodata-flavored. A precise cross-check against computed-jump
        inbound references is feasible but expensive (per-address ref
        walk); the function-level ``iter_switch_tables`` consumer below
        does that more cheaply by walking from the dispatch site.
        """
        if block is None:
            return False
        if not _is_rodata_block_name(str(block.getName())):
            return False
        data = self._data_at(addr_obj) or self._data_containing(addr_obj)
        if data is None:
            return False
        try:
            dt = data.getDataType()
        except Exception:
            return False
        if dt is None:
            return False
        # Pointer or Array-of-Pointer is the canonical jump-table shape.
        # We check by class name to avoid a hard import of Ghidra DataType
        # classes here; the type hierarchy in Ghidra has ``Pointer`` and
        # ``Array`` as stable interface names.
        try:
            from ghidra.program.model.data import Array, Pointer

            if isinstance(dt, Pointer):
                return True
            if isinstance(dt, Array):
                try:
                    inner = dt.getDataType()
                    return isinstance(inner, Pointer)
                except Exception:
                    return False
        except Exception:
            # Fallback if the Ghidra import fails for any reason
            tname = str(dt.getName()).lower()
            if "pointer" in tname or tname.endswith("*"):
                return True
        return False

    def _is_extern_synthetic(self, func: Any, block: Any) -> bool:
        """Detect Ghidra's analogue of CLE's synthetic-extern object.

        Ghidra represents unresolved imports through an "EXTERNAL" memory
        block (``isExternalBlock``) plus the Function's ``isExternal()``
        flag. When the address lives in that block, the v2 classifier
        treats it the same way CLE's externs are treated.
        """
        if block is not None:
            try:
                if block.isExternalBlock():
                    return True
            except Exception:
                pass
            try:
                # Older Ghidra exposes the same concept through the block name.
                if str(block.getName()).upper() == "EXTERNAL":
                    return True
            except Exception:
                pass
        if func is not None:
            try:
                if func.isExternal():
                    return True
            except Exception:
                pass
        return False

    # -- Public API ---------------------------------------------------------
    def lookup(self, addr: int) -> AddressMetadataView:
        """Resolve ``addr`` to the typed cursor view.

        Mutates the per-lookup ``_GhidraAddressMetadataView`` and returns
        it. Consumers read typed properties exclusively (``meta.kind``,
        ``meta.name``, ``meta.string_encoding`` etc.).
        """
        return self._classify_address(addr, allow_slot_recursion=True)

    def _classify_address(
        self,
        addr: int,
        *,
        allow_slot_recursion: bool,
    ) -> AddressMetadataView:
        """Classify ``addr`` and return a typed view.

        ``allow_slot_recursion`` controls the active wrapper AND the
        slot-detection branch:

        - ``True`` (public ``lookup()`` path): mutate the cursor view
          (``self._view``) in place and return it. Slot-target resolution
          is permitted - the resulting view's ``slot_target`` property
          will re-enter ``_classify_address`` with
          ``allow_slot_recursion=False`` so the inner call writes to a
          fresh wrapper instead of disturbing the cursor.
        - ``False`` (slot-target resolution path): build a FRESH
          standalone view bound to this lookup. Slot-detection signals
          are suppressed so the inner view's kind is guaranteed NOT to
          be a slot kind (CODE_PTR_TABLE_SLOT / JUMP_TABLE_SLOT) per
          task contract; ``slot_target`` is also guaranteed ``None`` so
          there is no infinite regress.
        """
        addr_obj = self._program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)
        block = self._block_at(addr_obj)
        block_name = str(block.getName()) if block is not None else ""

        # Enrichments computed once; reused across all classification branches.
        is_plt = _is_plt_block_name(block_name) if block is not None else False
        is_code_ptr_table_slot = _is_code_ptr_table_block_name(block_name) if block is not None else False
        tls = _is_tls_block_name(block_name) if block is not None else False
        is_string, string_encoding_str, string_bytes = self._classify_string(addr_obj)
        # Slot-detection signals are suppressed when re-entering for slot
        # resolution: the slot_target's kind must NOT itself be a slot kind.
        if allow_slot_recursion:
            is_vtable = self._is_vtable(addr_obj, block)
            is_jump_table_slot = self._is_jump_table_slot(addr_obj, block)
        else:
            is_vtable = False
            is_jump_table_slot = False
            is_code_ptr_table_slot = False

        # Per-branch raw fields (name / type-string / size / range bounds /
        # func ref). Branches set them; the final populate-typed-state step
        # at the bottom translates string -> enum once.
        type_str: str
        name: Optional[str]
        size: Optional[int]
        start_addr: Optional[int]
        end_addr: Optional[int]
        func: Any = None
        section_name: Optional[str] = block_name if block is not None else None

        # 1. Exact symbol match -- legacy bare ``"symbol"`` is replaced with
        #    a section-derived type. The symbol's name is preserved as-is.
        symbols = self._symbol_table.getSymbols(addr_obj)
        if symbols:
            sym = symbols[0]
            name = str(sym.getName())
            # Derive type from the containing section (if any). For a symbol
            # in ``.text`` whose Ghidra ``SymbolType`` is ``FUNCTION`` we
            # tag it ``local_function``; everything else uses the section's
            # natural type (``rodata`` / ``data`` / ``thread_local_data`` / ...).
            type_str = "unknown"
            if block is not None:
                base_type = _section_type_from_block(block)
                if base_type == "code":
                    is_function_symbol = False
                    try:
                        st = sym.getSymbolType()
                        # ``SymbolType.FUNCTION`` is the canonical enum value;
                        # compare by name to avoid importing the enum class.
                        is_function_symbol = str(st).upper() == "FUNCTION"
                    except Exception:
                        is_function_symbol = False
                    type_str = "local_function" if is_function_symbol else base_type
                else:
                    type_str = base_type
            size = 0
            start_addr = addr
            end_addr = addr
            # If the symbol is an actual function and we have its body, give
            # the classifier accurate range bounds rather than a zero-width.
            try:
                func = self._fm.getFunctionAt(addr_obj)
            except Exception:
                func = None
            if func is not None:
                try:
                    body = func.getBody()
                    size = int(body.getNumAddresses())
                    entry = int(func.getEntryPoint().getOffset())
                    start_addr = entry
                    end_addr = entry + size
                except Exception:
                    pass
        else:
            # 2. Function match (covers calls/jumps into known functions).
            func = self._fm.getFunctionContaining(addr_obj)
            if func is not None:
                entry = int(func.getEntryPoint().getOffset())
                body = func.getBody()
                size = int(body.getNumAddresses())
                is_external = func.isExternal() or func.isThunk()
                name = str(func.getName())
                type_str = "library_function" if is_external else "local_function"
                start_addr = entry
                end_addr = entry + size
            elif block is not None:
                # 3. Memory-block match -- the address is in a known section
                # but not inside any function. Use the section-type mapping.
                name = block_name
                type_str = _section_type_from_block(block)
                size = int(block.getSize())
                start_addr = int(block.getStart().getOffset())
                end_addr = int(block.getEnd().getOffset()) + 1
            else:
                # 4. Fallback -- address not in any known memory region.
                name = f"unknown_{addr:x}"
                type_str = "unknown"
                size = 0
                start_addr = addr
                end_addr = addr

        # Final enrichments that depend on ``func`` / ``block`` and apply
        # uniformly across all four branches above.
        is_thunk = func is not None and bool(getattr(func, "isThunk", lambda: False)())
        is_plt_combined = is_plt or is_thunk
        is_extern_synthetic = self._is_extern_synthetic(func, block)

        # Canonical-name derivation: pull the two additional identity axes
        # (``comment`` from the demangler, ``identity_key`` from the
        # thunk-target offset) off the resolved Function handle when we
        # have one, then collapse (raw_name, comment, identity_key) into
        # a cross-ISA-stable canonical name via the same helper the
        # FunctionDataManager uses. The two axes are also persisted on
        # the typed view so consumers (or audits) can re-derive or
        # inspect the canonical name. For non-function addresses (block
        # / fallback branches) ``func`` is None; both axes stay None and
        # ``canonical_function_name`` short-circuits to the raw name.
        comment: Optional[str] = (
            _ghidra_function_comment(func) if func is not None else None
        )
        identity_key: Optional[Hashable] = (
            _ghidra_identity_key(func) if func is not None else None
        )
        if name is not None:
            name = canonical_function_name(name, comment, identity_key)

        # Translate raw signals into typed slots ONCE here (no per-property
        # work on the read path).
        kind = self._derive_address_kind(
            type_str=type_str,
            is_string=is_string,
            is_vtable=is_vtable,
            is_jump_table_slot=is_jump_table_slot,
            is_code_ptr_table_slot=is_code_ptr_table_slot,
            is_plt=is_plt_combined,
            is_extern_synthetic=is_extern_synthetic,
        )
        section_kind = section_kind_from_type_string(type_str)
        string_encoding = encoding_from_string(string_encoding_str)

        # Slot-target / jump-table resolution. Only meaningful on the
        # public lookup path (``allow_slot_recursion=True``); the inner
        # slot_target re-classification path leaves these at None per
        # the task contract.
        slot_target_addr: Optional[int] = None
        jump_table_base_addr: Optional[int] = None
        jump_table_offset: Optional[int] = None
        if allow_slot_recursion:
            if is_vtable or is_code_ptr_table_slot:
                target = self._resolve_pointer_array_slot(addr)
                if target is not None:
                    slot_target_addr = target
            if is_jump_table_slot:
                jt_base = self._jump_table_base_addr(addr)
                if jt_base is not None:
                    jump_table_base_addr = jt_base
                    jump_table_offset = addr - jt_base
                    target = self._jump_table_target(jt_base, addr - jt_base)
                    if target is not None:
                        slot_target_addr = target

        # Library label: index entries currently leave it as the
        # "unknown" sentinel; surface None on the typed view so downstream
        # readers don't accidentally consume the placeholder string.
        library: Optional[str] = None

        view = self._view if allow_slot_recursion else _GhidraAddressMetadataView(self)
        view._populate(
            kind=kind,
            section_kind=section_kind,
            section_name=section_name,
            string_encoding=string_encoding,
            string_bytes=string_bytes,
            name=name,
            comment=comment,
            identity_key=identity_key,
            start_addr=start_addr,
            end_addr=end_addr,
            size=size,
            library=library,
            is_vtable=bool(is_vtable),
            tls=bool(tls),
            slot_target_addr=slot_target_addr,
            jump_table_base_addr=jump_table_base_addr,
            jump_table_offset=jump_table_offset,
        )
        return view

    @staticmethod
    def _derive_address_kind(
        *,
        type_str: str,
        is_string: bool,
        is_vtable: bool,
        is_jump_table_slot: bool,
        is_code_ptr_table_slot: bool,
        is_plt: bool,
        is_extern_synthetic: bool,
    ) -> AddressKind:
        """Combine raw signals into the precedence-aware ``AddressKind``.

        Order matches ``constant_handler._PRECEDENCE``:
            1. is_string                      -> STRING
            2. is_jump_table_slot             -> JUMP_TABLE_SLOT
            3. is_vtable / is_code_ptr_slot   -> CODE_PTR_TABLE_SLOT
            4. is_plt                         -> PLT_FUNCTION
            5. is_extern_synthetic on a func  -> EXT_FUNCTION_SYNTHETIC
            6. base type-string               -> base AddressKind
        """
        if is_string:
            return AddressKind.STRING
        if is_jump_table_slot:
            return AddressKind.JUMP_TABLE_SLOT
        if is_vtable or is_code_ptr_table_slot:
            return AddressKind.CODE_PTR_TABLE_SLOT
        if is_plt:
            return AddressKind.PLT_FUNCTION
        lowered = (type_str or "").lower()
        if is_extern_synthetic and lowered in {
            "extern_function",
            "library_function",
            "plt_function",
            "unknown_function",
        }:
            return AddressKind.EXT_FUNCTION_SYNTHETIC
        return address_kind_from_string(type_str)

    def _resolve_pointer_array_slot(self, addr: int) -> Optional[int]:
        """Read the pointer value stored at ``addr``.

        For init_array / fini_array / dtors / vtable slots, Ghidra has a
        typed ``Pointer`` ``Data`` whose ``getValue()`` returns the
        ``Address`` object pointing at the resolved target. Returns the
        target offset as a Python int, or ``None`` when no such Data is
        defined (the classifier then leaves ``slot_target`` empty and
        the consumer falls back to the bare ptr token).
        """
        try:
            addr_obj = self._program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)
            data = self._listing.getDataAt(addr_obj)
            if data is None:
                return None
            value = data.getValue()
            if value is None:
                return None
            # ``Address`` exposes ``getOffset()``; ``Scalar`` exposes
            # ``getValue()``. Both are plausible Pointer payloads.
            if hasattr(value, "getOffset"):
                return int(value.getOffset())
            if hasattr(value, "getValue"):
                return int(value.getValue())
            return None
        except Exception:
            return None

    def _jump_table_base_addr(self, addr: int) -> Optional[int]:
        """Return the start address of the ``Data`` array containing ``addr``.

        For a jump-table slot, the containing ``Data`` (an Array of
        Pointer) gives the table's base via ``getMinAddress()``.
        Returns ``None`` if no such containing Data exists.
        """
        try:
            addr_obj = self._program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)
            data = self._listing.getDataContaining(addr_obj)
            if data is None:
                return None
            min_addr = data.getMinAddress()
            if min_addr is None:
                return None
            return int(min_addr.getOffset())
        except Exception:
            return None

    def _jump_table_target(self, base_addr: int, offset: int) -> Optional[int]:
        """Look up the resolved target for a jump-table slot.

        Builds the switch-table cache lazily on first call: walks every
        function once via ``iter_switch_tables`` and indexes
        ``{table_base_addr -> [target_addrs in slot order]}``. Slot
        index is computed from ``offset / sizeof(pointer)``; for the
        common case the pointer width is 8 (LP64) or 4 (ILP32), but we
        derive it from the count of recovered targets vs. the table's
        Data length when available.
        """
        cache = self._ensure_switch_table_cache()
        targets = cache.get(int(base_addr))
        if not targets:
            return None
        # Slot index. The table is an array of pointers; the offset is in
        # bytes. We have to derive pointer size from the program; fall
        # back to dividing by 8 if no specific size is available.
        try:
            ptr_size = int(self._program.getDefaultPointerSize())
        except Exception:
            ptr_size = 8
        if ptr_size <= 0:
            return None
        idx = offset // ptr_size
        if 0 <= idx < len(targets):
            return targets[idx]
        return None

    def _ensure_switch_table_cache(self) -> dict[int, list[int]]:
        """Build (once) the ``{table_base -> [targets]}`` cache.

        Walks every function in the program via ``iter_switch_tables``
        and merges the results. Idempotent; subsequent calls return the
        cached dict directly. Cost is paid on first JUMP_TABLE_SLOT
        slot_target access; pure-data lookups never trigger this.
        """
        if self._switch_table_cache is not None:
            return self._switch_table_cache
        cache: dict[int, list[int]] = {}
        # ``iter_switch_tables`` lives on the provider; we don't have a
        # back-ref here, so re-implement the per-function walk inline.
        # Avoids coupling MetadataLookup to GhidraDisassemblyProvider.
        try:
            from ghidra.program.model.symbol import RefType  # noqa: F401
        except Exception:
            self._switch_table_cache = cache
            return cache
        try:
            funcs = list(self._fm.getFunctions(True))
        except Exception:
            funcs = []
        for func in funcs:
            try:
                body = func.getBody()
                insn_iter = self._listing.getInstructions(body, True)
            except Exception:
                continue
            while True:
                try:
                    if not insn_iter.hasNext():
                        break
                    insn = insn_iter.next()
                except Exception:
                    break
                try:
                    flow_type = insn.getFlowType()
                    if not (flow_type.isJump() and flow_type.isComputed()):
                        continue
                except Exception:
                    continue
                table_addr: Optional[int] = None
                targets: list[int] = []
                try:
                    refs_from = list(insn.getReferencesFrom() or ())
                except Exception:
                    refs_from = []
                for ref in refs_from:
                    try:
                        rtype = ref.getReferenceType()
                    except Exception:
                        continue
                    try:
                        if rtype.isData() and rtype.isRead():
                            if table_addr is None:
                                table_addr = int(ref.getToAddress().getOffset())
                            continue
                    except Exception:
                        pass
                    try:
                        if rtype.isJump() and rtype.isComputed():
                            targets.append(int(ref.getToAddress().getOffset()))
                    except Exception:
                        pass
                if table_addr is not None and targets:
                    cache.setdefault(table_addr, list(targets))
        self._switch_table_cache = cache
        return cache
