import logging
import re
from typing import Optional

import angr
import cle
from intervaltree import IntervalTree

from tokenizer.disasm.angr_provider import _AngrAddressMetadataView
from tokenizer.disasm.metadata import (
    AddressKind,
    AddressMetadataView,
    Encoding,
    SectionKind,
    address_kind_from_string,
    encoding_from_string,
    section_kind_from_type_string,
)
from tokenizer.function_deduper import canonical_function_name

# ASCII printable run of length >=4 terminated by NUL byte.
# Mirrors the heuristic in tokenizer/disasm/angr_provider.py:parse_data_sections;
# v2's string sidecar is built there, but the metadata-lookup also needs
# per-address `is_string` membership so the classifier can emit `string_ptr`
# at precedence step 7 on the angr path. UTF-16 / Pascal / non-ASCII strings
# are out of scope here per `tokenizer/disasm/angr_limitations.md`.
_ASCII_STRING_RE = re.compile(rb"[\x20-\x7e]{4,}\x00")


class AngrMetadataLookup:
    def __init__(self, path):
        self.project = angr.Project(path, auto_load_libs=True)
        # Define code-related sections to include in CFG
        self._code_regions = []
        for section in self.project.loader.main_object.sections:
            if section.is_executable and not section.name.startswith(".plt") and not section.name.startswith(".got"):
                self._code_regions.append((section.vaddr, section.vaddr + section.memsize))
        self.cfg = self.project.analyses.CFGFast(normalize=True, regions=self._code_regions)
        self.exact_lookup = {}
        self.range_lookup = IntervalTree()
        self.library_ranges = self._build_library_ranges()
        self._build_indices()
        # REUSED view wrapper - per-`lookup()` call we mutate its slots
        # via `_populate` and return the same instance. See
        # `AddressMetadataView` docstring for the lifecycle contract.
        self._view = _AngrAddressMetadataView()

    def _build_library_ranges(self):
        """
        Builds a list of tuples: (start_addr, end_addr, library_name)
        for all loaded binaries (main executable + libraries).
        """
        lib_ranges = []
        for binary in self.project.loader.all_objects:
            start = binary.min_addr
            end = binary.max_addr + 1  # exclusive end
            lib_name = getattr(binary, "provides", None)
            if lib_name is None:
                # fallback to filename or unknown
                lib_name = getattr(binary, "filename", None) or "unknown"
            lib_ranges.append((start, end, lib_name))
        return lib_ranges

    def _find_library_for_addr(self, addr):
        for start, end, lib_name in self.library_ranges:
            if start <= addr < end:
                return lib_name
        return "unknown"

    def _get_section_type(self, section):
        name = section.name or ""
        if name in {".init", ".fini", ".plt"}:
            return "code"
        elif name == ".bss" or name == ".tbss":
            # TLS BSS keeps the bss base type; TLS-ness is signalled
            # orthogonally via the `tls` meta flag (precedence step 10).
            return "bss"
        elif name == ".tdata":
            return "data"
        elif section.is_executable:
            return "code"
        elif section.is_writable:
            return "data"
        else:
            return "rodata"

    def _section_is_tls(self, section) -> bool:
        """ELF TLS sections (`.tdata`, `.tbss`) host thread-local storage.

        The classifier (precedence step 10) prepends a `thread_local` modifier
        before `rw_data_ptr` whenever this flag is set on the looked-up meta.
        We detect TLS by section name; CLE does not expose SHF_TLS as a
        per-section attribute on every backend.
        """
        return (section.name or "") in {".tdata", ".tbss"}

    def _find_section_for_addr(self, addr):
        """Return the main-object section containing `addr`, or None."""
        for section in self.project.loader.main_object.sections:
            if section.memsize == 0:
                continue
            if section.vaddr <= addr < section.vaddr + section.memsize:
                return section
        return None

    def _build_symbol_meta(self, sym, sym_type):
        """Build an exact-lookup meta dict for a non-function ELF symbol.

        `sym_type` is the CLE `SymbolType` enum. Returns None for symbol
        kinds the v2 classifier does not consume (sections, `TYPE_NONE`,
        `TYPE_OTHER`) so we don't pollute exact_lookup with addresses
        that should fall through to range-based section lookup instead.

        TLS objects are tagged `tls=True`; the type-string is derived from
        the section the symbol lives in (so a TLS object in `.tdata` gets
        `type="data"` matching how the range lookup labels `.tdata`).
        Regular `TYPE_OBJECT` symbols pick up their section's type
        (`rodata` / `data` / `bss`) so the classifier sees the same shape
        whether the address hit exact_lookup or range_lookup.
        """
        if sym_type in (
            cle.SymbolType.TYPE_SECTION,
            cle.SymbolType.TYPE_NONE,
            cle.SymbolType.TYPE_OTHER,
        ):
            return None

        section = self._find_section_for_addr(sym.rebased_addr)
        if sym_type == cle.SymbolType.TYPE_TLS_OBJECT:
            # TLS symbols' addresses are module-relative offsets, not virtual
            # addresses in the main object's section layout. The section
            # lookup may miss; default to `data` and let the classifier
            # treat the `tls` flag as authoritative.
            type_str = self._get_section_type(section) if section is not None else "data"
            tls = True
        elif sym_type == cle.SymbolType.TYPE_OBJECT:
            type_str = self._get_section_type(section) if section is not None else "data"
            tls = section is not None and self._section_is_tls(section)
        else:
            # Unknown/forward-compatible enum value: fall back to the section
            # if any, otherwise mark as a bare data symbol.
            type_str = self._get_section_type(section) if section is not None else "data"
            tls = section is not None and self._section_is_tls(section)

        meta: dict = {
            "name": sym.name,
            "type": type_str,
            "binding": sym.binding,
            "size": sym.size,
            "source": "symbol",
            "tls": tls,
        }
        if sym.size and sym.size > 0:
            meta["start_addr"] = sym.rebased_addr
            meta["end_addr"] = sym.rebased_addr + sym.size
        return meta

    def _build_indices(self):
        loader = self.project.loader
        main_obj = loader.main_object

        # -- Sections (if present)
        try:
            for section in main_obj.sections:
                if section.memsize == 0:
                    continue  # skip empty sections
                meta = {
                    "name": section.name,
                    "type": self._get_section_type(section),
                    "permissions": {
                        "r": section.is_readable,
                        "w": section.is_writable,
                        "x": section.is_executable,
                    },
                    "size": section.memsize,
                    "source": "section",
                    "tls": self._section_is_tls(section),
                }
                self.range_lookup[section.vaddr : section.vaddr + section.memsize] = meta
        except Exception as e:
            # some stripped binaries might not expose sections cleanly
            print(e)
            raise LookupError
            pass

        # -- Symbols (exact). Walk CLE's loader symbol table directly so we
        # can read each symbol's ELF `st_type` (TYPE_OBJECT / TYPE_FUNCTION /
        # TYPE_TLS_OBJECT) — the angr `kb.symbols` plugin does not expose
        # this without going back through CLE anyway. TYPE_FUNCTION entries
        # are left for the function-indexing pass below (it overwrites the
        # exact_lookup slot with richer per-function metadata).
        try:
            for sym in main_obj.symbols:
                if sym.rebased_addr is None:
                    continue
                sym_type = getattr(sym, "type", None)
                if sym_type == cle.SymbolType.TYPE_FUNCTION:
                    # Functions are reclassified below with library / plt
                    # / extern-synthetic information; skip here so we don't
                    # write a less-informative slot that the function pass
                    # later has to overwrite.
                    continue
                meta = self._build_symbol_meta(sym, sym_type)
                if meta is None:
                    continue
                self.exact_lookup[sym.rebased_addr] = meta
        except Exception as e:
            print(f"EXCEPTION: {e}")
            raise LookupError
            pass  # stripped binary fallback

        # -- Functions (range) with PLT / SimProcedure / extern / local split.
        # The v2 precedence list (see tokenizer/disasm/precedence.md) routes
        # each of these to a distinct token category:
        #   - step 2: `plt_func`     ← PLT stub in any loaded object
        #   - step 3: `local_func`   ← real entry in main object
        #   - step 4: `block`        ← inside main-object function body
        #     (range_lookup hit on a `local_function` meta, address != entry)
        #   - step 5: `ext_func` synthetic=false ← real entry in another
        #     loaded object (resolved import target)
        #   - step 6: `ext_func` synthetic=true  ← CLE SimProcedure stub
        # The is_plt / is_extern_synthetic booleans on the meta dict let the
        # classifier route without re-reading angr internals.
        try:
            for func in self.cfg.kb.functions.values():
                # Always include, but infer a minimal size if unknown
                size = func.size if func.size > 0 else 1

                library = self._find_library_for_addr(func.addr)

                is_plt = bool(func.is_plt)
                is_simprocedure = bool(func.is_simprocedure)

                # PLT stub: dispatch through the procedure linkage table. A
                # SimProcedure that also happens to be flagged is_plt is the
                # CLE-installed import resolver for the PLT slot; we keep it
                # under `plt_func` because the address still lives in `.plt`
                # of a real loaded object.
                if is_plt:
                    func_type = "plt_function"
                    source = "plt"
                    is_extern_synthetic = False
                elif is_simprocedure:
                    # SimProcedure outside `.plt`: a CLE synthetic extern-
                    # object slot for an unresolved or stub import.
                    func_type = "extern_function"
                    source = "extern"
                    is_extern_synthetic = True
                elif func.binary == main_obj:
                    func_type = "local_function"
                    source = "function"
                    is_extern_synthetic = False
                elif func.binary is not None:
                    # Real function entry in another loaded object.
                    func_type = "extern_function"
                    source = "extern"
                    is_extern_synthetic = False
                else:
                    func_type = "unknown_function"
                    source = "function"
                    is_extern_synthetic = False

                # Fallback to synthetic name if name is empty or autogenerated
                func_name = func.name
                if (
                    not func_name
                    or func_name.startswith("sub_")
                    or func_name.startswith("func_")
                    or func_name.startswith("unknown_")
                ):
                    func_name = f"sub_{func.addr:x}"

                meta = dict(
                    name=func_name,
                    type=func_type,
                    size=size,
                    start_addr=func.addr,
                    end_addr=func.addr + size,
                    source=source,
                    library=library,
                    is_plt=is_plt,
                    is_extern_synthetic=is_extern_synthetic,
                )

                self.exact_lookup[func.addr] = meta
                self.range_lookup[func.addr : func.addr + size] = meta
        except Exception as e:
            print(f"EXCEPTION during function indexing: {e}")

        # -- bss in stripped binaries
        try:
            for seg in self.project.loader.main_object.segments:
                if seg.memsize > seg.filesize:  # bss usually has memsize > filesize
                    bss_vaddr = seg.vaddr + seg.filesize
                    bss_size = seg.memsize - seg.filesize
                    meta = {
                        "name": ".bss",
                        "type": "bss",
                        "size": bss_size,
                        "source": "segment-inferred",
                        "tls": False,
                    }
                    self.range_lookup[bss_vaddr : bss_vaddr + bss_size] = meta
        except Exception as e:
            print(e)
            raise LookupError
            pass

        # -- ASCII string heuristic over read-only data. The classifier reads
        # `is_string` / `string_encoding` / `string_bytes` from the meta dict
        # at precedence step 7 (`string_ptr`). Encoding is always "ascii"
        # here — UTF-16 and other encodings are Ghidra-only (see
        # tokenizer/disasm/angr_limitations.md). False positives on
        # ASCII-byte runs inside non-string rodata are expected.
        self._string_tree: IntervalTree = IntervalTree()
        try:
            for section in main_obj.sections:
                if section.name != ".rodata":
                    continue
                if not section.is_readable or section.memsize == 0:
                    continue
                data = self.project.loader.memory.load(section.vaddr, section.memsize)
                for match in _ASCII_STRING_RE.finditer(data):
                    start = section.vaddr + match.start()
                    # Match length includes the trailing NUL; keep the NUL
                    # in `string_bytes` so byte-offset arithmetic by callers
                    # matches the on-disk layout (start_offset semantics in
                    # the v2 string sidecar).
                    raw = match.group()
                    self._string_tree[start : start + len(raw)] = raw
        except Exception as e:
            # Stripped or unusual binaries: skip string detection rather
            # than fail the whole lookup build. Conservative default
            # `is_string=False` will fire.
            print(f"EXCEPTION during string indexing: {e}")

    # ------------------------------------------------------------------
    # Typed-view population
    # ------------------------------------------------------------------
    # The lookup keeps its index entries as raw dicts (a pure indexing
    # concern, no consumer reads them). Per `lookup()` call, we translate
    # the matched index entry plus the per-address string-tree query into
    # the typed view's __slots__ in a single sweep — string -> enum
    # mapping happens here, ONCE per lookup, not at each property read.

    def _populate_view_slots(
        self,
        index_meta: dict,
        addr: int,
    ) -> None:
        """Translate ``index_meta`` + per-address string-tree query into
        typed slots on ``self._view``.
        """
        type_str: Optional[str] = index_meta.get("type")
        is_plt = bool(index_meta.get("is_plt"))
        is_extern_synthetic = bool(index_meta.get("is_extern_synthetic"))
        tls = bool(index_meta.get("tls"))

        # String-membership query is per-address, NOT per-index entry,
        # because the same address can land in a section meta and inside
        # an ASCII run simultaneously.
        string_hit = self._string_tree[addr]
        if string_hit:
            iv = min(string_hit, key=lambda iv: iv.end - iv.begin)
            is_string = True
            string_encoding = Encoding.ASCII
            string_bytes: Optional[bytes] = bytes(iv.data)
        else:
            is_string = False
            string_encoding = Encoding.UNKNOWN
            string_bytes = None

        # AddressKind precedence (matches what address_kind_from_meta did
        # in the transitional shim). Order corresponds to the v2 precedence
        # list in constant_handler._PRECEDENCE: string > vtable/code-ptr
        # > plt > extern_synthetic > base type.
        if is_string:
            kind = AddressKind.STRING
        elif is_plt:
            kind = AddressKind.PLT_FUNCTION
        elif is_extern_synthetic and (type_str or "").lower() in {
            "extern_function",
            "library_function",
            "plt_function",
            "unknown_function",
        }:
            kind = AddressKind.EXT_FUNCTION_SYNTHETIC
        else:
            kind = address_kind_from_string(type_str)

        section_kind = section_kind_from_type_string(type_str)

        # angr's index never carries a separate `section_name`; the meta
        # dict's `name` doubles as section name when source == "section".
        # The typed view exposes them separately so we conservatively
        # surface the section name only when the source IS a section.
        section_name: Optional[str] = None
        if index_meta.get("source") == "section":
            raw = index_meta.get("name")
            section_name = None if raw is None else str(raw)

        # `library` from the index has placeholder "unknown" for non-extern
        # entries; we surface None in that case so the typed property is
        # not accidentally populated with the placeholder string.
        raw_lib = index_meta.get("library")
        if raw_lib is None or raw_lib == "unknown":
            library: Optional[str] = None
        else:
            library = str(raw_lib)

        # Numeric / name fields: pass through, normalizing types. The
        # name is funnelled through ``canonical_function_name`` for
        # cross-provider parity with the Ghidra path (the helper
        # short-circuits to the raw name when both axes are None, which
        # is always the case on the angr path -- no demangler hook, no
        # thunk-identity surface, see ``angr_limitations.md``).
        raw_name = index_meta.get("name")
        name: Optional[str] = None if raw_name is None else canonical_function_name(
            str(raw_name), None, None
        )

        raw_size = index_meta.get("size")
        size: Optional[int] = None if raw_size is None else int(raw_size)

        raw_start = index_meta.get("start_addr")
        start_addr: Optional[int] = None if raw_start is None else int(raw_start)

        raw_end = index_meta.get("end_addr")
        end_addr: Optional[int] = None if raw_end is None else int(raw_end)

        self._view._populate(
            kind=kind,
            section_kind=section_kind,
            section_name=section_name,
            string_encoding=string_encoding,
            string_bytes=string_bytes,
            name=name,
            start_addr=start_addr,
            end_addr=end_addr,
            size=size,
            library=library,
            # angr cannot detect vtables (angr_limitations.md sec 2/3).
            is_vtable=False,
            tls=tls,
        )

    def lookup(self, addr) -> AddressMetadataView:
        """Resolve ``addr`` to a typed ``AddressMetadataView``.

        Mutates and returns the same per-lookup ``_AngrAddressMetadataView``
        instance each call (see ``AddressMetadataView`` lifecycle
        docstring). All typed slots are populated in one sweep here;
        consumers read typed properties exclusively.
        """
        logger = logging.getLogger(__name__)

        match = self.range_lookup[addr]
        # Find the most constrained (smallest) match
        if match:
            matches_list = list(match)
            # Find the interval with the smallest size
            match = min(matches_list, key=lambda iv: iv.end - iv.begin)

        if addr in self.exact_lookup:
            exact = self.exact_lookup[addr]

            if match is not None and match.data != exact:
                logger.fatal(f"Exact lookup mismatch for {addr:x}")

            self._populate_view_slots(exact, addr)
            return self._view

        if match:
            # Section-style entries don't carry start_addr/end_addr in
            # their stored dict; surface the matched interval bounds.
            meta = dict(match.data)
            meta.setdefault("start_addr", match.begin)
            meta.setdefault("end_addr", match.end)
            self._populate_view_slots(meta, addr)
            return self._view

        # Fallback to synthetic metadata to guarantee no empty result
        fallback_meta = {
            "start_addr": addr,
            "end_addr": addr,
            "name": f"unknown_{addr:x}",
            "type": "unknown",
            "size": 0,
            "source": "synthetic",
            "library": self._find_library_for_addr(addr),
        }
        self._populate_view_slots(fallback_meta, addr)
        return self._view


# Backward-compat alias
AddressMetaDataLookup = AngrMetadataLookup
