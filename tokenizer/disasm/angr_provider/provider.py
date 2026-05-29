"""angr-based ``DisassemblyProvider`` entrypoint.

Owns ``AngrDisassemblyProvider``: drives ``angr.Project`` load + CFGFast
analysis, exposes the typed function/block/instruction iteration via
the owned ``_AngrFunctionView`` cursor, and constructs the angr-side
``AddressMetaDataLookup`` (defined in ``tokenizer/address_meta_data_lookup.py``
to avoid a module-load cycle).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import angr

from tokenizer.disasm import DisassemblyProvider, MetadataLookup
from tokenizer.disasm.angr_provider.function_identity import _angr_identity_key
from tokenizer.disasm.angr_provider.op_classify import _resolve_architecture
from tokenizer.disasm.angr_provider.views import _AngrFunctionView
from tokenizer.disasm.types import Architecture, FunctionView
from tokenizer.function_deduper import canonical_function_name


# Re-exports of the angr-side MetadataLookup. The lookup class itself is
# defined in ``tokenizer/address_meta_data_lookup.py`` (its original home)
# to avoid a hard import cycle with this provider file.
def _import_lookup_classes() -> tuple[type, type]:
    """Lazy import of ``AngrMetadataLookup`` / ``AddressMetaDataLookup``.

    Deferred to a function so the module-load cycle
    (``angr_provider`` <-> ``address_meta_data_lookup``) is broken.
    """
    from tokenizer.address_meta_data_lookup import AddressMetaDataLookup, AngrMetadataLookup
    return AngrMetadataLookup, AddressMetaDataLookup


class AngrDisassemblyProvider(DisassemblyProvider):
    def __init__(self, binary_path: Path) -> None:
        self.binary_path = binary_path
        self.project: angr.Project = angr.Project(binary_path, auto_load_libs=False)
        self.cfg: angr.analyses.cfg.cfg_fast.CFGFast | None = None
        self._arch: Architecture = _resolve_architecture(self.project.arch)
        # Single reusable owned ``FunctionView`` cursor mutated by every
        # ``iter_functions`` step. Per the lifecycle docstring at the top
        # of ``tokenizer/disasm/types.py`` the wrapper is REUSED across
        # iteration; consumers that need to stash a snapshot must
        # ``copy.deepcopy(view)``.
        self._function_cursor: _AngrFunctionView = _AngrFunctionView(self._arch)

    def build_cfg(self) -> None:
        self.cfg = self.project.analyses.CFGFast(normalize=True)

    def get_text_section_bounds(self) -> tuple[int, int]:
        for section in self.project.loader.main_object.sections:
            if section.name == ".text":
                return section.vaddr, section.vaddr + section.memsize
        return 0, 0

    def parse_data_sections(
        self,
        sections: list[str] | None = None,
        output_csv_path: str | None = None,
    ) -> dict[str, list[str]]:
        if sections is None:
            sections = [".rodata"]

        all_entries = []
        addr_dict: dict[str, list[str]] = {}

        for sec in self.project.loader.main_object.sections:
            if sec.name not in sections:
                continue
            if sec.name == ".rodata" and sec.is_readable and sec.memsize > 0:
                data = self.project.loader.memory.load(sec.vaddr, sec.memsize)
                for match in re.finditer(b"[\x20-\x7e]{4,}\x00", data):
                    s = match.group().rstrip(b"\x00").decode("utf-8", errors="ignore")
                    start = sec.vaddr + match.start()
                    entry = {
                        "section": ".rodata",
                        "start": hex(start),
                        "end": hex(start + len(s) + 1),
                        "value": f'"{s}"',
                    }
                    all_entries.append(entry)
                    addr_dict[entry["start"]] = [entry["end"], entry["section"], entry["value"]]

        if output_csv_path:
            csv_path = Path(output_csv_path)
            consts_path = csv_path.parent / f"{csv_path.stem.replace('_output', '')}_consts.txt"
        else:
            consts_path = Path("parsed_constants.txt")

        with open(consts_path, "w") as f:
            for e in all_entries:
                f.write(f"{e['start']} - {e['end']}: {e['section']}: {e['value']}\n")

        print(f"Parsed {len(all_entries)} .rodata constants with exact addresses into {consts_path}")
        return addr_dict

    def create_metadata_lookup(self) -> MetadataLookup:
        # Lazy import keeps the ``angr_provider`` <-> ``address_meta_data_lookup``
        # circular-import broken at module load (see ``_import_lookup_classes``).
        _, AddressMetaDataLookupCls = _import_lookup_classes()
        return AddressMetaDataLookupCls(self.binary_path)

    def function_count(self) -> int:
        assert self.cfg is not None, "CFG not built yet -- call build_cfg() first"
        return len(self.cfg.functions)

    def iter_functions(self) -> Iterable[tuple[int, str, FunctionView]]:
        """Yield ``(addr, name, function_view)`` per surviving CFG function.

        The third tuple element is the SAME reused ``_AngrFunctionView``
        cursor mutated to point at each function in turn (see lifecycle
        docstring at the top of ``tokenizer/disasm/types.py``). Consumers
        that need to hold a function reference across an advance must
        ``copy.deepcopy(view)`` the snapshot.
        """
        assert self.cfg is not None, "CFG not built yet -- call build_cfg() first"
        # Sort by the CANONICAL name (the same string main_loop derives
        # from the view's three identity axes), not the raw angr name: a
        # thunk's canonical name is its identity key, which the raw name
        # does not track. The view mirrors these axes — its ``name`` is
        # ``str(func.name)``, its ``comment`` is unconditionally None, and
        # its ``identity_key`` is ``_angr_identity_key(func)`` — so the
        # key below reproduces the canonical main_loop will recompute.
        def _canonical_sort_key(item: tuple[int, Any]) -> str:
            func = item[1]
            return canonical_function_name(
                str(func.name), None, _angr_identity_key(func)
            )

        for func_addr, func in sorted(self.cfg.functions.items(), key=_canonical_sort_key):
            func_name = func.name
            if func_name in ("UnresolvableCallTarget", "UnresolvableJumpTarget"):
                continue
            self._function_cursor._set(func)
            yield func_addr, func_name, self._function_cursor
