import re
from pathlib import Path
from typing import Any, Iterable, Optional

import angr

from tokenizer.disasm import DisassemblyProvider, MetadataLookup
from tokenizer.disasm.metadata import (
    AddressKind,
    AddressMetadataView,
    Encoding,
    SectionKind,
)


# ---------------------------------------------------------------------------
# Capstone-operand uniform ``fp_type`` default
# ---------------------------------------------------------------------------
# The angr path delivers raw Capstone CsOpnd objects (X86Op, ArmOp, ...) to
# consumer code. Capstone never populates an FP-precision signal on these,
# so the angr-side ``op.fp_type`` is uniformly ``None`` (matches the typed
# ``Optional[FpType]`` shape exposed by the Ghidra path's ``_CapOperand``;
# see ``tokenizer/disasm/types.py``). Stamping the default at module load
# (rather than per-instance per-instruction) keeps the consumer API uniform
# across providers — ``op.fp_type`` is a direct typed read with no
# ``getattr`` soft-probe — and avoids touching the Capstone object on the
# hot path. ``angr_limitations.md`` §1 documents why this field stays
# ``None`` on the angr side.
def _stamp_fp_type_default() -> None:
    """Attach class-level ``fp_type = None`` defaults to every Capstone
    operand class the angr-backed providers deliver to consumers.

    Only the classes we actually traverse are stamped; per-ISA imports are
    wrapped so an ISA whose Capstone bindings are unavailable in the active
    install (e.g. a stripped Capstone build) is silently skipped.
    """
    for module_name, class_name in (
        ("capstone.x86", "X86Op"),
        ("capstone.arm", "ArmOp"),
        ("capstone.arm64", "Arm64Op"),
        ("capstone.mips", "MipsOp"),
        ("capstone.ppc", "PpcOp"),
        ("capstone.riscv", "RiscvOp"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        # Skip if a value is already present (e.g. a future Capstone release
        # exposes the field natively or another module already stamped it).
        if "fp_type" not in cls.__dict__:
            cls.fp_type = None


_stamp_fp_type_default()


# ---------------------------------------------------------------------------
# Typed view for the angr-side MetadataLookup
# ---------------------------------------------------------------------------
class _AngrAddressMetadataView:
    """Concrete typed view returned by ``AngrMetadataLookup.lookup()``.

    Pure storage + read-only typed properties. ``AngrMetadataLookup`` calls
    ``_populate(...)`` to populate every slot at lookup time; consumers
    read typed properties exclusively. angr cannot resolve slot targets
    (``angr_limitations.md`` sections 2-3), so ``slot_target`` /
    ``jump_table_base_addr`` / ``jump_table_offset`` always return
    ``None``; ``string_encoding`` is always ASCII or UNKNOWN
    (``angr_limitations.md`` section 4).

    LIFECYCLE: instance is REUSED across ``lookup()`` calls. Use
    ``copy.deepcopy(view)`` to stash across lookups.
    """

    __slots__ = (
        "_kind",
        "_section_kind",
        "_section_name",
        "_string_encoding",
        "_string_bytes",
        "_name",
        "_start_addr",
        "_end_addr",
        "_size",
        "_library",
        "_is_vtable",
        "_tls",
    )

    def __init__(self) -> None:
        self._kind: AddressKind = AddressKind.NONE
        self._section_kind: SectionKind = SectionKind.UNKNOWN
        self._section_name: Optional[str] = None
        self._string_encoding: Encoding = Encoding.UNKNOWN
        self._string_bytes: Optional[bytes] = None
        self._name: Optional[str] = None
        self._start_addr: Optional[int] = None
        self._end_addr: Optional[int] = None
        self._size: Optional[int] = None
        self._library: Optional[str] = None
        self._is_vtable: bool = False
        self._tls: bool = False

    def _populate(
        self,
        *,
        kind: AddressKind,
        section_kind: SectionKind,
        section_name: Optional[str],
        string_encoding: Encoding,
        string_bytes: Optional[bytes],
        name: Optional[str],
        start_addr: Optional[int],
        end_addr: Optional[int],
        size: Optional[int],
        library: Optional[str],
        is_vtable: bool,
        tls: bool,
    ) -> None:
        """Replace all slot state in one call. Used by the lookup at
        the start of every ``lookup()`` so the consumer sees a consistent
        view bound to the current address.
        """
        self._kind = kind
        self._section_kind = section_kind
        self._section_name = section_name
        self._string_encoding = string_encoding
        self._string_bytes = string_bytes
        self._name = name
        self._start_addr = start_addr
        self._end_addr = end_addr
        self._size = size
        self._library = library
        self._is_vtable = is_vtable
        self._tls = tls

    # -- Typed property surface (AddressMetadataView Protocol) --------------
    @property
    def kind(self) -> AddressKind:
        return self._kind

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def section_kind(self) -> SectionKind:
        return self._section_kind

    @property
    def section_name(self) -> Optional[str]:
        return self._section_name

    @property
    def start_addr(self) -> Optional[int]:
        return self._start_addr

    @property
    def end_addr(self) -> Optional[int]:
        return self._end_addr

    @property
    def size(self) -> Optional[int]:
        return self._size

    @property
    def library(self) -> Optional[str]:
        return self._library

    @property
    def string_encoding(self) -> Encoding:
        return self._string_encoding

    @property
    def string_bytes(self) -> Optional[bytes]:
        return self._string_bytes

    @property
    def is_vtable(self) -> bool:
        return self._is_vtable

    @property
    def tls(self) -> bool:
        return self._tls

    @property
    def slot_target(self) -> Optional[AddressMetadataView]:
        # angr cannot resolve slot targets (angr_limitations.md sections 2-3).
        return None

    @property
    def jump_table_base_addr(self) -> Optional[int]:
        return None

    @property
    def jump_table_offset(self) -> Optional[int]:
        return None

    def __deepcopy__(self, memo) -> "_AngrAddressMetadataView":
        clone = _AngrAddressMetadataView()
        clone._kind = self._kind
        clone._section_kind = self._section_kind
        clone._section_name = self._section_name
        clone._string_encoding = self._string_encoding
        # bytes is immutable; safe to share
        clone._string_bytes = self._string_bytes
        clone._name = self._name
        clone._start_addr = self._start_addr
        clone._end_addr = self._end_addr
        clone._size = self._size
        clone._library = self._library
        clone._is_vtable = self._is_vtable
        clone._tls = self._tls
        return clone


# Re-exports of the angr-side MetadataLookup. The lookup class itself is
# defined in ``tokenizer/address_meta_data_lookup.py`` (its original home)
# to avoid a hard import cycle with this provider file. The names are
# re-exported here so callers (and the task-validation step) can import
# both lookup + view via the provider module.
def _import_lookup_classes() -> tuple[type, type]:
    """Lazy import of ``AngrMetadataLookup`` / ``AddressMetaDataLookup``.

    Deferred to a function so the module-load cycle
    (``angr_provider`` <-> ``address_meta_data_lookup``) is broken.
    """
    from tokenizer.address_meta_data_lookup import AddressMetaDataLookup, AngrMetadataLookup
    return AngrMetadataLookup, AddressMetaDataLookup


def __getattr__(name: str):
    """Module-level ``__getattr__`` for re-export laziness.

    ``from tokenizer.disasm.angr_provider import AngrMetadataLookup``
    triggers this on first access; the import resolves at that point so
    we don't pay the ``address_meta_data_lookup`` cost at module load.
    """
    if name in {"AngrMetadataLookup", "AddressMetaDataLookup"}:
        AngrMetadataLookup, AddressMetaDataLookup = _import_lookup_classes()
        return {"AngrMetadataLookup": AngrMetadataLookup, "AddressMetaDataLookup": AddressMetaDataLookup}[name]
    raise AttributeError(f"module 'tokenizer.disasm.angr_provider' has no attribute {name!r}")


__all__ = [  # noqa: F822 - "AngrMetadataLookup" / "AddressMetaDataLookup" resolved by __getattr__
    "AddressMetaDataLookup",
    "AngrDisassemblyProvider",
    "AngrMetadataLookup",
    "_AngrAddressMetadataView",
]


class AngrDisassemblyProvider(DisassemblyProvider):
    def __init__(self, binary_path: Path) -> None:
        self.binary_path = binary_path
        self.project: angr.Project = angr.Project(binary_path, auto_load_libs=False)
        self.cfg: angr.analyses.cfg.cfg_fast.CFGFast | None = None

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
        assert self.cfg is not None, "CFG not built yet — call build_cfg() first"
        return len(self.cfg.functions)

    def iter_functions(self) -> Iterable[tuple[int, str, Any]]:
        assert self.cfg is not None, "CFG not built yet — call build_cfg() first"
        for func_addr, func in sorted(self.cfg.functions.items(), key=lambda item: item[1].name):
            func_name = func.name
            if func_name in ("UnresolvableCallTarget", "UnresolvableJumpTarget"):
                continue
            yield func_addr, func_name, func
