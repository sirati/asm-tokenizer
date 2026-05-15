import re
from pathlib import Path
from typing import Any, Iterable

import angr

from tokenizer.disasm import DisassemblyProvider, MetadataLookup
from tokenizer.disasm.metadata import _DictBackedAddressMetadataView


# ---------------------------------------------------------------------------
# Typed view for the angr-side MetadataLookup
# ---------------------------------------------------------------------------
class _AngrAddressMetadataView(_DictBackedAddressMetadataView):
    """Concrete typed-view for the angr-side ``MetadataLookup``.

    Inherits the dict-backed transitional surface and base-class typed
    properties from ``_DictBackedAddressMetadataView``. angr cannot
    resolve slot targets (see ``tokenizer/disasm/angr_limitations.md``
    sections 2 and 3) so ``slot_target`` / ``jump_table_base_addr`` /
    ``jump_table_offset`` stay at their ``None`` defaults from the base
    class. ``string_encoding`` reads from the v1 dict (always ASCII or
    UNKNOWN per ``angr_limitations.md`` section 4); the base class's
    ``encoding_from_string`` mapping handles that uniformly.

    Conservative behavior is the v2 contract for angr: the classifier
    falls through to a lower precedence step rather than emitting
    misclassified tokens.
    """

    __slots__ = ()

    def legacy_dict(self) -> tuple[dict, str]:
        """Phase-D.2 transitional adapter - returns ``(v1_dict, kind_str)``.

        Thin override of the base implementation so the canonical
        location is co-located with the rest of the angr-side typed
        view; the actual delegation lives in
        ``_DictBackedAddressMetadataView.legacy_dict``. Removed when
        Phase D.3 retires every ``meta.get(...)`` / ``meta[...]`` call
        site.
        """
        return super().legacy_dict()


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
