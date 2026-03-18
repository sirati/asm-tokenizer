import re
from pathlib import Path
from typing import Any, Iterable

import angr

from tokenizer.address_meta_data_lookup import AddressMetaDataLookup
from tokenizer.disasm import DisassemblyProvider, MetadataLookup


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
        return AddressMetaDataLookup(self.binary_path)

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
