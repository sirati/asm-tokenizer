"""The binary reader's consumed path fields, without opening a dataset."""
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class BinaryFiles:
    sections_bin: Path
    matched_data: Path
    matched_index: Path
    unmatched_data: Path
    unmatched_index: Path
    variants_bin: Path
    variants_sidecar: Path
    function_names_sidecar: Path
    extern_providers: Path

    @classmethod
    def at(cls, base_path: Path, binary_name: str) -> BinaryFiles:
        base = Path(base_path)
        return cls(
            sections_bin=base / f'{binary_name}_sections.bin',
            matched_data=base / f'{binary_name}_data.bin',
            matched_index=base / f'{binary_name}_index.bin',
            unmatched_data=base / f'{binary_name}_unmatched_data.bin',
            unmatched_index=base / f'{binary_name}_unmatched_index.bin',
            variants_bin=base / f'{binary_name}_variants.bin',
            variants_sidecar=base / f'{binary_name}_variants.csv',
            function_names_sidecar=base / f'{binary_name}_function_names.txt',
            extern_providers=base / f'{binary_name}_extern_providers.txt')

    def paths(self) -> tuple[Path, ...]:
        return tuple(vars(self)[schema.name] for schema in fields(self))
