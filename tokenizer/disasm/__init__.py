from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Protocol


class MetadataLookup(Protocol):
    def lookup(self, addr: int) -> tuple[dict, str]: ...


class DisassemblyProvider(ABC):
    """Abstract base class for disassembly backends.

    Encapsulates binary loading, CFG construction, section parsing,
    function/block/instruction iteration, and metadata lookup construction.

    Instruction objects are backend-specific (e.g. Capstone CsInsn for angr).
    Architecture providers consume them directly.
    """

    @abstractmethod
    def __init__(self, binary_path: Path) -> None: ...

    @abstractmethod
    def build_cfg(self) -> None: ...

    @abstractmethod
    def get_text_section_bounds(self) -> tuple[int, int]: ...

    @abstractmethod
    def parse_data_sections(
        self,
        sections: list[str] | None = None,
        output_csv_path: str | None = None,
    ) -> dict[str, list[str]]: ...

    @abstractmethod
    def create_metadata_lookup(self) -> MetadataLookup: ...

    @abstractmethod
    def function_count(self) -> int: ...

    @abstractmethod
    def iter_functions(self) -> Iterable[tuple[int, str, Any]]: ...


def get_disassembly_provider(backend: str, binary_path: Path) -> DisassemblyProvider:
    if backend == "angr":
        from tokenizer.disasm.angr_provider import AngrDisassemblyProvider

        return AngrDisassemblyProvider(binary_path)
    if backend == "ghidra":
        from tokenizer.disasm.ghidra_provider import GhidraDisassemblyProvider

        return GhidraDisassemblyProvider(binary_path)
    raise ValueError(f"Unsupported disassembly backend: {backend}")
