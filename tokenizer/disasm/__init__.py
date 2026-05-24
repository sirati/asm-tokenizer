from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Protocol, Tuple

from tokenizer.disasm.metadata import (
    AddressKind,
    AddressMetadataView,
    Encoding,
    SectionKind,
)
from tokenizer.disasm.types import (
    AddressSizePrefixView,
    Architecture,
    ArmConditionCode,
    BlockView,
    BlocksView,
    BranchHintPrefixView,
    ConditionCodePrefixView,
    CrxFieldView,
    FpType,
    FunctionView,
    InstructionPrefixView,
    InstructionView,
    InstructionsView,
    JumpTableView,
    LockPrefixView,
    MemoryOperandView,
    OperandKind,
    OperandSizePrefixView,
    OperandView,
    OperandsView,
    PpcBranchConditionPrefixView,
    PpcUpdateCr0PrefixView,
    PrefixesView,
    RegisterView,
    RepPrefixView,
    SegmentOverridePrefixView,
    ShiftKind,
    ShiftModifierView,
    UpdateFlagsPrefixView,
    WritebackPrefixView,
    X86BranchHint,
    X86Segment,
)


class MetadataLookup(Protocol):
    def lookup(self, addr: int) -> AddressMetadataView:
        """Resolve ``addr`` to a typed ``AddressMetadataView``.

        The returned view is the SAME wrapper instance per
        ``MetadataLookup`` (lifecycle per ``AddressMetadataView``
        docstring). Consumers read typed properties exclusively
        (``meta.kind``, ``meta.name``, ``meta.string_encoding`` etc.);
        the transitional dict-shim and v1-tuple-unpacking adapters from
        Phase D.1/D.2 are gone (Phase D.3 / task #40). Use
        ``copy.deepcopy(meta)`` to stash across lookups.
        """
        ...


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

    def iter_switch_tables(self, function: Any) -> Iterable[Tuple[int, list[int]]]:
        """Yield ``(jump_table_addr, [target_block_addrs])`` per switch table.

        Recovers indirect-jump (switch-statement) tables within
        ``function``. ``jump_table_addr`` is the address of the
        backing pointer-array in rodata; ``target_block_addrs`` is the
        slot-order list of resolved target block start addresses.

        Default implementation returns an empty iterator. Providers
        whose backends support switch-table recovery (e.g. Ghidra)
        override this; providers that don't (angr, per
        ``angr_limitations.md`` §3) inherit the empty default and
        the function-finalization hook emits no jump-table footer.
        Kept on the abstract base (not as an extra Protocol) so the
        finalization hook in ``fill_constant_candidates`` can call
        it uniformly without provider-specific branching.
        """
        return iter(())

    def close(self) -> None:
        """Release backend-held resources for this binary.

        Worker processes are reused across tasks (the framework
        respawns only when `always_restart_worker=True`). Backends
        with process-global state — notably Ghidra's JVM, which
        holds a Project + Program plus analysis threads — must
        unload the current binary here so the next task gets a
        clean slate. Backends that hold only Python-side state
        (angr) can leave this as the default no-op.

        Callers are expected to invoke `close()` from a
        `try`/`finally` block surrounding all uses of the provider.
        """

    def __enter__(self) -> "DisassemblyProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def get_disassembly_provider(
    backend: str,
    binary_path: Path,
    duplicate_function_dump_path: Path | None = None,
) -> DisassemblyProvider:
    """Return a ``DisassemblyProvider`` for ``backend`` on ``binary_path``.

    ``duplicate_function_dump_path`` is a Ghidra-only debug knob: when
    set, the Ghidra provider's ``iter_functions`` walks the function
    list once, detects name-collisions, and writes a 5-layer-deep
    pickle metadata snapshot for each colliding function to that path.
    The angr backend silently ignores the parameter - angr exposes no
    Java-handle equivalent (the dump is for offline picking of a
    cross-ISA-stable disambiguator, and Ghidra is the default
    provider for that pathway per ``ghidra_default_provider`` in
    project memory).
    """
    if backend == "angr":
        from tokenizer.disasm.angr_provider import AngrDisassemblyProvider

        return AngrDisassemblyProvider(binary_path)
    if backend == "ghidra":
        from tokenizer.disasm.ghidra_provider import GhidraDisassemblyProvider

        return GhidraDisassemblyProvider(
            binary_path,
            duplicate_function_dump_path=duplicate_function_dump_path,
        )
    raise ValueError(f"Unsupported disassembly backend: {backend}")
