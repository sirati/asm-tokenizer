import logging
import os
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

_logger = logging.getLogger(__name__)


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
    debug_render: bool = False,
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

    ``debug_render`` is the run-scoped debug-rendering mode (set by the
    ``--debug`` CLI entry point): when True, instruction views attach
    operand-text render thunks to their ``InsnDebugLabel``s so debug
    output shows the full ``"<mnemonic> <op_str>"`` text. Ghidra-only
    cost knob: on Ghidra the rendering is a per-operand JVM round-trip,
    so production (False) skips it entirely; angr's Capstone rendering
    is a plain attribute read, so the angr provider captures the
    operand text unconditionally and ignores the parameter.
    """
    if backend == "angr":
        from tokenizer.disasm.angr_provider import AngrDisassemblyProvider

        return AngrDisassemblyProvider(binary_path)
    if backend == "ghidra":
        from tokenizer.disasm.ghidra_provider import GhidraDisassemblyProvider

        return GhidraDisassemblyProvider(
            binary_path,
            duplicate_function_dump_path=duplicate_function_dump_path,
            debug_render=debug_render,
        )
    raise ValueError(f"Unsupported disassembly backend: {backend}")


def configure_worker_jvm_processor_cap(workers_per_node: int | None) -> None:
    """Size this worker's disassembly JVM to its fair share of the node.

    The worker entry point calls this ONCE at startup (before any
    provider is constructed) with the per-node worker count the dispatch
    runs with. Each worker's Ghidra JVM is then capped to
    ``ceil(machine_cores / workers_per_node)`` processors so a node
    running ``workers_per_node`` workers stops oversubscribing the CPU at
    ``workers_per_node × machine_width`` (the production starvation
    incident: every worker's JVM sized its analysis pools from the full
    host core count).

    ``workers_per_node is None`` (the dispatch exposed no worker count)
    OR an undetectable machine core count → NO cap, with a single WARN —
    never crash. ``machine_cores`` is ``os.cpu_count()`` (the container
    sees the host CPUs).

    The cap is a Ghidra-only knob (angr sizes nothing from
    ``availableProcessors``); the install delegates to the Ghidra
    provider's process-global JVM-startup config. This stays the single
    consumer-facing surface so the worker never reaches into the
    backend-specific submodule.
    """
    from tokenizer.disasm.ghidra_provider.jvm_processor_cap import (
        compute_processor_cap,
    )
    from tokenizer.disasm.ghidra_provider.provider import set_processor_cap

    machine_cores = os.cpu_count()
    if workers_per_node is None or machine_cores is None or machine_cores <= 0:
        _logger.warning(
            "JVM processor cap disabled (workers_per_node=%r, "
            "machine_cores=%r); Ghidra sizes its thread pools from the "
            "full host width — expect CPU oversubscription when multiple "
            "workers share a node.",
            workers_per_node,
            machine_cores,
        )
        set_processor_cap(None)
        return
    cap = compute_processor_cap(machine_cores, workers_per_node)
    _logger.info(
        "JVM processor cap = %d (machine_cores=%d / workers_per_node=%d)",
        cap,
        machine_cores,
        workers_per_node,
    )
    set_processor_cap(cap)
