"""Ghidra-based disassembly provider using pyghidra (headless, no IPC).

pyghidra gives direct access to the Ghidra Java API from CPython via JPype.
This provider translates Ghidra's Instruction/Register/Scalar objects into
Capstone-compatible adapter objects so existing ArchitectureProviders work
unchanged.

Requirements:
    pip install pyghidra
    GHIDRA_INSTALL_DIR env var or pass install_dir to pyghidra.start()
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tokenizer.disasm import DisassemblyProvider, MetadataLookup

# ---------------------------------------------------------------------------
# Capstone-compatible adapter objects
# ---------------------------------------------------------------------------
# The existing ArchitectureProviders access deeply ISA-specific Capstone
# attributes.  Rather than abstracting those away (which would create a
# bloated union type), we produce thin wrappers that expose the same
# attribute surface as Capstone objects.
#
# Capstone operand type constants (mirrored here so we don't need capstone
# installed when only Ghidra is used):
_OP_REG = 1
_OP_IMM = 2
_OP_MEM = 3


@dataclass
class _CapMemOperand:
    """Capstone-compatible memory operand fields."""

    base: int = 0
    index: int = 0
    scale: int = 1
    disp: int = 0
    segment: int = 0


@dataclass
class _CapShift:
    """Capstone-compatible shift descriptor (ARM32)."""

    type: int = 0
    value: int = 0


@dataclass
class _CapOperand:
    """Capstone-compatible operand object.

    Fields are set depending on ``type``:
        1 (REG)  -> ``reg``
        2 (IMM)  -> ``imm``
        3 (MEM)  -> ``mem``
       64 (CRX, PPC only) -> ``crx``
    """

    type: int = 0
    reg: int = 0
    imm: int = 0
    mem: _CapMemOperand = field(default_factory=_CapMemOperand)
    size: int = 0  # x86 operand size in bytes
    shift: _CapShift = field(default_factory=_CapShift)

    @dataclass
    class _CRX:
        reg: int = 0

    crx: _CRX = field(default_factory=_CRX)


@dataclass
class _CapInsnInner:
    """Stands in for ``insn.insn`` — the raw Capstone CsInsn that
    ArchitectureProviders access for ISA-specific fields."""

    _insn_name: str = ""
    cc: int = 0  # ARM32 condition code
    update_flags: bool = False  # ARM32 S suffix
    writeback: bool = False  # ARM32 ! suffix
    bc: int = 0  # PPC branch condition
    update_cr0: bool = False  # PPC Rc bit

    def insn_name(self) -> str:
        return self._insn_name


class _CapInstruction:
    """Capstone-compatible instruction adapter wrapping a Ghidra Instruction.

    Attributes consumed by existing code:
        mnemonic, op_str, operands, prefix, insn (._CapInsnInner),
        reg_name(reg_id)
    """

    __slots__ = ("mnemonic", "op_str", "operands", "prefix", "insn", "_reg_map")

    def __init__(
        self,
        mnemonic: str,
        op_str: str,
        operands: list[_CapOperand],
        prefix: bytes,
        insn_inner: _CapInsnInner,
        reg_map: dict[int, str],
    ):
        self.mnemonic = mnemonic
        self.op_str = op_str
        self.operands = operands
        self.prefix = prefix
        self.insn = insn_inner
        self._reg_map = reg_map

    def reg_name(self, reg_id: int) -> str:
        return self._reg_map.get(reg_id, f"reg{reg_id}")


@dataclass
class _CapBlock:
    """Capstone-compatible block adapter.

    ``fill_constant_candidates`` accesses:
        block.addr, block.size, block.capstone.insns
    """

    addr: int
    size: int
    capstone: Any = None  # set after construction

    @dataclass
    class _CapstoneHolder:
        insns: list[_CapInstruction]

    def set_insns(self, insns: list[_CapInstruction]) -> None:
        self.capstone = self._CapstoneHolder(insns)


@dataclass
class _CapFunction:
    """Capstone-compatible function adapter.

    ``fill_constant_candidates`` accesses:
        func.blocks  (iterable, used multiple times — must be a list)
    """

    _blocks: list[_CapBlock]

    @property
    def blocks(self) -> list[_CapBlock]:
        return self._blocks


# ---------------------------------------------------------------------------
# Ghidra metadata lookup
# ---------------------------------------------------------------------------
class GhidraMetadataLookup:
    """Address metadata lookup built from Ghidra's analysis results.

    Conforms to the ``MetadataLookup`` protocol (just needs a
    ``lookup(addr) -> (dict, str)`` method).
    """

    def __init__(self, program: Any, function_manager: Any) -> None:
        self._program = program
        self._fm = function_manager
        self._memory = program.getMemory()
        self._symbol_table = program.getSymbolTable()
        self._listing = program.getListing()

    def lookup(self, addr: int) -> tuple[dict, str]:
        from ghidra.program.model.address import AddressSet

        addr_obj = self._program.getAddressFactory().getDefaultAddressSpace().getAddress(addr)

        # Exact symbol match
        symbols = self._symbol_table.getSymbols(addr_obj)
        if symbols:
            sym = symbols[0]
            meta = {
                "name": str(sym.getName()),
                "type": "symbol",
                "size": 0,
                "source": "symbol",
                "start_addr": addr,
                "end_addr": addr,
            }
            return meta, "exact"

        # Function match
        func = self._fm.getFunctionContaining(addr_obj)
        if func is not None:
            entry = int(func.getEntryPoint().getOffset())
            body = func.getBody()
            size = int(body.getNumAddresses())
            is_external = func.isExternal() or func.isThunk()
            meta = {
                "name": str(func.getName()),
                "type": "library_function" if is_external else "local_function",
                "size": size,
                "start_addr": entry,
                "end_addr": entry + size,
                "source": "function",
                "library": "unknown",
            }
            return meta, "range"

        # Memory block match
        block = self._memory.getBlock(addr_obj)
        if block is not None:
            meta = {
                "name": str(block.getName()),
                "type": "rodata"
                if not block.isWrite() and not block.isExecute()
                else "code"
                if block.isExecute()
                else "data",
                "size": int(block.getSize()),
                "start_addr": int(block.getStart().getOffset()),
                "end_addr": int(block.getEnd().getOffset()) + 1,
                "source": "section",
            }
            return meta, "range"

        # Fallback
        fallback = {
            "start_addr": addr,
            "end_addr": addr,
            "name": f"unknown_{addr:x}",
            "type": "unknown",
            "size": 0,
            "source": "synthetic",
            "library": "unknown",
        }
        return fallback, "synthetic"


# ---------------------------------------------------------------------------
# Instruction translation helpers
# ---------------------------------------------------------------------------


def _build_register_map(program: Any) -> dict[int, str]:
    """Build a mapping from Ghidra register hash -> lowercase register name.

    Ghidra doesn't use integer register IDs like Capstone.  We assign a
    stable int id to each register (hash of the Register object) and keep
    a reverse map so ``_CapInstruction.reg_name()`` works.
    """
    reg_map: dict[int, str] = {}
    language = program.getLanguage()
    for reg in language.getRegisters():
        reg_map[hash(reg)] = str(reg.getName()).lower()
    return reg_map


def _ghidra_insn_to_cap(
    ghidra_insn: Any,
    reg_map: dict[int, str],
    program: Any,
) -> _CapInstruction:
    """Translate a single Ghidra Instruction into a _CapInstruction."""
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import Register
    from ghidra.program.model.scalar import Scalar

    mnemonic = str(ghidra_insn.getMnemonicString()).lower()
    num_ops = ghidra_insn.getNumOperands()

    op_strs = []
    operands: list[_CapOperand] = []

    for i in range(num_ops):
        op_str_i = str(ghidra_insn.getDefaultOperandRepresentation(i))
        op_strs.append(op_str_i)
        objects = ghidra_insn.getOpObjects(i)

        if not objects:
            continue

        first = objects[0]

        if isinstance(first, Register):
            reg_id = hash(first)
            if str(first.getName()).lower() not in reg_map.values():
                reg_map[reg_id] = str(first.getName()).lower()
            operands.append(_CapOperand(type=_OP_REG, reg=reg_id))

        elif isinstance(first, Scalar):
            operands.append(_CapOperand(type=_OP_IMM, imm=int(first.getValue())))

        elif isinstance(first, Address):
            # Ghidra represents memory references as Address objects.
            # We model this as an immediate (address target) — the
            # architecture providers will route it through the constant
            # handler just like Capstone immediates.
            operands.append(_CapOperand(type=_OP_IMM, imm=int(first.getOffset())))

        else:
            # Multiple objects in one operand slot -> likely memory operand
            # e.g. [base + disp]
            mem = _CapMemOperand()
            for obj in objects:
                if isinstance(obj, Register):
                    rid = hash(obj)
                    if str(obj.getName()).lower() not in reg_map.values():
                        reg_map[rid] = str(obj.getName()).lower()
                    if mem.base == 0:
                        mem.base = rid
                    else:
                        mem.index = rid
                elif isinstance(obj, Scalar):
                    mem.disp = int(obj.getValue())
                elif isinstance(obj, Address):
                    mem.disp = int(obj.getOffset())
            operands.append(_CapOperand(type=_OP_MEM, mem=mem))

    # If we got a single operand slot with multiple objects that we didn't
    # already handle as MEM above, re-check:
    # (The loop above handles it, but let's also handle the case where a
    # single slot has both Register+Scalar → memory operand)
    # This is already covered above.

    insn_inner = _CapInsnInner(_insn_name=mnemonic)

    return _CapInstruction(
        mnemonic=mnemonic,
        op_str=", ".join(op_strs),
        operands=operands,
        prefix=b"",
        insn_inner=insn_inner,
        reg_map=reg_map,
    )


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GhidraDisassemblyProvider(DisassemblyProvider):
    """Disassembly provider backed by Ghidra via pyghidra (headless, no IPC).

    Usage::

        provider = GhidraDisassemblyProvider(Path("binary"))
        provider.build_cfg()   # runs Ghidra auto-analysis
        for addr, name, func in provider.iter_functions():
            ...
    """

    def __init__(self, binary_path: Path) -> None:
        import pyghidra

        if not pyghidra.started():
            pyghidra.start()

        self.binary_path = binary_path
        # open_program is the simplest API: import, auto-analyze, return
        # a FlatProgramAPI context manager.  We defer analysis (analyze=False)
        # so build_cfg() controls when it happens.
        self._ctx = pyghidra.open_program(binary_path, analyze=False)
        self._flat_api = self._ctx.__enter__()
        self._program = self._flat_api.getCurrentProgram()
        self._fm = self._program.getFunctionManager()
        self._listing = self._program.getListing()
        self._memory = self._program.getMemory()
        self._reg_map = _build_register_map(self._program)
        self._analyzed = False

    def build_cfg(self) -> None:
        """Run Ghidra's auto-analysis (the equivalent of CFGFast)."""
        self._flat_api.analyzeAll(self._program)
        self._analyzed = True

    def get_text_section_bounds(self) -> tuple[int, int]:
        for block in self._memory.getBlocks():
            name = str(block.getName())
            if name == ".text":
                start = int(block.getStart().getOffset())
                return start, start + int(block.getSize())
        return 0, 0

    def parse_data_sections(
        self,
        sections: list[str] | None = None,
        output_csv_path: str | None = None,
    ) -> dict[str, list[str]]:
        if sections is None:
            sections = [".rodata"]

        all_entries: list[dict[str, str]] = []
        addr_dict: dict[str, list[str]] = {}

        for block in self._memory.getBlocks():
            name = str(block.getName())
            if name not in sections:
                continue
            if name == ".rodata" and block.isInitialized() and not block.isWrite():
                size = int(block.getSize())
                if size <= 0:
                    continue
                start_addr = block.getStart()
                buf = bytearray(size)
                block.getBytes(start_addr, buf)
                data = bytes(buf)
                base_addr = int(start_addr.getOffset())

                for match in re.finditer(b"[\x20-\x7e]{4,}\x00", data):
                    s = match.group().rstrip(b"\x00").decode("utf-8", errors="ignore")
                    start = base_addr + match.start()
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
        return GhidraMetadataLookup(self._program, self._fm)

    def function_count(self) -> int:
        return int(self._fm.getFunctionCount())

    def iter_functions(self) -> Iterable[tuple[int, str, _CapFunction]]:
        """Iterate functions, yielding Capstone-compatible adapter objects.

        Each yielded ``func`` has a ``.blocks`` attribute containing
        ``_CapBlock`` objects whose ``.capstone.insns`` are
        ``_CapInstruction`` adapters — the same shape the tokenizer expects.
        """
        assert self._analyzed, "Analysis not run — call build_cfg() first"

        from ghidra.program.model.block import SimpleBlockModel
        from ghidra.util.task import TaskMonitor

        block_model = SimpleBlockModel(self._program)
        monitor = TaskMonitor.DUMMY

        funcs = []
        for func in self._fm.getFunctions(True):
            name = str(func.getName())
            addr = int(func.getEntryPoint().getOffset())
            funcs.append((addr, name, func))

        for addr, name, ghidra_func in sorted(funcs, key=lambda t: t[1]):
            body = ghidra_func.getBody()
            block_iter = block_model.getCodeBlocksContaining(body, monitor)

            cap_blocks: list[_CapBlock] = []
            while block_iter.hasNext():
                gblock = block_iter.next()
                block_start = int(gblock.getMinAddress().getOffset())
                block_size = int(gblock.getMaxAddress().getOffset()) - block_start + 1

                insns: list[_CapInstruction] = []
                insn_iter = self._listing.getInstructions(gblock, True)
                while insn_iter.hasNext():
                    ghidra_insn = insn_iter.next()
                    insn_addr = int(ghidra_insn.getAddress().getOffset())
                    if not body.contains(ghidra_insn.getAddress()):
                        continue
                    cap_insn = _ghidra_insn_to_cap(ghidra_insn, self._reg_map, self._program)
                    insns.append(cap_insn)

                cb = _CapBlock(addr=block_start, size=block_size)
                cb.set_insns(insns)
                cap_blocks.append(cb)

            if not cap_blocks:
                continue

            yield addr, name, _CapFunction(_blocks=cap_blocks)
