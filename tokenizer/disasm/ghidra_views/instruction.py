"""Instruction cursor + its operand/prefix container views.

Owns:
- ``_GhidraInstructionView``: reusable instruction wrapper.
- ``_GhidraOperandsView``: container view over an instruction's operands.
- ``_GhidraPrefixesView``: container view over an instruction's prefixes.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from tokenizer.disasm.ghidra_views.operand import _GhidraOperandView
from tokenizer.disasm.types import (
    Architecture,
    InstructionPrefixView,
    OperandsView,
    OperandView,
    PrefixesView,
)


# ---------------------------------------------------------------------------
# Container views (NOT lists - iterate, reuse child wrapper)
# ---------------------------------------------------------------------------
class _GhidraOperandsView:
    """Container view over an instruction's operands.

    NOT a list. ``__iter__`` advances the parent ``_GhidraInstructionView``'s
    reusable ``_GhidraOperandView`` per operand and yields the same wrapper
    instance each time. ``__getitem__`` is intentionally absent (random
    access would invalidate adjacent reads).
    """

    __slots__ = ("_instruction",)

    def __init__(self, instruction: "_GhidraInstructionView") -> None:
        self._instruction = instruction

    def __len__(self) -> int:
        return self._instruction._operand_count

    def __iter__(self) -> Iterator[OperandView]:
        return self._instruction._iter_operands()


class _GhidraPrefixesView:
    """Container view over an instruction's prefixes.

    Per the protocol: prefix instances are typed-distinct and few
    (0-3 typical), so ``__getitem__`` is supported and the underlying
    storage is just a list. The instruction wrapper rebuilds this list
    when its cursor advances.
    """

    __slots__ = ("_prefixes",)

    def __init__(self) -> None:
        self._prefixes: list[InstructionPrefixView] = []

    def _populate(self, prefixes: list[InstructionPrefixView]) -> None:
        self._prefixes = prefixes

    def __len__(self) -> int:
        return len(self._prefixes)

    def __iter__(self) -> Iterator[InstructionPrefixView]:
        return iter(self._prefixes)

    def __getitem__(self, idx: int) -> InstructionPrefixView:
        return self._prefixes[idx]


# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------
class _GhidraInstructionView:
    """Reusable instruction wrapper.

    Holds a single backing ``Ghidra Instruction`` reference at any time.
    ``_advance(ghidra_insn)`` repoints at the next instruction and resets
    per-cursor caches (mnemonic split, prefixes list, ISA, operand spec
    lazily computed on iter_operands).

    The wrapper depends on a ``decode_helper`` injected by the provider;
    that helper exposes:
      - ``split_mnemonic(raw)`` -> (base, suffix_prefix_name, suffix_prefix_byte)
      - ``alias_mnemonic(base)`` -> canonical base
      - ``architecture(program)`` -> Architecture (cached on the view)
      - ``compute_fp_type(insn, op_idx, arch, base_mnemonic)`` -> Optional[FpType]
      - ``build_prefixes(insn, arch)`` -> list[InstructionPrefixView]
      - ``decompose_x86_memory(insn, op_idx, reg_map)`` -> callback that
        populates a passed-in _GhidraMemoryOperandView
      - ``decompose_arm_memory(insn, op_idx, reg_map)`` -> callback
      - ``decompose_base_disp_memory(insn, op_idx, reg_map)`` -> callback
      - ``operand_spec(insn, op_idx, arch, base_mnemonic, reg_map)``
        -> dict ready to pass as kwargs to ``_GhidraOperandView._advance``
    """

    __slots__ = (
        "_arch",
        "_program",
        "_reg_map",
        "_decode",
        "_ghidra_insn",
        "_address",
        "_mnemonic",
        "_base_mnemonic",
        "_op_str",
        "_operand_count",
        "_op_specs",
        "_prefixes",
        "_operand_view",
        "_operands_view",
    )

    def __init__(
        self,
        arch: Architecture,
        program: Any,
        reg_map: Any,
        decode: Any,
    ) -> None:
        self._arch: Architecture = arch
        self._program = program
        self._reg_map = reg_map
        self._decode = decode
        self._ghidra_insn: Optional[Any] = None
        self._address: int = 0
        self._mnemonic: str = ""
        self._base_mnemonic: str = ""
        self._op_str: str = ""
        self._operand_count: int = 0
        # Per-instruction operand specs, computed eagerly in ``_advance``
        # so we can pair-merge SLEIGH-split disp+base operands before
        # iteration (e.g. RISC-V c.sdsp / c.lwsp variants). Iteration
        # walks this list rather than re-asking the decode helper.
        self._op_specs: list[dict] = []
        self._prefixes = _GhidraPrefixesView()
        self._operand_view = _GhidraOperandView(arch, reg_map)
        self._operands_view = _GhidraOperandsView(self)

    def _advance(self, ghidra_insn: Any) -> None:
        """Repoint at the next Ghidra Instruction.

        Resets per-cursor state (mnemonic split, op_str, operand count,
        prefixes list). Operand decomposition is deferred to ``__iter__``
        on the operands view.
        """
        self._ghidra_insn = ghidra_insn
        self._address = int(ghidra_insn.getAddress().getOffset())

        raw_mnemonic = str(ghidra_insn.getMnemonicString())
        base_mnemonic, suffix_prefix_name, _suffix_prefix_byte = self._decode.split_mnemonic(raw_mnemonic)
        base_mnemonic = self._decode.alias_mnemonic(base_mnemonic)
        self._base_mnemonic = base_mnemonic
        if suffix_prefix_name is not None:
            self._mnemonic = f"{suffix_prefix_name} {base_mnemonic}"
        else:
            self._mnemonic = base_mnemonic

        # op_str: comma-joined per-operand default representation
        try:
            num_ops = int(ghidra_insn.getNumOperands())
        except Exception:
            num_ops = 0
        op_strs: list[str] = []
        for i in range(num_ops):
            try:
                op_strs.append(str(ghidra_insn.getDefaultOperandRepresentation(i)))
            except Exception:
                op_strs.append("")
        self._op_str = ", ".join(op_strs)

        # Eager operand-spec decode + SLEIGH disp+base pair merge. Done
        # here (rather than lazily inside ``_iter_operands``) because the
        # pair-merge collapses two source operands into one synthetic
        # MEM operand: ``len(insn.operands)`` (which reads
        # ``self._operand_count``) must reflect the post-merge count, so
        # we cannot defer until iteration.
        #
        # Disp+base pairs occur on RISC-V compressed-instruction
        # encodings (``c.sdsp ra, 0x8(sp)``) where Ghidra's SLEIGH spec
        # reports the disp scalar and the base register as adjacent
        # flat operands (op[1]=DYNAMIC scalar 0x8, op[2]=REGISTER sp)
        # instead of bundling them into one composite memory operand.
        from ghidra.program.model.lang import OperandType
        from tokenizer.disasm.types import OperandKind as _OperandKind

        raw_specs = [
            self._decode.operand_spec(
                ghidra_insn,
                i,
                self._arch,
                self._base_mnemonic,
                self._reg_map,
            )
            for i in range(num_ops)
        ]
        merged_specs: list[dict] = []
        i = 0
        while i < len(raw_specs):
            cur = raw_specs[i]
            if (
                i + 1 < len(raw_specs)
                and cur["kind"] == _OperandKind.IMM
                and bool(int(cur["type_int"]) & OperandType.DYNAMIC)
                and raw_specs[i + 1]["kind"] == _OperandKind.REG
            ):
                merged_specs.append(
                    self._decode.synthesize_disp_base_mem_spec(
                        cur, raw_specs[i + 1]
                    )
                )
                i += 2
            else:
                merged_specs.append(cur)
                i += 1
        self._op_specs = merged_specs
        self._operand_count = len(merged_specs)

        # Prefixes (typed list) - rebuilt fresh per instruction; they are
        # typed-distinct, low-count instances so the small allocation is
        # acceptable per the protocol contract.
        prefixes = self._decode.build_prefixes(ghidra_insn, self._arch)
        self._prefixes._populate(prefixes)

    def _iter_operands(self) -> Iterator[OperandView]:
        """Yield the reusable ``_GhidraOperandView`` for each operand of
        the current instruction. The same wrapper instance is yielded
        each time, mutated to point at the next operand. Specs are
        pre-computed (and pair-merged) by ``_advance``."""
        if self._ghidra_insn is None:
            return
        op_view = self._operand_view
        for spec in self._op_specs:
            op_view._advance(**spec)
            yield op_view

    @property
    def address(self) -> int:
        return self._address

    @property
    def mnemonic(self) -> str:
        return self._mnemonic

    @property
    def base_mnemonic(self) -> str:
        return self._base_mnemonic

    @property
    def op_str(self) -> str:
        return self._op_str

    @property
    def operands(self) -> OperandsView:
        return self._operands_view

    @property
    def prefixes(self) -> PrefixesView:
        return self._prefixes

    def __deepcopy__(self, memo) -> "_GhidraInstructionView":
        clone = _GhidraInstructionView(self._arch, self._program, self._reg_map, self._decode)
        clone._ghidra_insn = self._ghidra_insn
        clone._address = self._address
        clone._mnemonic = self._mnemonic
        clone._base_mnemonic = self._base_mnemonic
        clone._op_str = self._op_str
        clone._operand_count = self._operand_count
        # Carry the pre-computed (already pair-merged) operand specs so
        # the clone iterates the same operand sequence without re-decoding.
        # Spec dicts hold closures over the stable ghidra_insn Java handle
        # so reference-sharing the list is safe; deep-copying the dicts
        # would also deep-copy the closures (no value gain).
        clone._op_specs = list(self._op_specs)
        # Snapshot the prefix list (each prefix instance is itself
        # immutable per typed protocol).
        clone._prefixes._populate(list(self._prefixes._prefixes))
        # The operand_view + operands_view are fresh empty wrappers;
        # iterating them re-decodes lazily against the snapshotted
        # ghidra_insn (still a stable Java handle).
        return clone
