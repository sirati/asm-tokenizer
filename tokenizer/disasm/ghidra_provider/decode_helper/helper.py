"""Provider-owned decode helper injected into ``_GhidraInstructionView`` wrappers.

Owns ``_GhidraDecodeHelper``: the per-instruction decode facade the
owned-view wrappers (``ghidra_views``) consume. The heavy concerns it
exposes are delegated to focused submodules:
- mnemonic split + alias canonicalization (``mnemonic``)
- architecture detection + prefix build + FP-type (``prefix_build``)
- per-operand classification (``operand_classify.operand_spec``)
- lazy decompose callbacks + SLEIGH-split fusion
  (``decompose_callbacks``).
"""

from __future__ import annotations

from typing import Any, Optional

from tokenizer.disasm.ghidra_provider.decode_helper.decompose_callbacks import (
    synthesize_disp_base_mem_spec,
)
from tokenizer.disasm.ghidra_provider.decode_helper.operand_classify import (
    operand_spec,
)
from tokenizer.disasm.ghidra_provider.mnemonic import (
    _GHIDRA_MNEMONIC_ALIASES,
    _RegisterMap,
    _split_ghidra_mnemonic,
    _strip_arm_cc_suffix,
)
from tokenizer.disasm.ghidra_provider.pcode_inspect import has_load_store
from tokenizer.disasm.ghidra_provider.prefix_build import (
    _compute_fp_type,
    _ghidra_processor_to_architecture,
    _prefix_builder_for_arch,
)
from tokenizer.disasm.types import (
    Architecture,
    ArmConditionCode,
    FpType,
)


# ---------------------------------------------------------------------------
# Decode helper - injected into _GhidraInstructionView wrappers
# ---------------------------------------------------------------------------
class _GhidraDecodeHelper:
    """Provider-owned helper exposing the per-instruction decode surface
    the owned-view wrappers (``ghidra_views.py``) need.

    Construction: one instance per ``GhidraDisassemblyProvider`` (program
    + reg_map are stable for the program's lifetime). The helper is
    passed to each ``_GhidraFunctionView`` -> ``_GhidraBlockView`` ->
    ``_GhidraInstructionView`` constructor so the view chain stays
    self-contained without cross-importing the provider.

    The helper centralizes:
      - mnemonic split + alias canonicalization
      - architecture detection (cached per program)
      - per-instruction has-LOAD/STORE PCode signal (computed once,
        cached at the cursor level via ``InstructionView.has_load_store``)
      - per-operand FP-type computation
      - per-instruction typed-prefix list build
      - per-operand decompose-mem callback construction (lazy: returns a
        zero-arg callable that, when invoked, populates a passed
        ``_GhidraMemoryOperandView``)
      - per-operand spec dict (kwargs ready for
        ``_GhidraOperandView._advance``)
    """

    __slots__ = ("_program", "_reg_map", "_arch")

    def __init__(self, program: Any, reg_map: "_RegisterMap") -> None:
        self._program = program
        self._reg_map = reg_map
        self._arch: Architecture = _ghidra_processor_to_architecture(program)

    @property
    def arch(self) -> Architecture:
        return self._arch

    def split_mnemonic(self, raw: str) -> tuple[str, str | None, int | None]:
        return _split_ghidra_mnemonic(raw)

    def strip_arm_cc(
        self,
        mnemonic: str,
    ) -> tuple[str, ArmConditionCode | None]:
        """Strip the Ghidra ARM/AArch64 cc-suffix from ``mnemonic``."""
        return _strip_arm_cc_suffix(mnemonic)

    def has_load_store(self, ghidra_insn: Any) -> bool:
        """Return ``True`` when the instruction's PCode contains a LOAD or
        STORE op. Cached at the ``_GhidraInstructionView`` cursor level
        (``InstructionView.has_load_store``); the per-instruction signal
        feeds both the operand-classifier inside ``operand_spec`` and
        the downstream resolved-target keep/drop policy.
        """
        return has_load_store(ghidra_insn)

    def alias_mnemonic(self, base: str) -> str:
        # Ghidra's MIPS SLEIGH spec emits `_sra` / `_li` / ... for the
        # delay-slot variants of `sra` / `li` / .... The underscore is a
        # Ghidra display convention, not a real MIPS-ISA mnemonic
        # distinction (gas/objdump/Capstone all just write `sra`).
        # Delay-slot membership is already encoded positionally by the
        # token sequence, so the duplicate vocab entry is noise.
        if self._arch == Architecture.MIPS and base.startswith("_") and len(base) > 1:
            base = base[1:]
        return _GHIDRA_MNEMONIC_ALIASES.get(base, base)

    def architecture(self, _program: Any) -> Architecture:
        return self._arch

    def compute_fp_type(
        self,
        ghidra_insn: Any,
        operand_index: int,
        arch: Architecture,
        base_mnemonic: str,
    ) -> Optional[FpType]:
        return _compute_fp_type(ghidra_insn, operand_index, arch, base_mnemonic)

    def build_prefixes(self, ghidra_insn: Any, arch: Architecture) -> list[Any]:
        return _prefix_builder_for_arch(arch)(ghidra_insn)

    def synthesize_disp_base_mem_spec(
        self,
        disp_spec: dict,
        base_spec: dict,
    ) -> dict:
        """Synthesize a MEM operand spec from a (disp IMM, base REG) pair.

        Used when Ghidra's SLEIGH spec splits a disp(base) memory
        operand into two adjacent flat operands - notably the RISC-V
        compressed-instruction encodings (``c.sdsp ra, 0x8(sp)`` is
        reported as 3 operands). Delegates to ``decompose_callbacks``.
        """
        return synthesize_disp_base_mem_spec(disp_spec, base_spec)

    def operand_spec(
        self,
        ghidra_insn: Any,
        op_idx: int,
        arch: Architecture,
        base_mnemonic: str,
        reg_map: "_RegisterMap",
        *,
        instruction_has_mem_access: bool,
    ) -> dict:
        """Return a kwargs dict for ``_GhidraOperandView._advance``.

        Thin facade over ``operand_classify.operand_spec``; threads the
        helper-cached ``self._reg_map`` into the lazy decompose
        callbacks (the original wiring) while passing the explicit
        ``reg_map`` through for the REG-id lookups.
        """
        return operand_spec(
            self._reg_map,
            ghidra_insn,
            op_idx,
            arch,
            base_mnemonic,
            reg_map,
            instruction_has_mem_access=instruction_has_mem_access,
        )
