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

from typing import Any

from tokenizer.disasm.ghidra_provider.decode_helper.decompose_callbacks import (
    is_sleigh_split_disp_base_pair,
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
from tokenizer.disasm.ghidra_provider.pcode_inspect import (
    InstructionPcodeSummary,
    collect_instruction_pcode_summary,
)
from tokenizer.disasm.ghidra_provider.prefix_build import (
    _ghidra_processor_to_architecture,
    _prefix_builder_for_arch,
)
from tokenizer.disasm.types import (
    Architecture,
    ArmConditionCode,
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
      - per-instruction one-pass PCode summary (``pcode_summary``:
        LOAD/STORE presence + FLOAT_* signature + mem-access size,
        computed once per cursor advance and threaded into every
        ``operand_spec`` call)
      - per-instruction typed-prefix list build
      - per-operand decompose-mem callback construction (lazy: returns a
        zero-arg callable that, when invoked, populates a passed
        ``_GhidraMemoryOperandView``)
      - per-operand spec dict (kwargs ready for
        ``_GhidraOperandView._advance``)
    """

    __slots__ = ("_program", "_reg_map", "_arch", "_debug_render")

    def __init__(
        self, program: Any, reg_map: "_RegisterMap", debug_render: bool = False
    ) -> None:
        self._program = program
        self._reg_map = reg_map
        self._arch: Architecture = _ghidra_processor_to_architecture(program)
        self._debug_render = debug_render

    @property
    def arch(self) -> Architecture:
        return self._arch

    @property
    def debug_render(self) -> bool:
        """Run-scoped debug-rendering mode (``--debug`` CLI runs).

        When False (production), the instruction views withhold the
        per-instruction operand-text rendering entirely: ``debug_label``
        hands out non-rendering labels and the per-operand
        ``getDefaultOperandRepresentation`` JVM round-trips never run on
        the decode path.
        """
        return self._debug_render

    def split_mnemonic(self, raw: str) -> tuple[str, str | None, int | None]:
        return _split_ghidra_mnemonic(raw)

    def strip_arm_cc(
        self,
        mnemonic: str,
    ) -> tuple[str, ArmConditionCode | None]:
        """Strip the Ghidra ARM/AArch64 cc-suffix from ``mnemonic``."""
        return _strip_arm_cc_suffix(mnemonic)

    def pcode_summary(self, ghidra_insn: Any) -> InstructionPcodeSummary:
        """Walk the instruction's PCode ONCE and return the bundled
        instruction-level signals (LOAD/STORE presence, FLOAT_*
        signature, mem-access size). Computed once per cursor advance
        by ``_GhidraInstructionView._advance`` and threaded into every
        ``operand_spec`` call; ``summary.has_load_store`` also feeds
        ``InstructionView.has_load_store`` for the downstream
        resolved-target keep/drop policy.
        """
        return collect_instruction_pcode_summary(ghidra_insn)

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

    def build_prefixes(self, ghidra_insn: Any, arch: Architecture) -> list[Any]:
        return _prefix_builder_for_arch(arch)(ghidra_insn)

    def is_sleigh_split_disp_base_pair(
        self,
        disp_spec: dict,
        base_spec: dict,
    ) -> bool:
        """True iff two ADJACENT operand specs are the SLEIGH-split
        disp(base) pair ``synthesize_disp_base_mem_spec`` fuses. Keeps
        the ``OperandType`` bitmask knowledge behind the decode facade
        so the instruction cursor's merge loop stays JVM-class-free.
        """
        return is_sleigh_split_disp_base_pair(disp_spec, base_spec)

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
        pcode_summary: InstructionPcodeSummary,
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
            pcode_summary=pcode_summary,
        )
