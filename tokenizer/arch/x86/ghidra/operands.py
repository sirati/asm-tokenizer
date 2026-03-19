"""Ghidra-native memory operand tokenization for x86/x64.

Tokenizes memory operands directly from Ghidra API objects (Register,
Scalar, Address) using getOpObjects() ordering semantics — no string parsing.

Object-count rules for getOpObjects():
    2 general registers  -> first Scalar = scale, remaining Scalars = disp
    0-1 general registers -> all Scalars = disp
    Address objects       -> disp
"""

import warnings
from typing import Any, List

from tokenizer.arch.x86.operands import SIZE_MAP
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, Tokens

_SEGMENT_REGISTERS = frozenset({"fs", "gs", "cs", "ds", "es", "ss"})


def _infer_size_from_ghidra_insn(ghidra_insn: Any, default: int = 8) -> int:
    """Infer memory operand size from sibling register operands.

    Checks getResultObjects() then getInputObjects() for general-purpose
    Registers and returns the largest getMinimumByteSize().  This avoids
    picking up 1-byte flags registers (CF, PF, ZF, SF, OF) that appear
    in result objects for arithmetic instructions.
    """
    from ghidra.program.model.lang import Register

    max_size = 0
    for source in (ghidra_insn.getResultObjects(), ghidra_insn.getInputObjects()):
        if source is None:
            continue
        for obj in source:
            if isinstance(obj, Register):
                name = str(obj.getName()).lower()
                if name not in _SEGMENT_REGISTERS:
                    size = int(obj.getMinimumByteSize())
                    if size > max_size:
                        max_size = size
    return max_size if max_size > 0 else default


def tokenize_operand_memory_ghidra(
    ghidra_raw_data: Any,
    insn: Any,
    lookup: Any,
    text_end: int,
    text_start: int,
    func_max_addr: int,
    func_min_addr: int,
    vocab_manager: VocabularyManager,
    constant_handler: ConstantHandler,
) -> List[Tokens]:
    """Tokenize x86/x64 memory operand from raw Ghidra objects.

    Produces the same token sequence as the angr/Capstone path:
        size_ptr [segment:] mem[ base [+ index [* scale]] [+/- disp] ]mem
    """
    from ghidra.program.model.address import Address
    from ghidra.program.model.lang import Register
    from ghidra.program.model.scalar import Scalar

    tokens: list[Tokens] = []
    ghidra_insn = ghidra_raw_data.ghidra_insn
    objects = ghidra_raw_data.op_objects
    reg_map = ghidra_raw_data.reg_map

    # -- Classify objects from getOpObjects() ---------------------------------
    segment_reg_id: int = 0
    general_regs: list[int] = []  # register IDs, ordered (first=base, second=index)
    scalars: list[int] = []  # unsigned values, ordered
    signed_scalars: list[int] = []  # signed values, same order
    disp: int = 0

    for obj in objects:
        if isinstance(obj, Register):
            name = str(obj.getName()).lower()
            rid = reg_map.get_id(name)
            if name in _SEGMENT_REGISTERS:
                segment_reg_id = rid
            else:
                general_regs.append(rid)
        elif isinstance(obj, Scalar):
            scalars.append(int(obj.getValue()))
            signed_scalars.append(int(obj.getSignedValue()))
        elif isinstance(obj, Address):
            disp = int(obj.getOffset())

    # -- Assign base, index, scale, disp using object-count rules -------------
    base: int = general_regs[0] if len(general_regs) >= 1 else 0
    index: int = general_regs[1] if len(general_regs) >= 2 else 0
    scale: int = 1

    if len(general_regs) >= 2 and scalars:
        # 2 registers: first Scalar is scale, remaining are displacement
        scale = scalars[0]
        if len(scalars) > 1:
            disp = signed_scalars[1]
    elif len(general_regs) <= 1 and scalars:
        # 0-1 registers: all Scalars are displacement
        disp = signed_scalars[0]

    has_base = base != 0
    has_index = index != 0
    has_disp = disp != 0

    # -- Size inference -------------------------------------------------------
    mem_size = _infer_size_from_ghidra_insn(ghidra_insn)

    if mem_size in SIZE_MAP:
        tokens.append(vocab_manager.PlatformToken(SIZE_MAP[mem_size], PlatformInstructionTypes.POINTER_LENGTHS))
    else:
        next_size = min((s for s in SIZE_MAP if s > mem_size), default=max(SIZE_MAP))
        tokens.append(vocab_manager.PlatformToken(SIZE_MAP[next_size], PlatformInstructionTypes.POINTER_LENGTHS))
        warnings.warn(f"unexpected memory operand size: {mem_size}, using '{SIZE_MAP[next_size]}'")

    # -- Segment register -----------------------------------------------------
    if segment_reg_id > 0:
        seg_name = insn.reg_name(segment_reg_id)
        tokens.append(vocab_manager.PlatformToken(f"{seg_name}:", PlatformInstructionTypes.MEMORY_ACCESS_MODE))

    # -- Open bracket ---------------------------------------------------------
    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.OPEN_BRACKET))

    # -- Base register --------------------------------------------------------
    if has_base:
        tokens.append(vocab_manager.get_registry_token(insn.reg_name(base), base))

    # -- Index register -------------------------------------------------------
    if has_index:
        if has_base:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))
        tokens.append(vocab_manager.get_registry_token(insn.reg_name(index), index))

    # -- Scale ----------------------------------------------------------------
    if scale != 1:
        assert scale > 0
        if has_index:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MULTIPLY))
            tokens.append(vocab_manager.Valued_Const(abs(scale)))
        else:
            warnings.warn(f"Scale {scale} used without index register")

    # -- Displacement sign ----------------------------------------------------
    if disp < 0:
        tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS))
    elif has_disp and (has_base or has_index):
        tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))

    # -- Displacement value (same logic as angr path) -------------------------
    if not has_disp:
        pass
    elif disp <= 0xFF:
        tokens.append(vocab_manager.Valued_Const(abs(disp)))
    else:
        force_opaque = False

        if has_disp and not has_base:
            force_opaque = True
        elif disp > (1 << 18):
            force_opaque = True

        meta, kind = lookup.lookup(disp)

        if force_opaque:
            disp_token = constant_handler.process_constant(
                disp,
                is_arithmetic=False,
                meta=meta,
                library_type=meta.get("library", "unknown") if meta else "unknown",
                insn_mnemonic=insn.mnemonic,
            )
            tokens.extend(disp_token)
        elif meta is not None:
            if (text_start <= disp < text_end) or (disp < func_min_addr or disp > func_max_addr):
                disp_token = constant_handler.process_constant(
                    disp,
                    is_arithmetic=False,
                    meta=meta,
                    library_type=meta.get("library", "unknown"),
                    insn_mnemonic=insn.mnemonic,
                )
                tokens.extend(disp_token)
            else:
                disp_token = constant_handler.process_constant(disp, is_arithmetic=True, insn_mnemonic=insn.mnemonic)
                tokens.extend(disp_token)
        else:
            disp_token = constant_handler.process_constant(disp, is_arithmetic=True, insn_mnemonic=insn.mnemonic)
            tokens.extend(disp_token)

    # -- Close bracket --------------------------------------------------------
    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.CLOSE_BRACKET))

    return tokens
