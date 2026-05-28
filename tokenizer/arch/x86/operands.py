import warnings
from typing import List

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.constant_handler import ConstantHandler
from tokenizer.disasm.types import (
    AddressSizePrefixView,
    BranchHintPrefixView,
    InstructionView,
    LockPrefixView,
    OperandSizePrefixView,
    OperandView,
    RepPrefixView,
    SegmentOverridePrefixView,
    X86BranchHint,
    X86Segment,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import MemoryOperandSymbol, Tokens
from tokenizer.utils import num_hex_digits

# Typed-prefix → token-name mapping. The string values mirror the v1
# `inv_prefix_tokens` byte-keyed table in ``arch/x86/data_store.json``
# (lock / rep / segment-override / op-size / addr-size); the pre-G.3
# consumer indexed that table by raw byte. ``BranchHintPrefixView`` did
# not exist in the legacy decode path - Capstone's branch-hint bytes
# (0x2E / 0x3E) were unconditionally classified as the CS / DS segment
# prefix and emitted as ``"cs:"`` / ``"ds:"``. We preserve that exact
# token output here so the migration does not silently retag any v1
# instruction.
_SEGMENT_TOKEN_NAMES: dict[X86Segment, str] = {
    X86Segment.CS: "cs:",
    X86Segment.SS: "ss:",
    X86Segment.DS: "ds:",
    X86Segment.ES: "es:",
    X86Segment.FS: "fs:",
    X86Segment.GS: "gs:",
}
_BRANCH_HINT_TOKEN_NAMES: dict[X86BranchHint, str] = {
    # 0x2E in branch context (was emitted as ``cs:`` pre-typed-prefixes)
    X86BranchHint.NOT_TAKEN: "cs:",
    # 0x3E in branch context (was emitted as ``ds:`` pre-typed-prefixes)
    X86BranchHint.TAKEN: "ds:",
}
# REP prefix → token-name dispatch. Both 0xF3 (repeat_until_zero=True)
# and 0xF2 (repeat_until_zero=False) have "degenerate" longer spellings
# that Capstone glues onto the mnemonic ("repe cmpsb", "repz scasw",
# etc.). When the mnemonic begins with one of the listed candidates we
# emit that exact word verbatim; otherwise we fall back to the canonical
# short form (``rep`` / ``repne``) recorded in the v1 prefix table.
# Order matters - longest match wins.
_REP_TRUE_CANDIDATES: tuple[str, ...] = ("repe", "repz", "rep")
_REP_FALSE_CANDIDATES: tuple[str, ...] = ("repne", "repnz")
_REP_TRUE_FALLBACK: str = "rep"
_REP_FALSE_FALLBACK: str = "repne"


def emit_x86_prefix_tokens(
    insn: InstructionView,
    vocab_manager: VocabularyManager,
) -> List[Tokens]:
    """Emit one ``PlatformToken`` per typed prefix on ``insn.prefixes``.

    Single-concern: this function owns the typed-prefix-to-token
    translation. Both the angr and Ghidra x86 providers call it; neither
    needs to know what the typed prefix subclasses are.

    Token mapping preserves the v1 ``inv_prefix_tokens`` table (see the
    module-level dispatch dicts above). REP prefixes consult the
    instruction's full mnemonic to recover any degenerate longer
    spelling Capstone glued on (``repe``/``repz``/``repnz``).
    """
    tokens: List[Tokens] = []
    mnemonic = insn.mnemonic
    for prefix in insn.prefixes:
        if isinstance(prefix, LockPrefixView):
            tokens.append(vocab_manager.PlatformToken("lock", PlatformInstructionTypes.PREFIXES))
        elif isinstance(prefix, RepPrefixView):
            if prefix.repeat_until_zero:
                candidates = _REP_TRUE_CANDIDATES
                name = _REP_TRUE_FALLBACK
            else:
                candidates = _REP_FALSE_CANDIDATES
                name = _REP_FALSE_FALLBACK
            for candidate in candidates:
                if mnemonic.startswith(candidate):
                    name = candidate
                    break
            tokens.append(vocab_manager.PlatformToken(name, PlatformInstructionTypes.PREFIXES))
        elif isinstance(prefix, SegmentOverridePrefixView):
            tokens.append(
                vocab_manager.PlatformToken(
                    _SEGMENT_TOKEN_NAMES[prefix.segment], PlatformInstructionTypes.PREFIXES
                )
            )
        elif isinstance(prefix, OperandSizePrefixView):
            tokens.append(
                vocab_manager.PlatformToken("operand_size_override", PlatformInstructionTypes.PREFIXES)
            )
        elif isinstance(prefix, AddressSizePrefixView):
            tokens.append(
                vocab_manager.PlatformToken("address_size_override", PlatformInstructionTypes.PREFIXES)
            )
        elif isinstance(prefix, BranchHintPrefixView):
            tokens.append(
                vocab_manager.PlatformToken(
                    _BRANCH_HINT_TOKEN_NAMES[prefix.hint], PlatformInstructionTypes.PREFIXES
                )
            )
    return tokens


SIZE_MAP: dict[int, str] = {
    1: "byte_ptr",
    2: "word_ptr",
    4: "dword_ptr",
    8: "qword_ptr",
    10: "xword_ptr",  # x87 extended precision (80-bit)
    14: "fpu_env_ptr",  # x87 environment (16-bit mode)
    16: "xmmword_ptr",
    28: "fpu_env_ptr",  # x87 environment (32-bit mode) - ADD THIS
    32: "ymmword_ptr",
    64: "zmmword_ptr",
    94: "fpu_state_ptr",  # x87 state (16-bit mode)
    108: "fpu_state_ptr",  # x87 state (32-bit mode)
}


def tokenize_operand_memory(
    insn: InstructionView,
    lookup,
    op: OperandView,
    text_end,
    text_start,
    func_max_addr,
    func_min_addr,
    vocab_manager: VocabularyManager,
    constant_handler: ConstantHandler,
) -> List[Tokens]:
    """
    Tokenize x86/x64 memory operand and return list of tokens.

    Returns:
        List of TokensRepl objects for this memory operand
    """
    tokens = []

    mem = op.mem
    base = mem.base
    index = mem.index
    segment = mem.segment
    scale = mem.scale
    disp = mem.disp
    resolved_target = mem.resolved_target

    has_reg = not base.is_absent
    has_index = not index.is_absent
    # Resolved-target case (Ghidra-only): when the provider's analyzer
    # has lifted a data address distinct from the operand's literal
    # disp, substitute it for classification so precedence step 7
    # (string_ptr) / step 9 (ro_data_ptr) fire. For typical x86
    # ``lea rsi, [rip+offset]`` Ghidra resolves the disp directly so
    # ``resolved_target`` is None and the existing disp-based lookup
    # path runs unchanged.
    has_resolved = resolved_target is not None
    classified_value = resolved_target if has_resolved else disp
    has_disp = classified_value != 0 or has_resolved

    if op.size in SIZE_MAP:
        tokens.append(vocab_manager.PlatformToken(SIZE_MAP[op.size], PlatformInstructionTypes.POINTER_LENGTHS))
    else:
        # Find the next bigger size in SIZE_MAP or take the largest available
        next_size = min((s for s in SIZE_MAP if s > op.size), default=max(SIZE_MAP))
        tokens.append(vocab_manager.PlatformToken(SIZE_MAP[next_size], PlatformInstructionTypes.POINTER_LENGTHS))
        warnings.warn(
            f"unexpected memory operand size: {op.size}, using next bigger '{SIZE_MAP[next_size]}' at {next_size}bytes for instruction {insn.mnemonic} {insn.op_str}"
        )

    if not segment.is_absent:
        tokens.append(
            vocab_manager.PlatformToken(
                f"{segment.name}:", PlatformInstructionTypes.MEMORY_ACCESS_MODE
            )
        )

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.OPEN_BRACKET))

    # Register the base and index registers
    if has_reg:
        tokens.append(vocab_manager.get_registry_token(base.name, base.id))

    if has_index:
        if has_reg:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))

        tokens.append(vocab_manager.get_registry_token(index.name, index.id))

    # Process scale as a constant if in expected range
    if scale != 1:
        assert scale > 0
        if has_index:
            tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.MULTIPLY))
            tokens.append(vocab_manager.ValuedConst(abs(scale)))
        else:
            warnings.warn(
                f"Scale {scale} used without index register in instruction "
                f"{insn.mnemonic} {insn.op_str}"
            )

    if has_disp and (has_reg or has_index):
        tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS))

    # Process displacement
    if not has_disp:
        pass  # noop ignore
    elif (
        classified_value <= 0xFF and not has_resolved
    ):  # if we are in range 00 to 0xFF we always use constant, same if we are negative as its defo not an addr
        # The resolved-target path always takes the metadata-aware
        # emitter below; a resolved string at e.g. 0x42 would otherwise
        # collapse to a bare ``valued_const`` and lose precedence step 7.
        # Sign is decomposed by the constant handler's ``_emit_valued_const``
        # (postfix ``value_negative``); ``is_arithmetic=True`` short-circuits
        # steps 2-10 so this stays a bare valued_const emission for positives.
        tokens.extend(
            constant_handler.process_constant_v2(classified_value, is_arithmetic=True)
        )

    else:
        force_opaque = False

        if has_disp and not has_reg:
            force_opaque = True

        elif classified_value > (1 << 18):  # = 262,144
            force_opaque = True

        elif has_resolved:
            # Provider-resolved address: skip the in-text / arithmetic
            # heuristic and let the metadata-aware emitter classify
            # directly. Without this, a resolved-target in the .text
            # range (e.g. a literal-pool slot itself) would route to
            # the arithmetic path.
            force_opaque = True

        meta = lookup.lookup(classified_value)

        # Memory operand → if the load is FP-typed Ghidra stamps the typed
        # ``fp_type`` on this OperandView at decode time; the v2 emitter
        # appends a postfix ``floatXX`` after the ptr token (precedence.md
        # "Postfix FP"). The angr path uniformly reports None
        # (``angr_limitations.md`` §1).
        fp_postfix = op.fp_type

        if force_opaque:
            disp_token = constant_handler.process_constant_v2(
                classified_value,
                meta=meta,
                is_arithmetic=False,
                fp_postfix_type=fp_postfix,
            )
            tokens.extend(disp_token)
        # For larger displacements, check if pointing to known constant or code or opaque.
        # Check if displacement is in text section or outside function bounds
        elif (text_start <= classified_value < text_end) or (classified_value < func_min_addr or classified_value > func_max_addr):
            disp_token = constant_handler.process_constant_v2(
                classified_value,
                meta=meta,
                is_arithmetic=False,
                fp_postfix_type=fp_postfix,
            )
            tokens.extend(disp_token)
        else:
            # Local constant - treat as valued constant literal
            disp_token = constant_handler.process_constant_v2(classified_value, is_arithmetic=True)
            tokens.extend(disp_token)

    tokens.append(vocab_manager.MemoryOperand(MemoryOperandSymbol.CLOSE_BRACKET))

    return tokens


def tokenize_operand_immediate(
    addressing_control_flow_instructions,
    arithmetic_instructions,
    insn: InstructionView,
    lookup,
    op: OperandView,
    func_max_addr,
    func_min_addr,
    constant_handler: ConstantHandler,
) -> List[Tokens]:
    """
    Tokenize x86/x64 immediate operand and return list of tokens.

    Returns:
        List of TokensRepl objects for this immediate operand
    """
    tokens = []

    raw_imm = op.imm
    # Width-only computation: hex-digit count is sign-agnostic and
    # selects the encoding-size branch, distinct from the value handed
    # to the constant-handler.
    imm_val_hex_len = num_hex_digits(abs(raw_imm))

    # Sign of ``raw_imm`` is owned downstream by
    # ``constant_handler.process_constant_v2`` (postfix ``value_negative``
    # metatoken on the ``valued_const_v2`` token). This call site does
    # NOT inspect sign — the value flows through to the emitter
    # unmodified.
    #
    # Immediate operand → an FP-tagged immediate (precedence.md step 1) is
    # an inline ``floatXX`` with IEEE bit pattern. Only the Ghidra path
    # stamps a non-None ``fp_type`` on the operand; the angr/Capstone path
    # uniformly reports None (see ``angr_limitations.md`` §1).
    fp_immediate = op.fp_type

    if imm_val_hex_len <= 2:  # Small immediate (0x00 to 0xFF)
        imm_token = constant_handler.process_constant_v2(
            raw_imm,
            fp_immediate_type=fp_immediate,
        )
        tokens.extend(imm_token)
    elif imm_val_hex_len <= (128 / 4):  # Larger immediate (up to 128-bit)
        if insn.mnemonic in arithmetic_instructions:
            # Arithmetic instruction - treat as valued constant literal
            imm_token = constant_handler.process_constant_v2(
                raw_imm,
                is_arithmetic=True,
                fp_immediate_type=fp_immediate,
            )
            tokens.extend(imm_token)
        elif insn.mnemonic in addressing_control_flow_instructions:
            # Control-flow target: hand the raw value + metadata to the v2
            # precedence walk and let ``_pred_block`` / ``_pred_local_func``
            # / extern / etc. decide. Intra-function targets land at step 4
            # (``block_v2``) per ``precedence.md``; function entries fall
            # to step 3 (``local_func``). No pre-classification here — the
            # constant_handler owns the address-kind dispatch.
            # todo we have a major issue here: a lot of targets are NOI in this table, e.g. I got .plt but angr can resolve it cfg.kb.functions.get(call_target_addr)
            meta = lookup.lookup(raw_imm)
            imm_token = constant_handler.process_constant_v2(
                raw_imm,
                meta=meta,
                is_arithmetic=False,
                fp_immediate_type=fp_immediate,
            )
            tokens.extend(imm_token)
        else:  # Fallback - create opaque constant
            meta = lookup.lookup(raw_imm)
            imm_token = constant_handler.process_constant_v2(
                raw_imm,
                meta=meta,
                is_arithmetic=False,
                fp_immediate_type=fp_immediate,
            )
            tokens.extend(imm_token)

    return tokens
