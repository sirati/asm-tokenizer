"""v1 legacy ``process_constant`` flow + removed-aggregation stubs.

``_V1LegacyMixin`` holds the heuristic-driven Block / Opaque /
Valued_Const dispatch + opaque-const helpers + the removed-aggregation
surface (stubs that raise ``NotImplementedError``). Composed into
``ConstantHandler`` via subclassing in ``core.py``.

The mixin assumes the composed subclass exposes:
- ``self.vocab_manager`` (``VocabularyManager``)
- ``self.resolver`` (``TokenResolver``)
- ``self.constant_dict`` (``Dict[str, List[str]]``)
- ``self.block_ranges`` (``numpy.ndarray``)
- ``self.opaque_const_tokens`` / ``self.opaque_const_usage`` /
  ``self.opaque_metadata`` (v1 in-memory state).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from tokenizer.disasm.metadata import AddressKind, AddressMetadataView
from tokenizer.tokens import MemoryOperandSymbol, Tokens

logger = logging.getLogger(__name__)


class _V1LegacyMixin:
    """v1 legacy entry + opaque-const helpers + removed-aggregation stubs."""

    # v1 mapping from typed ``AddressKind`` back to the legacy decision
    # vocabulary. Used by ``_create_opaque_const_with_offset`` to keep the
    # heuristic predicates (function-range guard, decomposable-type list)
    # readable without spelling out the AddressKind set inline.
    _V1_FUNCTION_KINDS = frozenset({
        AddressKind.LOCAL_FUNCTION,
        AddressKind.EXT_FUNCTION_REAL,
        AddressKind.EXT_FUNCTION_SYNTHETIC,
        AddressKind.PLT_FUNCTION,
        AddressKind.UNKNOWN,  # legacy "unknown_function" mapped to UNKNOWN
    })
    _V1_DECOMPOSABLE_KINDS = frozenset({
        AddressKind.DATA,
        AddressKind.RODATA,
        AddressKind.BSS,
        # legacy "code" was decomposable; AddressKind.UNKNOWN covers it
        # (provider mapping for "code" -> UNKNOWN per metadata.py).
        AddressKind.UNKNOWN,
    })

    # ----------------------------------------------------------------------
    # v1 legacy entry point -- preserved verbatim for callers that haven't
    # migrated to ``process_constant_v2``. Phase 1.C.3 (task #10) replaces
    # every call site; this method goes away then.
    # ----------------------------------------------------------------------

    def process_constant(
        self,
        value: int,
        is_arithmetic: bool = False,
        meta: Optional[AddressMetadataView] = None,
        library_type: str = "unknown",
        insn_mnemonic: Optional[str] = None,
    ) -> List[Tokens]:
        """v1 legacy entry -- heuristic-driven Block / Opaque / Valued_Const.

        See ``process_constant_v2`` for the v2 successor. The decision
        logic is byte-for-byte the pre-rewrite behavior; only the read
        API has been migrated from dict-shaped ``meta`` to typed
        ``AddressMetadataView`` (Phase D.3).
        """
        # Metadata ranges starting at 0 represent abstract constant
        # domains, not real memory segments -- treat as arithmetic.
        if meta is not None and meta.start_addr == 0:
            is_arithmetic = True

        # Small-constant / arithmetic short-circuit (legacy 0..0xFF rule).
        if is_arithmetic or 0x00 <= value <= 0xFF or value in self.constant_dict:
            return [self.vocab_manager.Valued_Const(value)]

        match_mask = (self.block_ranges[:, 0] <= value) & (value < self.block_ranges[:, 1])
        if np.any(match_mask):
            idx = match_mask.nonzero()[0][0]
            if self.block_ranges[idx, 0] == value:
                return [self.vocab_manager.Block(idx)]
            return [
                self.vocab_manager.Block(idx),
                self.vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS),
                self.vocab_manager.Valued_Const(value - self.block_ranges[idx, 0]),
            ]
        return self._create_opaque_const_with_offset(value, meta, library_type, insn_mnemonic)

    def _create_opaque_const_with_offset(
        self,
        value: int,
        meta: Optional[AddressMetadataView] = None,
        library_type: str = "unknown",
        insn_mnemonic: Optional[str] = None,
    ) -> List[Tokens]:
        """Create an opaque constant token, decomposing into base+offset if pointing into a range."""
        if (
            meta is not None
            and meta.start_addr is not None
            and meta.end_addr is not None
            and value > meta.start_addr
        ):
            start_addr = meta.start_addr
            end_addr = meta.end_addr
            range_length = end_addr - start_addr
            offset = value - start_addr

            # Apply heuristics to decide if we should decompose
            should_decompose = True
            reason = ""

            # Heuristic 5: Don't decompose local_function / library_function /
            # unknown_function ranges (they should be exact).
            if meta.kind in self._V1_FUNCTION_KINDS:
                should_decompose = False
                reason = f"function range (kind={meta.kind.name})"

            # Heuristic 3: Call instructions should not be decomposed
            # (must point to function header).
            elif insn_mnemonic and insn_mnemonic.startswith("call"):
                should_decompose = False
                reason = "call instruction must point to function header"

            # Heuristic 1: Range cannot start at 0 (not in binary).
            elif start_addr == 0:
                should_decompose = False
                reason = "range starts at 0"

            # Heuristic 2: Range cannot be longer than 2^16.
            elif range_length > (1 << 16):
                should_decompose = False
                reason = f"range too large (length={range_length:#x} > 0x10000)"

            # Heuristic 6: Prefer decomposing data/rodata/bss sections.
            elif meta.kind not in self._V1_DECOMPOSABLE_KINDS:
                should_decompose = False
                reason = f"unexpected metadata kind: {meta.kind.name}"

            # If value points into the range (not at the start), consider decomposition.
            if start_addr < value < end_addr and should_decompose:
                insn_info = f" in {insn_mnemonic}" if insn_mnemonic else ""
                logger.debug(
                    f"Decomposing: range {start_addr:#x}-{end_addr:#x} (length={range_length:#x}, kind={meta.kind.name}) "
                    f"for target {value:#x}, offset={offset:#x}{insn_info}"
                )
                base_token = self._create_opaque_const(start_addr, meta, library_type)
                return [
                    self.vocab_manager.MemoryOperand.OPEN_BRACKET,
                    base_token,
                    self.vocab_manager.MemoryOperand.PLUS,
                    self.vocab_manager.Valued_Const(offset),
                    self.vocab_manager.MemoryOperand.CLOSE_BRACKET,
                ]
            elif start_addr < value < end_addr and not should_decompose:
                insn_info = f" in {insn_mnemonic}" if insn_mnemonic else ""
                logger.debug(
                    f"Skipping decomposition: range {start_addr:#x}-{end_addr:#x} (length={range_length:#x}, kind={meta.kind.name}) "
                    f"for target {value:#x}, offset={offset:#x}{insn_info}, reason: {reason}"
                )

        # Otherwise just create a simple opaque constant.
        return [self._create_opaque_const(value, meta, library_type)]

    def _create_opaque_const(
        self,
        value: int,
        meta: Optional[AddressMetadataView] = None,
        library_type: str = "unknown",
    ) -> Tokens:
        """Create an opaque constant token (v1)."""
        if value not in self.opaque_const_tokens:
            opaque_id = self.resolver.get_opaque_id(value)
            token = self.vocab_manager.Opaque_Const(opaque_id)
            self.opaque_const_tokens[value] = token
            self.opaque_const_usage[value] = 1

            if meta is not None:
                # Mirrors the v1 tuple shape; ``library_type`` is the
                # caller-supplied label (kept separate from the typed
                # view's ``library`` because v1 callers passed
                # heuristic library tags here).
                self.opaque_metadata[value] = (
                    hex(meta.start_addr) if meta.start_addr is not None else "",
                    hex(meta.end_addr) if meta.end_addr is not None else "",
                    meta.name,
                    meta.kind.name,
                    library_type,
                )
        else:
            self.opaque_const_usage[value] += 1

        return self.opaque_const_tokens[value]

    # ----------------------------------------------------------------------
    # Removed v1 aggregation surface -- stubs raising NotImplementedError so
    # downstream callers (opaque_remapping.py, main_loop.py) fail loudly
    # rather than silently producing wrong output. Phase 2.A.1 / 2.B.7
    # rewires those callers to read the per-category metadata directly
    # off ``TokenResolver.metadata[category]``.
    # ----------------------------------------------------------------------

    def get_sorted_opaque_constants(self):
        raise NotImplementedError(
            "ConstantHandler.get_sorted_opaque_constants removed in v2 -- "
            "v2 identity is monotonic per category (no usage-frequency sort). "
            "See Phase 2.A.1 (CSV writer) / 2.B.7 (opaque_remapping migration)."
        )

    def create_opaque_mapping(self):
        raise NotImplementedError(
            "ConstantHandler.create_opaque_mapping removed in v2 -- "
            "v2 identity is monotonic per category (no usage-frequency sort). "
            "See Phase 2.B.7 (opaque_remapping migration)."
        )

    def get_usage_stats(self):
        raise NotImplementedError(
            "ConstantHandler.get_usage_stats removed in v2 -- "
            "usage tracking is not part of the v2 design. "
            "See Phase 2.A.1 (CSV writer)."
        )

    def get_metadata(self):
        raise NotImplementedError(
            "ConstantHandler.get_metadata removed in v2 -- "
            "v2 metadata lives in TokenResolver.metadata[Category.*]. "
            "See Phase 2.A.1 (CSV writer)."
        )

    def get_metadata_list_by_opaque_id(self):
        raise NotImplementedError(
            "ConstantHandler.get_metadata_list_by_opaque_id removed in v2 -- "
            "v2 metadata is per-category, indexed by per-category identity. "
            "See Phase 2.A.1 (CSV writer)."
        )

    def reorder_metadata_for_mapping(self, opaque_mapping):
        raise NotImplementedError(
            "ConstantHandler.reorder_metadata_for_mapping removed in v2 -- "
            "v2 has no frequency-sort reordering step. "
            "See Phase 2.B.7 (opaque_remapping migration)."
        )
