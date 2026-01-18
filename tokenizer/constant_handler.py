import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import BlockToken, MemoryOperandSymbol, OpaqueConstToken, TokenResolver, Tokens


class ConstantHandler:
    """Handles constant value processing and token creation"""

    def __init__(
        self,
        vocab_manager: VocabularyManager,
        resolver: TokenResolver,
        constant_dict: Dict[str, List[str]],
        block_ranges: np.ndarray,
    ):
        self.vocab_manager = vocab_manager
        self.resolver = resolver
        self.constant_dict = constant_dict
        self.block_ranges = block_ranges

        # Track usage counts for constants
        self.opaque_const_usage: Dict[int, int] = {}

        # Track created tokens by their original value
        self.opaque_const_tokens: Dict[int, Tokens] = {}

        # Track metadata for opaque constants
        self.opaque_metadata: Dict[int, Tuple] = {}

        # Track block tokens
        self.block_tokens: Dict[int, Tokens] = {}

    def process_constant(
        self,
        value: int,
        is_arithmetic: bool = False,
        meta: Optional[Dict] = None,
        library_type: str = "unknown",
        insn_mnemonic: Optional[str] = None,
    ) -> List[Tokens]:
        """
        Process a constant value and return the appropriate token.

        Args:
            value: value of the constant
            is_arithmetic: Whether this constant is used in arithmetic operations
            meta: Optional metadata for opaque constants
            library_type: Type of library for opaque constants

        Returns:
            TokensRepl object representing the constant
        """
        """# Remove '0x' prefix if present
        clean_hex = hex_value[2:] if hex_value.startswith('0x') else hex_value

        # Convert to integer for range checking
        try:
            value = int(clean_hex, 16)
        except ValueError:
            raise ValueError(f"Invalid hex value: {hex_value}")"""

        # Metadata ranges starting at 0 represent abstract constant domains, not real memory segments
        if meta is not None and meta.get("start_addr") == 0:
            is_arithmetic = True

        # Check if it's a small constant (0x00 to 0xFF)
        if is_arithmetic or 0x00 <= value <= 0xFF or value in self.constant_dict:
            return [self.vocab_manager.Valued_Const(value)]

        #
        match_mask = (self.block_ranges[:, 0] <= value) & (value < self.block_ranges[:, 1])
        if np.any(match_mask):
            idx = match_mask.nonzero()[0][0]
            if self.block_ranges[idx, 0] == value:
                return [self.vocab_manager.Block(idx)]
            else:
                return [
                    self.vocab_manager.Block(idx),
                    self.vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS),
                    self.vocab_manager.Valued_Const(value - self.block_ranges[idx, 0]),
                ]
        else:
            return self._create_opaque_const_with_offset(value, meta, library_type, insn_mnemonic)

    def _create_opaque_const_with_offset(
        self,
        value: int,
        meta: Optional[Dict] = None,
        library_type: str = "unknown",
        insn_mnemonic: Optional[str] = None,
    ) -> List[Tokens]:
        """Create an opaque constant token, decomposing into base+offset if pointing into a range"""
        if meta is not None and "start_addr" in meta and "end_addr" in meta and value > meta["start_addr"]:
            start_addr = meta["start_addr"]
            end_addr = meta["end_addr"]
            range_length = end_addr - start_addr
            offset = value - start_addr

            # Apply heuristics to decide if we should decompose
            should_decompose = True
            reason = ""

            # Heuristic 5: Don't decompose local_function or library_function ranges (they should be exact)
            if meta.get("type") in ["local_function", "library_function", "unknown_function"]:
                should_decompose = False
                reason = f"function range (type={meta.get('type')})"

            # Heuristic 3: Call instructions should not be decomposed (must point to function header)
            elif insn_mnemonic and insn_mnemonic.startswith("call"):
                should_decompose = False
                reason = "call instruction must point to function header"

            # Heuristic 1: Range cannot start at 0 (not in binary)
            elif start_addr == 0:
                should_decompose = False
                reason = "range starts at 0"

            # Heuristic 2: Range cannot be longer than 2^16
            elif range_length > (1 << 16):
                should_decompose = False
                reason = f"range too large (length={range_length:#x} > 0x10000)"

            # Heuristic 6: Prefer decomposing data/rodata/bss sections
            elif meta.get("type") not in ["data", "rodata", "bss", "code"]:
                should_decompose = False
                reason = f"unexpected metadata type: {meta.get('type')}"

            logger = logging.getLogger(__name__)

            # If value points into the range (not at the start), consider decomposition
            if start_addr < value < end_addr and should_decompose:
                insn_info = f" in {insn_mnemonic}" if insn_mnemonic else ""
                logger.debug(
                    f"Decomposing: range {start_addr:#x}-{end_addr:#x} (length={range_length:#x}, type={meta.get('type')}) "
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
                    f"Skipping decomposition: range {start_addr:#x}-{end_addr:#x} (length={range_length:#x}, type={meta.get('type')}) "
                    f"for target {value:#x}, offset={offset:#x}{insn_info}, reason: {reason}"
                )

        # Otherwise just create a simple opaque constant
        return [self._create_opaque_const(value, meta, library_type)]

    def _create_opaque_const(self, value: int, meta: Optional[Dict] = None, library_type: str = "unknown") -> Tokens:
        """Create an opaque constant token"""
        if value not in self.opaque_const_tokens:
            # Get or create opaque ID
            opaque_id = self.resolver.get_opaque_id(value)
            token = self.vocab_manager.Opaque_Const(opaque_id)
            self.opaque_const_tokens[value] = token
            self.opaque_const_usage[value] = 1

            # Store metadata if provided
            if meta is not None:
                self.opaque_metadata[value] = (
                    hex(meta["start_addr"]),
                    hex(meta["end_addr"]),
                    meta["name"],
                    meta["type"],
                    library_type,
                )
        else:
            self.opaque_const_usage[value] += 1

        return self.opaque_const_tokens[value]

    def get_sorted_opaque_constants(self) -> List[Tuple[int, Tokens, int]]:
        """Get opaque constants sorted by usage count (descending)"""
        return sorted(
            [(val, token, self.opaque_const_usage[val]) for val, token in self.opaque_const_tokens.items()],
            key=lambda x: x[2],
            reverse=True,
        )

    def create_opaque_mapping(self) -> Dict[BlockToken, Tokens]:
        """
        Create mapping from old opaque tokens to new tokens based on sorted usage.
        This is used for reassigning IDs after sorting by usage.
        """
        sorted_opaques = self.get_sorted_opaque_constants()
        mapping = {}

        old_token: OpaqueConstToken
        # Create new tokens with sequential IDs based on usage ranking
        for new_id, (value, old_token, usage_count) in enumerate(sorted_opaques):
            if new_id != old_token.id:
                new_token = self.vocab_manager.Opaque_Const(new_id)
                mapping[old_token] = new_token

        return mapping

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all constants"""
        return {"value_constants": self.value_constant_usage.copy(), "opaque_constants": self.opaque_const_usage.copy()}

    def get_metadata(self) -> Dict[str, Tuple]:
        """Get metadata for opaque constants"""
        return self.opaque_metadata.copy()

    def get_metadata_list_by_opaque_id(self) -> List[Tuple]:
        """
        Get metadata as a list indexed by opaque ID (after remapping).
        Returns a list where index corresponds to the opaque token ID.

        Returns:
            List of metadata tuples indexed by opaque ID
        """
        sorted_opaques = self.get_sorted_opaque_constants()

        # Create list indexed by new opaque ID
        metadata_list = []
        for new_id, (value, old_token, usage_count) in enumerate(sorted_opaques):
            if value in self.opaque_metadata:
                metadata_list.append(self.opaque_metadata[value])
            else:
                # No metadata for this opaque constant (shouldn't happen normally)
                metadata_list.append(None)

        return metadata_list

    def reorder_metadata_for_mapping(self, opaque_mapping: Dict[Tokens, Tokens]) -> None:
        """
        Reorder metadata to match the new token ordering after opaque mapping.

        Args:
            opaque_mapping: Dictionary mapping old opaque tokens to new sorted tokens
        """
        if not opaque_mapping:
            return

        # Create a reverse mapping from old tokens to hex values
        token_to_hex = {}
        for hex_val, token in self.opaque_const_tokens.items():
            token_to_hex[token] = hex_val

        # Create new metadata dict with reordered entries
        new_metadata = {}

        # Process each mapping
        for old_token, new_token in opaque_mapping.items():
            if old_token in token_to_hex:
                hex_val = token_to_hex[old_token]
                if hex_val in self.opaque_metadata:
                    # The metadata stays with the hex value, but we need to ensure
                    # it's accessible via the new token's ID
                    new_metadata[hex_val] = self.opaque_metadata[hex_val]

        # Update the metadata dict
        for hex_val, metadata in new_metadata.items():
            self.opaque_metadata[hex_val] = metadata
