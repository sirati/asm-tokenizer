"""
FunctionData class for representing a single function version.
"""

from typing import Dict

import numpy as np


class FunctionData:
    """Represents a single function version with its data."""

    def __init__(
        self,
        func_name: str,
        metadata: Dict,
        tokens: np.ndarray,
        insn_runlength: np.ndarray,
        block_runlength: np.ndarray,
        variant_tokens: np.ndarray,
    ):
        self.func_name = func_name
        self.metadata = metadata
        self.tokens = tokens
        self.insn_runlength = insn_runlength
        self.block_runlength = block_runlength
        # Variant-axis token IDs (uint16) resolved against the unified vocab.
        # Always populated by the dataloader; zero-length only on a corrupt
        # dataset. See plan §"Variant-prefixed function load".
        self.variant_tokens = variant_tokens

    def __len__(self):
        return len(self.tokens)

    def full_token_stream(self) -> np.ndarray:
        """Variant tokens concatenated with instruction tokens.

        Method (not property) because the concat is O(n+m) and copies memory;
        an attribute would mis-signal the cost as free.
        """
        return np.concatenate([self.variant_tokens, self.tokens])

    def __repr__(self):
        return f"FunctionData({self.func_name}, {self.metadata.get('arch')}-{self.metadata.get('compiler')}-{self.metadata.get('opt')}, {len(self)} tokens)"
