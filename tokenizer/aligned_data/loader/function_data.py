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
    ):
        self.func_name = func_name
        self.metadata = metadata
        self.tokens = tokens
        self.insn_runlength = insn_runlength
        self.block_runlength = block_runlength

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return f"FunctionData({self.func_name}, {self.metadata.get('arch')}-{self.metadata.get('compiler')}-{self.metadata.get('opt')}, {len(self)} tokens)"
