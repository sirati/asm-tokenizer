"""
MatchedFunction class for representing a function with multiple compilation variants.
"""

from typing import List

import numpy as np

from .function_data import FunctionData


class MatchedFunction:
    """Represents a matched function with multiple variants."""

    def __init__(self, func_name: str, variants: List[FunctionData]):
        self.func_name = func_name
        self.variants = variants

    def __len__(self):
        """Return average token count across all variants."""
        return int(np.mean([len(v) for v in self.variants]))

    def __repr__(self):
        return f"MatchedFunction({self.func_name}, {len(self.variants)} variants, avg {len(self)} tokens)"
