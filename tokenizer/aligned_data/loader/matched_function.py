"""
MatchedFunction class for representing a function with multiple compilation versions.
"""

from typing import List

import numpy as np

from .function_data import FunctionData


class MatchedFunction:
    """Represents a matched function with multiple versions."""

    def __init__(self, func_name: str, versions: List[FunctionData]):
        self.func_name = func_name
        self.versions = versions

    def __len__(self):
        """Return average token count across all versions."""
        return int(np.mean([len(v) for v in self.versions]))

    def __repr__(self):
        return f"MatchedFunction({self.func_name}, {len(self.versions)} versions, avg {len(self)} tokens)"
