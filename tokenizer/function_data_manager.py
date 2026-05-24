"""
Optimized function data storage using fixed-size arrays for better performance.
"""

import numpy as np
from typing import Any, Hashable, Iterator, Tuple, Optional, Dict, List
from dataclasses import dataclass

from .function_deduper import FunctionDeduper
from .function_token_list import FunctionTokenList


@dataclass
class FunctionData:
    """Consolidated data structure for function analysis results"""

    tokens: FunctionTokenList  # list[Tokens]
    tokens_base64: str
    block_runlength_base64: str
    instruction_runlength_base64: str
    metadata_cell: str


class FunctionDataManager:
    """
    Optimized storage for function data using fixed-size arrays.
    Provides O(1) access and better memory locality compared to dictionaries.
    """

    def __init__(self, total_functions: int):
        """
        Initialize the manager with pre-allocated arrays.

        Args:
            total_functions: Total number of functions to allocate space for
        """
        self.total_functions = total_functions
        self.current_index = 0

        # Hash map to track function name occurrences
        self.func_name_occurrences: Dict[str, int] = {}

        # Access map: (func_name, occurrence_index) -> array_index
        self.access_map: Dict[Tuple[str, int], int] = {}

        # Reverse lookup: (func_name, identity_key) -> final_func_name
        # of the FIRST accepted record with that identity. A later call
        # with the SAME (name, identity_key) AND matching content folds
        # into that record and re-returns its final name.
        self._identity_to_final: Dict[Tuple[str, Hashable], str] = {}

        # Semantic-merge gate (see :mod:`tokenizer.function_deduper`).
        # When the gate reports a duplicate, ``add_function_data`` skips
        # the allocation, leaves arrays + occurrence counters untouched,
        # and returns the existing record's final name.
        self._deduper = FunctionDeduper()

        # Pre-allocated arrays using numpy object arrays for complex types
        self.func_name_addr_array = np.empty(total_functions, dtype=object)
        self.func_disas_array = np.empty(total_functions, dtype=object)
        self.func_disas_token_array = np.empty(total_functions, dtype=object)
        self.function_data_array = np.empty(total_functions, dtype=object)

    def add_function_data(
        self,
        func_name: str,
        func_addr: int,
        func_disas: Any,
        func_disas_token: Any,
        function_data: FunctionData,
        identity_key: Optional[Hashable] = None,
    ) -> str:
        """
        Add all function data in one operation.

        Args:
            func_name: Original function name
            func_addr: Function address
            func_disas: Function disassembly data
            func_disas_token: Function token data
            function_data: FunctionData instance
            identity_key: Optional provider-supplied "stronger-than-name"
                identity. When two calls share the same ``func_name``,
                the same ``identity_key``, AND the same emitted token
                body (``function_data.tokens_base64``), the second call
                is a semantic duplicate of the first: no new record is
                allocated, the occurrence counter is not bumped, and
                the first record's final name is returned. When
                ``identity_key`` is ``None`` (or the three-way check
                fails) the legacy ``_N``-suffix path runs unchanged.
                See ``FunctionView.identity_key`` in
                ``tokenizer/disasm/types.py`` for the provider contract.

        Returns:
            Final function name. New records use the legacy
            occurrence-suffix scheme (``name``, ``name_1``, ``name_2``,
            ...); folded duplicates re-return the existing record's
            final name.
        """
        # Semantic-merge fast path: if this is a duplicate of a record
        # already in the manager (same name + identity_key + content),
        # fold into the existing record and return its final name. No
        # new slot is consumed; the occurrence counter is untouched.
        if self._deduper.is_duplicate(
            func_name, identity_key, function_data.tokens_base64
        ):
            return self._identity_to_final[(func_name, identity_key)]

        if self.current_index >= self.total_functions:
            raise IndexError(f"Cannot add more functions: array is full ({self.total_functions})")

        # Handle duplicate function names
        if func_name in self.func_name_occurrences:
            occurrence_index = self.func_name_occurrences[func_name]
            final_func_name = f"{func_name}_{occurrence_index}"
            self.func_name_occurrences[func_name] = occurrence_index + 1
        else:
            final_func_name = func_name
            self.func_name_occurrences[func_name] = 1
            occurrence_index = 0

        # Store the access mapping
        self.access_map[(func_name, occurrence_index)] = self.current_index

        # Record the identity-key reverse lookup for future fold checks.
        # ``identity_key=None`` is never recorded (the deduper's
        # ``is_duplicate`` short-circuits to False on None, so the
        # reverse-lookup entry would never be consulted).
        if identity_key is not None:
            self._identity_to_final.setdefault(
                (func_name, identity_key), final_func_name
            )

        # Store data in arrays
        self.func_name_addr_array[self.current_index] = func_addr
        self.func_disas_array[self.current_index] = func_disas
        self.func_disas_token_array[self.current_index] = func_disas_token
        self.function_data_array[self.current_index] = function_data

        self.current_index += 1
        return final_func_name

    def get_function_addr(self, func_name: str, occurrence: int = 0) -> Optional[int]:
        """Get function address by name and occurrence index."""
        key = (func_name, occurrence)
        if key in self.access_map:
            index = self.access_map[key]
            return self.func_name_addr_array[index]
        return None

    def get_function_disas(self, func_name: str, occurrence: int = 0) -> Optional[Any]:
        """Get function disassembly by name and occurrence index."""
        key = (func_name, occurrence)
        if key in self.access_map:
            index = self.access_map[key]
            return self.func_disas_array[index]
        return None

    def get_function_disas_token(self, func_name: str, occurrence: int = 0) -> Optional[Any]:
        """Get function token disassembly by name and occurrence index."""
        key = (func_name, occurrence)
        if key in self.access_map:
            index = self.access_map[key]
            return self.func_disas_token_array[index]
        return None

    def get_function_data(self, func_name: str, occurrence: int = 0) -> Optional[FunctionData]:
        """Get function data by name and occurrence index."""
        key = (func_name, occurrence)
        if key in self.access_map:
            index = self.access_map[key]
            return self.function_data_array[index]
        return None

    def iter_func_name_addr(self) -> Iterator[Tuple[str, int, int]]:
        """Iterate over all function name-address pairs."""
        # Create a sorted list of keys to ensure consistent ordering
        sorted_keys = sorted(self.access_map.keys())
        for func_name, occurrence in sorted_keys:
            index = self.access_map[(func_name, occurrence)]
            yield func_name, occurrence, self.func_name_addr_array[index]

    def iter_func_disas(self) -> Iterator[Tuple[str, int, Any]]:
        """Iterate over all function disassembly data."""
        # Create a sorted list of keys to ensure consistent ordering
        sorted_keys = sorted(self.access_map.keys())
        for func_name, occurrence in sorted_keys:
            index = self.access_map[(func_name, occurrence)]
            yield func_name, occurrence, self.func_disas_array[index]

    def iter_func_disas_token(self) -> Iterator[Tuple[str, int, Any]]:
        """Iterate over all function token disassembly data."""
        # Create a sorted list of keys to ensure consistent ordering
        sorted_keys = sorted(self.access_map.keys())
        for func_name, occurrence in sorted_keys:
            index = self.access_map[(func_name, occurrence)]
            yield func_name, occurrence, self.func_disas_token_array[index]

    def iter_function_data(self) -> Iterator[Tuple[str, int, FunctionData]]:
        """Iterate over all function data."""
        # Create a sorted list of keys to ensure consistent ordering
        sorted_keys = sorted(self.access_map.keys())
        for func_name, occurrence in sorted_keys:
            index = self.access_map[(func_name, occurrence)]
            yield func_name, occurrence, self.function_data_array[index]

    def iter_all_data(self) -> Iterator[Tuple[str, int, int, Any, Any, FunctionData]]:
        """Iterate over all data at once for maximum efficiency."""
        # Create a sorted list of keys to ensure consistent ordering
        sorted_keys = sorted(self.access_map.keys())
        for func_name, occurrence in sorted_keys:
            index = self.access_map[(func_name, occurrence)]
            yield (
                func_name,
                occurrence,
                self.func_name_addr_array[index],
                self.func_disas_array[index],
                self.func_disas_token_array[index],
                self.function_data_array[index],
            )

    def get_used_count(self) -> int:
        """Get the number of actually used slots."""
        return self.current_index

    def get_function_names(self) -> List[str]:
        """Get all function names in order they were added."""
        names = []
        for i in range(self.current_index):
            # Find the function name for this index
            for (func_name, occurrence), index in self.access_map.items():
                if index == i:
                    if occurrence == 0:
                        names.append(func_name)
                    else:
                        names.append(f"{func_name}_{occurrence}")
                    break
        return names

    def compact_arrays(self) -> None:
        """
        Compact arrays by removing unused slots.
        Since we don't support removal, this is mainly useful if the initial
        total_functions estimate was too large.
        """
        if self.current_index == self.total_functions:
            return  # Already compact

        # Create new arrays with only used slots
        new_func_name_addr = np.empty(self.current_index, dtype=object)
        new_func_disas = np.empty(self.current_index, dtype=object)
        new_func_disas_token = np.empty(self.current_index, dtype=object)
        new_function_data = np.empty(self.current_index, dtype=object)
        new_is_jump_only = np.empty(self.current_index, dtype=bool)

        # Copy data to new arrays (data is already contiguous from 0 to current_index)
        new_func_name_addr[:] = self.func_name_addr_array[: self.current_index]
        new_func_disas[:] = self.func_disas_array[: self.current_index]
        new_func_disas_token[:] = self.func_disas_token_array[: self.current_index]
        new_function_data[:] = self.function_data_array[: self.current_index]
        new_is_jump_only[:] = self.is_jump_only[: self.current_index]

        # Replace old arrays
        self.func_name_addr_array = new_func_name_addr
        self.func_disas_array = new_func_disas
        self.func_disas_token_array = new_func_disas_token
        self.function_data_array = new_function_data
        self.is_jump_only = new_is_jump_only
        self.total_functions = self.current_index
