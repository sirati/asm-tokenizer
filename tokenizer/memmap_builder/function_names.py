"""Build-time registry for the per-binary function-names sidecar.

Pass 1 of the memmap builder feeds every matched + unmatched function
name through :meth:`FunctionNamesRegistry.add`. Between pass 1 and
pass 2 the registry is :meth:`finalize`-d (alphabetical, deduplicated)
and written to ``<binary>_function_names.txt`` via
:meth:`write_sidecar`. Pass 2 then resolves each function-name cell
in the section CSVs to its 1-indexed line number for the base64
indirection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

_PRELUDE_LINE = f"# format={MEMMAP_FORMAT_VERSION}\n"


class FunctionNamesRegistry:
    """Collect function names; emit a sorted/deduplicated sidecar.

    The registry has two phases:

    * **collecting** -- ``.add(name)`` records names; ``.finalize()`` is
      not yet called. Calls to ``.line_no`` or ``.write_sidecar`` raise.
    * **finalized** -- ``.finalize()`` froze the registry into a sorted
      tuple + name->line dict. Further ``.add`` calls raise; lookups
      and ``.write_sidecar`` succeed.
    """

    def __init__(self) -> None:
        self._names: set[str] = set()
        self._sorted: Optional[Tuple[str, ...]] = None
        self._name_to_line: Optional[Dict[str, int]] = None

    @property
    def finalized(self) -> bool:
        return self._sorted is not None

    def add(self, name: str) -> None:
        if self.finalized:
            raise RuntimeError(
                "FunctionNamesRegistry is finalized; cannot add more names"
            )
        self._names.add(name)

    def finalize(self) -> None:
        """Freeze the registry: sort, dedupe, build the line-no index.

        Line 1 is the first name AFTER the ``# format=N`` prelude line.
        Idempotent: a second call is a no-op rather than an error so
        callers can finalize defensively without tracking state.
        """
        if self.finalized:
            return
        self._sorted = tuple(sorted(self._names))
        # 1-indexed: line 1 is the first name after the prelude.
        self._name_to_line = {name: i + 1 for i, name in enumerate(self._sorted)}

    def line_no(self, name: str) -> int:
        if not self.finalized:
            raise RuntimeError(
                "FunctionNamesRegistry is not finalized; call .finalize() first"
            )
        assert self._name_to_line is not None  # for type checkers
        return self._name_to_line[name]

    def write_sidecar(self, output_dir: Path, binary_name: str) -> Path:
        """Write ``<output_dir>/<binary_name>_function_names.txt``.

        Line 1 is the ``# format=N`` prelude; lines 2..N+1 are the
        sorted unique function names. Returns the written path.
        """
        if not self.finalized:
            raise RuntimeError(
                "FunctionNamesRegistry is not finalized; call .finalize() first"
            )
        assert self._sorted is not None  # for type checkers
        path = Path(output_dir) / f"{binary_name}_function_names.txt"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(_PRELUDE_LINE)
            for name in self._sorted:
                f.write(name)
                f.write("\n")
        return path
