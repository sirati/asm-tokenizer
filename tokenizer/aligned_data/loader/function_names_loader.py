"""Loader for the per-binary ``<binary>_function_names.txt`` sidecar.

Mirrors :class:`tokenizer.memmap_builder.function_names.FunctionNamesRegistry`
on the read side: validates the ``# format=N`` prelude, returns
bidirectional ``name <-> line_no`` lookup dicts. The section CSVs
store base64 line numbers; loaders use ``line_to_name`` to resolve
them back to function names, and ``name_to_line`` to translate
inbound queries.

Hard cutover: a missing or wrong-version prelude raises
:class:`ValueError` with a migration-pointing message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION

_EXPECTED_PRELUDE = f"# format={MEMMAP_FORMAT_VERSION}"


def load_function_names(
    path: Path,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load the function-names sidecar at ``path``.

    Returns ``(name_to_line, line_to_name)`` with 1-indexed line
    numbers. The first line of the file must be exactly
    ``# format=<MEMMAP_FORMAT_VERSION>``; any deviation raises
    :class:`ValueError`. A missing file also raises
    :class:`ValueError` (hard cutover -- callers are not allowed to
    silently fall back to a sidecar-less path).
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"{path}: function-names sidecar missing; re-run memmap_builder "
            f"to regenerate the sidecar at format_version={MEMMAP_FORMAT_VERSION}"
        )
    with open(path, "r", encoding="utf-8", newline="") as f:
        first_line = f.readline()
        if first_line.rstrip("\r\n") != _EXPECTED_PRELUDE:
            raise ValueError(
                f"{path}: missing or unsupported function-names sidecar prelude; "
                f"expected first line {_EXPECTED_PRELUDE!r}, got {first_line!r}; "
                f"re-run memmap_builder to regenerate the sidecar at "
                f"format_version={MEMMAP_FORMAT_VERSION}"
            )
        name_to_line: Dict[str, int] = {}
        line_to_name: Dict[int, str] = {}
        for line_no, raw in enumerate(f, start=1):
            name = raw.rstrip("\r\n")
            name_to_line[name] = line_no
            line_to_name[line_no] = name
    return name_to_line, line_to_name
