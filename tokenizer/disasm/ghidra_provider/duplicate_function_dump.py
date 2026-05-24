"""Duplicate-function metadata dump (collision-group orchestrator).

Concern: given a stream of ``(name, ghidra_func)`` pairs for the
functions in a single Ghidra-analysed binary, identify the groups
where two or more functions share the same ``name`` (the within-binary
positional disambiguator is unstable across ISA variants - see the
parent task's investigation notes) and write a pickle dump of each
colliding function's 5-layer-deep metadata snapshot for offline
human inspection.

Format choice: pickle (``pickle.HIGHEST_PROTOCOL``). Binary-only and
Python-specific, picked because the snapshot helper's L5 terminal
layer emits ``str(repr(value))`` leaves that round-trip cleanly and
because pickle accepts arbitrary nested dicts/lists/primitives
without per-type serialiser hooks.

This module owns only the orchestration concern - input collection,
collision detection by name, pickle serialisation, file write. The
per-function Java-handle-to-dict introspection lives in
``function_metadata_snapshot``; the snapshot helper is the only place
this module touches the Java API.

Output shape (pickled dict):

    {
      "binary": "<binary basename>",
      "duplicate_groups": [
        {
          "name": "<colliding function name>",
          "count": <N>,
          "functions": [
            {"entry": <int>, "snapshot": {<5-layer dict>}},
            ...
          ]
        },
        ...
      ]
    }

When no colliding names exist, the file is still written but
``duplicate_groups`` is an empty list (lets downstream tooling
distinguish "ran, found nothing" from "never ran").
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

from tokenizer.disasm.ghidra_provider.function_metadata_snapshot import (
    snapshot_function,
)


def write_duplicate_function_dump(
    funcs: Iterable[tuple[int, str, Any]],
    binary_name: str,
    output_path: Path,
) -> int:
    """Write the duplicate-function metadata dump for ``funcs`` to ``output_path``.

    ``funcs`` is the same ``(entry_addr, name, ghidra_func)`` triple
    list the provider's ``iter_functions`` builds before sorting -
    passing it through verbatim avoids a second walk over the
    function manager.

    Returns the number of duplicate groups written (i.e. the number
    of distinct names that appear two or more times). Callers can
    log this value as proof-of-life without re-reading the dump.
    """
    groups = _group_by_name(funcs)
    duplicate_groups: list[dict[str, Any]] = [
        _build_group_entry(name, entries) for name, entries in groups.items() if len(entries) > 1
    ]
    duplicate_groups.sort(key=lambda g: g["name"])

    payload: dict[str, Any] = {
        "binary": binary_name,
        "duplicate_groups": duplicate_groups,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return len(duplicate_groups)


def _group_by_name(
    funcs: Iterable[tuple[int, str, Any]],
) -> dict[str, list[tuple[int, Any]]]:
    """Group ``(entry, name, ghidra_func)`` triples by ``name``.

    Returns ``{name: [(entry, ghidra_func), ...]}``. Preserves the
    insertion order of ``funcs`` within each group so the offline
    inspector sees entries in iteration order (matches the
    positional disambiguator's numbering convention).
    """
    groups: dict[str, list[tuple[int, Any]]] = {}
    for entry, name, ghidra_func in funcs:
        groups.setdefault(name, []).append((entry, ghidra_func))
    return groups


def _build_group_entry(name: str, entries: list[tuple[int, Any]]) -> dict[str, Any]:
    """Build the payload dict for one collision group."""
    return {
        "name": name,
        "count": len(entries),
        "functions": [
            {"entry": entry, "snapshot": snapshot_function(ghidra_func)}
            for entry, ghidra_func in entries
        ],
    }
