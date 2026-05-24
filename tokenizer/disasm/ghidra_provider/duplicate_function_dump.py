"""Duplicate-function metadata dump (collision-group orchestrator).

Concern: given a stream of ``(name, ghidra_func)`` pairs for the
functions in a single Ghidra-analysed binary, identify the groups
where two or more functions share the same ``name`` (the within-binary
positional disambiguator is unstable across ISA variants - see the
parent task's investigation notes) and write a JSON dump of each
colliding function's 3-layer-deep metadata snapshot for offline
human inspection.

This module owns only the orchestration concern - input collection,
collision detection by name, JSON serialisation, file write. The
per-function Java-handle-to-dict introspection lives in
``function_metadata_snapshot``; the snapshot helper is the only place
this module touches the Java API.

Output shape (JSON):

    {
      "binary": "<binary basename>",
      "duplicate_groups": [
        {
          "name": "<colliding function name>",
          "count": <N>,
          "functions": [
            {"entry": <int>, "snapshot": {<3-layer dict>}},
            ...
          ]
        },
        ...
      ]
    }

When no colliding names exist, the file is still written but
``duplicate_groups`` is an empty list (lets downstream tooling
distinguish "ran, found nothing" from "never ran").

JSON serialisability: the snapshot helper returns dicts / lists /
primitives plus a ``{"_java_class": ..., "_repr": ...}`` summary for
out-of-budget sub-objects. Every leaf is JSON-safe at construction
time. A final ``json.dumps(..., default=repr)`` traps any unforeseen
non-serialisable straggler so the dump never aborts the disassembly.
"""

from __future__ import annotations

import json
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
    log this value as proof-of-life without re-reading the JSON.
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
    # ``default=repr`` is a defensive backstop - the snapshot driver
    # already coerces every leaf to JSON-safe types; if something
    # exotic slips through we want the dump to land with a repr
    # placeholder rather than aborting the disassembly.
    output_path.write_text(json.dumps(payload, indent=2, default=repr))
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
    """Build the JSON dict for one collision group."""
    return {
        "name": name,
        "count": len(entries),
        "functions": [
            {"entry": entry, "snapshot": snapshot_function(ghidra_func)}
            for entry, ghidra_func in entries
        ],
    }
