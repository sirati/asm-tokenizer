"""Regression: realized-geometry sidecars must not become phantom binaries.

The Phase-4 realized-GEOMETRY pass emits ``<binary>_realized.bin`` +
``<binary>_realized_index.bin`` (and the ``_unmatched_realized_*``
companions) alongside each binary's catalog. Those index sidecars share
the ``_index.bin`` tail with the matched-arm ``<binary>_index.bin``
binary-existence signal, so the collection's discovery MUST exclude them
the same way :func:`discover_binaries` does -- otherwise each geometry
sidecar is mistaken for a separate ``<binary>_realized`` "binary" that
carries no ``.idx`` and aborts discovery under the default
:attr:`MissingIndexPolicy.RAISE`.

This test lays a single decodable binary, runs BOTH realized passes (so
the ``_realized_index.bin`` files physically exist via the production
generator), and asserts discovery still finds exactly the one real
binary. Reverting the exclusion (covering only the ``_lengths_*`` arm)
re-introduces the ``<binary>_realized`` phantom and re-raises.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.aligned_data.realized_lengths import generate_realized_geometry
from tokenizer.aligned_data.sorted_index import (
    IndexedMemmapCollection,
    LengthReduction,
    ReductionKind,
)

from ._length_helpers import ensure_sidecar
from .fixtures import build_combined_fixture, make_test_vocab_manager


_MAX = LengthReduction(ReductionKind.MAX)
_DEPTH = 3


def test_geometry_sidecar_not_a_phantom_binary(tmp_path: Path) -> None:
    memmap_dir = tmp_path / "pkg"
    # ``build_combined_fixture`` lays the catalog (matched + unmatched
    # ``_index.bin``, ``_data.bin``, ``_sections`` ...) under a fixed stem.
    base = build_combined_fixture(memmap_dir)
    binary_name = _sole_binary_name(base)

    # Phase-4a realized-LENGTH + realized-GEOMETRY sidecars (the latter is
    # the one that drops ``<binary>_realized_index.bin`` next to the
    # catalog -- the file the buggy exclusion missed).
    ensure_sidecar(base, binary_name)
    generate_realized_geometry(base, binary_name)
    assert (base / f"{binary_name}_realized_index.bin").is_file()
    assert (base / f"{binary_name}_unmatched_realized_index.bin").is_file()

    from tokenizer.aligned_data.sorted_index import write_sorted_index_files

    write_sorted_index_files(
        base, binary_name, reductions=[_MAX], depths=[_DEPTH],
    )

    # Default policy is RAISE: a phantom ``<binary>_realized`` member
    # (which carries no ``.idx``) would abort here. The exclusion keeps
    # discovery to the one real binary.
    with IndexedMemmapCollection.discover(
        [base],
        reduction=_MAX,
        depth=_DEPTH,
        vocab_manager=make_test_vocab_manager(),
    ) as coll:
        assert coll.binary_names == [binary_name]


def _sole_binary_name(memmap_dir: Path) -> str:
    """The single matched-arm binary stem the fixture wrote."""
    names = [
        entry.name[: -len("_index.bin")]
        for entry in memmap_dir.iterdir()
        if entry.is_file()
        and entry.name.endswith("_index.bin")
        and not entry.name.endswith("_unmatched_index.bin")
    ]
    assert len(names) == 1, names
    return names[0]
