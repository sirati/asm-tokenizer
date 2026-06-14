"""Byte-identical gate for the pass-1 entry-spill restructure.

The builder spills pass-1 entry metadata to a temp file
(:class:`tokenizer.memmap_builder._entry_spool.EntrySpool`) and streams
it back in pass 2 instead of retaining the whole corpus as live Python
lists. That changes the memory profile only — the produced output must
be byte-for-byte identical to a build that kept the entries in RAM.

This test pins that invariant end-to-end: a dense synthetic corpus is
driven through ``build_memmap_files`` twice over the SAME inputs —

* once with the real on-disk ``EntrySpool``;
* once with ``EntrySpool`` monkeypatched to an in-memory list-backed
  shim with the identical append/iterate/close surface (i.e. the
  pre-spill "retain the whole list" semantics).

Every emitted artefact (``*_sections.bin``, ``*_data.bin``,
``*_index.bin``, ``*_unmatched_*``, ``*_sections.csv``,
``*_function_names.txt``, ``*_extern_providers.txt``,
``*_variants.{bin,csv}``) must have an identical sha256 across the two
builds. Any divergence means the spill round-trip is not faithful.

The corpus exercises the structural shapes that drive the entry stream
through every pass-2 branch: a function matched across all variants, a
matched function with one variant dropped, a single-stream unmatched
function, a DUPLICATED name (forced down the unmatched arm), local /
PLT / EXTERN call edges (with extern libraries) and cross-arm call
resolution (a matched function calling an unmatched callee and vice
versa).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import tokenizer.memmap_builder.builder as builder_mod
from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.memmap_builder.builder import (
    BinaryVersionInfo,
    build_memmap_files,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.saver import save_vocabulary
from tokenizer.vocab_unifier.unifier import unify_vocab


# ---------------------------------------------------------------------------
# In-memory shim: identical API to EntrySpool, but retains every entry as a
# live Python list — the pre-restructure "hold the whole corpus" behaviour.
# ---------------------------------------------------------------------------
class _InMemoryEntryList:
    """List-backed stand-in for :class:`EntrySpool`.

    Same ``append`` / ``__iter__`` / ``close`` surface; never touches
    disk. Re-iterable like the real spool. Used to reproduce the exact
    pre-spill semantics so the two builds can be byte-compared.
    """

    def __init__(self, dir=None) -> None:  # noqa: A002 - mirror EntrySpool
        self._entries: List[dict] = []

    def append(self, entry: dict) -> None:
        self._entries.append(entry)

    def __iter__(self):
        return iter(self._entries)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Dense corpus synthesis: real per-binary v2 CSVs with function rows that
# decode through the production parsed-record iterator.
# ---------------------------------------------------------------------------
_HEADER = [
    "function_name",
    "occurrence",
    "tokens_base64",
    "block_runlength_base64",
    "instruction_runlength_base64",
    "metadata",
]


def _u16(values: List[int]) -> np.ndarray:
    return np.asarray(values, dtype=np.uint16)


def _meta(local=(), plt=(), ext=()) -> str:
    """Build a v2 ``metadata`` JSON cell from typed callee names.

    ``ext`` entries are ``(name, library)`` pairs so the EXTERN branch
    populates ``extern_libraries``; ``local`` / ``plt`` are bare names.
    """
    return json.dumps(
        {
            "local_funcs": [{"name": n} for n in local],
            "plt_funcs": [{"name": n} for n in plt],
            "ext_funcs": [{"name": n, "library": lib} for n, lib in ext],
        }
    )


def _row(func_name, occurrence, token_ids, block_rl, insn_rl, metadata) -> List[str]:
    return [
        func_name,
        str(occurrence),
        ndarray_to_base64(_u16(token_ids)),
        ndarray_to_base64(_u16(block_rl)),
        ndarray_to_base64(_u16(insn_rl)),
        metadata,
    ]


def _write_csv(csv_path: Path, platform: str, rows: List[List[str]]) -> None:
    """Write a per-binary v2 CSV in the production wire shape:
    ``version=2`` prelude row, the column header, the function rows, then
    the vocab def line as the final row (``is_vocab_row``-filtered by the
    iterator; the prelude is consumed by ``open_csv_skip_vocab``).

    Token IDs in the rows must be registered on the saved vocab so the
    unified-vocab build covers them; we register a generous Block_V2
    range so every synthetic token id resolves.
    """
    vm = VocabularyManager(platform=platform, format_version=2)
    for bid in range(0, 64):
        vm.Block_V2(bid)

    with open(csv_path, "w", newline="", encoding="ascii") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["version=2"])
        writer.writerow(_HEADER)
        for row in rows:
            writer.writerow(row)
        save_vocabulary(vm, writer)


def _build_corpus(tmp_path: Path) -> tuple[List[BinaryVersionInfo], Path]:
    """Lay down 3 per-binary CSVs + a unified vocab; return version
    infos + the unified-vocab path.

    Function design (canonical names shared across streams so the
    lockstep merge produces matched / unmatched / duplicated arms):

    * ``alpha`` — present in all 3 streams with DISTINCT bodies →
      matched (3 variants). Calls ``beta`` (LOCAL, cross-arm into an
      unmatched callee) + an EXTERN ``malloc``.
    * ``beta`` — present only in the x64 stream → unmatched. Calls
      ``alpha`` (LOCAL, cross-arm into a matched callee).
    * ``gamma`` — present in x64 + x86 with distinct bodies → matched
      (2 variants); arm64 omits it. Calls a PLT ``printf``.
    * ``delta`` — present in x64 twice (occurrence 0 and 1, distinct
      bodies) → DUPLICATED, forced down the unmatched arm. Calls
      ``alpha``.
    * ``epsilon`` — present in all 3 streams but every body identical →
      matched pass-1 drops it (all variants dedup to one offset), and
      the caller re-routes it to the unmatched arm.
    """
    # Distinct token bodies per (func, variant) so dedup doesn't collapse
    # matched variants to a single offset (which would drop the function).
    def body(seed):
        return ([seed % 60, (seed + 1) % 60, (seed + 2) % 60], [1, 1], [2])

    x64_rows = [
        _row("alpha", 0, *body(1), _meta(local=["beta"], ext=[("malloc", "libc.so")])),
        _row("beta", 0, *body(10), _meta(local=["alpha"])),
        _row("delta", 0, *body(20), _meta(local=["alpha"])),
        _row("delta", 1, *body(21), _meta(local=["alpha"])),
        _row("epsilon", 0, *body(30), _meta()),
        _row("gamma", 0, *body(40), _meta(plt=["printf"])),
    ]
    x86_rows = [
        _row("alpha", 0, *body(2), _meta(local=["beta"], ext=[("malloc", "libc.so")])),
        _row("epsilon", 0, *body(30), _meta()),
        _row("gamma", 0, *body(41), _meta(plt=["printf"])),
    ]
    arm64_rows = [
        _row("alpha", 0, *body(3), _meta(local=["beta"], ext=[("malloc", "libc.so")])),
        _row("epsilon", 0, *body(30), _meta()),
    ]

    specs = [
        ("x64-gcc-13.2.0-O2_pkg", "x64", x64_rows),
        ("x86-gcc-13.2.0-O2_pkg", "x86", x86_rows),
        ("arm64-clang-15.0.0-O3_pkg", "arm64", arm64_rows),
    ]

    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_files: List[Path] = []
    versions: List[BinaryVersionInfo] = []
    for basename, arch, rows in specs:
        path = tmp_path / f"{basename}_output.csv"
        _write_csv(path, arch, rows)
        csv_files.append(path)

    unified_vocab_path = tmp_path / "unified_vocab.csv"
    unify_vocab(csv_files, unified_vocab_path)

    arches = {"x64": ("x64", "gcc", "13.2.0", "O2"),
              "x86": ("x86", "gcc", "13.2.0", "O2"),
              "arm64": ("arm64", "clang", "15.0.0", "O3")}
    for (basename, arch, _rows), path in zip(specs, csv_files):
        a, c, v, o = arches[arch]
        versions.append(
            BinaryVersionInfo(
                path=path,
                mapping_path=path.with_suffix(".mapping.b64c"),
                arch=a,
                compiler=c,
                compilerversion=v,
                opt=o,
                pkg="pkg",
                filename=basename,
            )
        )
    return versions, unified_vocab_path


def _sha_tree(out_dir: Path) -> Dict[str, str]:
    """sha256 every emitted artefact under ``out_dir``, keyed by name."""
    shas: Dict[str, str] = {}
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            shas[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return shas


def _run_build(tmp_path: Path, out_name: str) -> Dict[str, str]:
    versions, unified_vocab_path = _build_corpus(tmp_path / out_name / "src")
    out_dir = tmp_path / out_name / "out"
    out_dir.mkdir(parents=True)
    build_memmap_files(versions, out_dir, "demo", unified_vocab_path)
    return _sha_tree(out_dir)


def test_spill_build_is_byte_identical_to_in_memory(tmp_path, monkeypatch) -> None:
    """The on-disk-spool build must be byte-identical to the in-memory
    (whole-list) build across every emitted artefact."""
    # Build 1: real on-disk EntrySpool (the restructured path).
    spool_shas = _run_build(tmp_path, "spool")

    # Build 2: EntrySpool replaced by the list-backed shim (pre-spill
    # semantics). The corpus inputs are regenerated identically.
    monkeypatch.setattr(builder_mod, "EntrySpool", _InMemoryEntryList)
    inmem_shas = _run_build(tmp_path, "inmem")

    # Same set of files, and every file byte-identical.
    assert set(spool_shas) == set(inmem_shas), (
        f"file set diverged: spool-only={set(spool_shas) - set(inmem_shas)}, "
        f"inmem-only={set(inmem_shas) - set(spool_shas)}"
    )
    mismatches = {
        name: (spool_shas[name], inmem_shas[name])
        for name in spool_shas
        if spool_shas[name] != inmem_shas[name]
    }
    assert not mismatches, (
        "spill build diverged from in-memory build (memory profile may "
        f"change, bytes must not): {mismatches}"
    )

    # Guard against a vacuous pass: the corpus must actually emit the
    # matched + unmatched section catalogs with real content (an
    # empty-corpus fixture would make the byte-compare trivially true).
    out_dir = tmp_path / "spool" / "out"
    assert (out_dir / "demo_sections.bin").stat().st_size > len(b"MSEC")
    assert (out_dir / "demo_data.bin").stat().st_size > 16  # > prelude
    assert (out_dir / "demo_unmatched_data.bin").stat().st_size > 16
    # Matched + unmatched section CSVs both carry rows; extern providers
    # captured the EXTERN ``malloc`` library — proving the EXTERN branch ran.
    assert (out_dir / "demo_sections.csv").stat().st_size > 0
    assert (out_dir / "demo_unmatched_sections.csv").stat().st_size > 0
    extern_lines = (out_dir / "demo_extern_providers.txt").read_text().splitlines()
    assert any("libc.so" in line for line in extern_lines), extern_lines
