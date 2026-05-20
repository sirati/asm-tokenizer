"""Tests for ``tokenizer.memmap_builder.variants.VariantRegistry``.

Covers the new byte-offset semantics:
  * ``ref(vkey)`` returns the byte offset assigned by ``write_sidecar``
    (formerly: row index into the verbose CSV).
  * ``write_sidecar`` emits ``<bin>_variants.bin`` (uint16 LE records
    via ``variant_tokens.record.write_record``) and slim
    ``<bin>_variants.csv`` with ``filename,offset`` columns only.
  * Round-trip: read the bin back at each registry-returned offset
    and decode through the unified vocab; result matches the input.
  * Dedup: two ``add``s of the same vkey produce a single bin record.

These tests reuse the ``FakeVocab`` test double from the
``variant_tokens`` suite — ``write_record`` only calls
``vocab.get_token_id``, so a full ``VocabularyManager`` (with its
token-class dispatch and digit reservation) is unnecessary here.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tokenizer.memmap_builder.builder import BinaryVersionInfo, VersionKey
from tokenizer.memmap_builder.variants import VariantRegistry
from tokenizer.variant_tokens.encoder import decode_record
from tokenizer.variant_tokens.prefixes import build_axis_strings
from tokenizer.variant_tokens.record import read_record
from tokenizer.variant_tokens.tests._fakes import FakeVocab


def _make_version(
    *,
    arch: str = "x86_64",
    compiler: str = "gcc",
    compilerversion: str = "13.2.0",
    opt: str = "-O2",
    pkg: str = "",
    variant_id: int = 0,
    filename: str = "",
    extra_metadata: Dict[str, Any] | None = None,
) -> BinaryVersionInfo:
    """Build a ``BinaryVersionInfo`` with sensible defaults.

    Path fields are unused by the registry (it only reads the
    canonical-4 + ``variant_id`` + ``filename`` + ``extra_metadata``),
    so we point them at ``/dev/null`` to keep test setup terse.
    """
    return BinaryVersionInfo(
        path=Path("/dev/null"),
        mapping_path=Path("/dev/null"),
        arch=arch,
        compiler=compiler,
        compilerversion=compilerversion,
        opt=opt,
        pkg=pkg,
        variant_id=variant_id,
        extra_metadata=extra_metadata or {},
        filename=filename,
    )


def _read_slim_csv_rows(csv_path: Path) -> List[List[str]]:
    """Open a slim ``_variants.csv``, skip the ``# format=N`` prelude
    line, and return the remaining rows parsed via ``csv.reader``.

    The prelude is consumed line-wise (not via ``csv.reader``) because
    ``csv.reader`` would surface it as a single-cell row and the
    surrounding tests assert on ``rows[0]`` being the standard header.
    """
    with open(csv_path, encoding="ascii") as fh:
        fh.readline()  # skip "# format=N" prelude
        return list(csv.reader(fh))


def _make_vocab_for(versions: List[BinaryVersionInfo]) -> FakeVocab:
    """Register every axis string for ``versions`` into a fresh fake vocab.

    Mirrors the unifier's pass-1 behaviour in miniature: every variant
    token the encoder will need has a stable ID before the registry
    runs.
    """
    vocab = FakeVocab(base_id=256)
    for version in versions:
        for token in build_axis_strings(version):
            vocab.register(token)
    return vocab


# ---------------------------------------------------------------------------
# Construction + dedup
# ---------------------------------------------------------------------------


def test_dedup_collapses_bin_records_but_keeps_csv_provenance(tmp_path):
    """Two ``BinaryVersionInfo`` entries with the same vkey collapse to
    one bin record but keep BOTH rows in the slim CSV.

    The bin file is content-addressed by vkey (one record per unique
    variant), while the CSV records every source artefact the caller
    handed in so downstream tooling does not lose provenance. The two
    CSV rows therefore share the same ``offset`` cell.
    """
    v1 = _make_version(filename="hello-x64-gcc-13.2.0-O2_corpora")
    v2 = _make_version(filename="hello-x64-gcc-13.2.0-O2_mirror")  # same vkey
    vocab = _make_vocab_for([v1])

    registry = VariantRegistry.from_versions([v1, v2], vocab)
    registry.write_sidecar(tmp_path, "hello")

    csv_path = tmp_path / "hello_variants.csv"
    rows = _read_slim_csv_rows(csv_path)
    # header + 2 data rows (one per BVI; bin still has one record).
    assert len(rows) == 3
    assert rows[0] == ["filename", "variant_id", "offset"]
    # Both data rows share the same offset cell.
    assert rows[1][2] == rows[2][2]
    # Filenames are distinct; ordering matches encounter order.
    assert [r[0] for r in rows[1:]] == [
        "hello-x64-gcc-13.2.0-O2_corpora",
        "hello-x64-gcc-13.2.0-O2_mirror",
    ]


def test_distinct_variant_ids_stay_distinct(tmp_path):
    """Two versions sharing canonical-4 but differing in ``variant_id``
    are distinct vkeys → two bin records, two CSV rows."""
    v1 = _make_version(
        variant_id=0,
        filename="hello-O2-noflag",
        extra_metadata={"hardening": "none"},
    )
    v2 = _make_version(
        variant_id=1,
        filename="hello-O2-fortify",
        extra_metadata={"hardening": "full"},
    )
    vocab = _make_vocab_for([v1, v2])

    registry = VariantRegistry.from_versions([v1, v2], vocab)
    registry.write_sidecar(tmp_path, "hello")

    csv_path = tmp_path / "hello_variants.csv"
    rows = _read_slim_csv_rows(csv_path)
    assert len(rows) == 3  # header + 2 data rows
    filenames = [r[0] for r in rows[1:]]
    assert filenames == ["hello-O2-noflag", "hello-O2-fortify"]


# ---------------------------------------------------------------------------
# ref(vkey) returns byte offset (the headline semantic flip)
# ---------------------------------------------------------------------------


def test_ref_returns_byte_offset_as_lowercase_hex(tmp_path):
    """``ref`` returns the offset assigned by the bin writer.

    Bare lowercase hex, no ``0x`` prefix — matches the
    ``f"{data_offset:x}"`` convention every other byte-offset cell in
    the section CSV uses.
    """
    v1 = _make_version(filename="a", extra_metadata={})
    v2 = _make_version(opt="-Os", filename="b", extra_metadata={"sani": "addr"})
    vocab = _make_vocab_for([v1, v2])

    registry = VariantRegistry.from_versions([v1, v2], vocab)
    registry.write_sidecar(tmp_path, "bin")

    vkey_1 = VersionKey(arch="x86_64", compiler="gcc",
                       compilerversion="13.2.0", opt="-O2", variant_id=0)
    vkey_2 = VersionKey(arch="x86_64", compiler="gcc",
                       compilerversion="13.2.0", opt="-Os", variant_id=0)

    ref_1 = registry.ref(vkey_1)
    ref_2 = registry.ref(vkey_2)
    # First record always starts at byte 0.
    assert ref_1 == "0"
    # No ``0x`` prefix, lowercase only.
    assert not ref_2.startswith("0x")
    assert ref_2 == ref_2.lower()
    # v1 had no metadata (n=4, 10 bytes); v2's record begins right after.
    assert int(ref_2, 16) == 10


def test_ref_raises_before_write_sidecar(tmp_path):
    """The byte offset is only known after ``write_sidecar`` runs.
    Calling ``ref`` earlier is a programming error — must KeyError so
    a buggy reorder surfaces loudly rather than emitting bogus refs."""
    v1 = _make_version()
    vocab = _make_vocab_for([v1])
    registry = VariantRegistry.from_versions([v1], vocab)
    vkey = VersionKey(arch="x86_64", compiler="gcc",
                     compilerversion="13.2.0", opt="-O2", variant_id=0)
    with pytest.raises(KeyError):
        registry.ref(vkey)


def test_ref_raises_for_unregistered_vkey(tmp_path):
    v1 = _make_version()
    vocab = _make_vocab_for([v1])
    registry = VariantRegistry.from_versions([v1], vocab)
    registry.write_sidecar(tmp_path, "bin")
    foreign = VersionKey(arch="aarch64", compiler="clang",
                        compilerversion="15.0.0", opt="-Os", variant_id=42)
    with pytest.raises(KeyError):
        registry.ref(foreign)


# ---------------------------------------------------------------------------
# Bin-file format — verify via read_record + decode_record round-trip
# ---------------------------------------------------------------------------


def test_bin_records_decode_back_to_input(tmp_path):
    """End-to-end: the offset ``ref`` returns is a valid memmap slice
    point; ``read_record`` + ``decode_record`` at that offset recover
    the original axis values."""
    v1 = _make_version(
        arch="amd64",  # collapses to x64 in the encoded token
        compiler="gcc",
        compilerversion="13.2.0",
        opt="-O2",
        filename="prog-x64-gcc-13.2.0-O2",
        extra_metadata={"hardening": "full"},
    )
    v2 = _make_version(
        arch="aarch64",  # collapses to arm64
        compiler="clang",
        compilerversion="15.0.0",
        opt="-Os",
        filename="prog-arm64-clang-15.0.0-Os",
        extra_metadata={"sanitizer": ["address", "undefined"]},
    )
    vocab = _make_vocab_for([v1, v2])

    registry = VariantRegistry.from_versions([v1, v2], vocab)
    bin_path = registry.write_sidecar(tmp_path, "prog")
    assert bin_path == tmp_path / "prog_variants.bin"

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")

    vkey_1 = VersionKey(arch="amd64", compiler="gcc",
                       compilerversion="13.2.0", opt="-O2", variant_id=0)
    vkey_2 = VersionKey(arch="aarch64", compiler="clang",
                       compilerversion="15.0.0", opt="-Os", variant_id=0)

    off_1 = int(registry.ref(vkey_1), 16)
    off_2 = int(registry.ref(vkey_2), 16)

    rec_1 = read_record(mmap, off_1)
    rec_2 = read_record(mmap, off_2)
    decoded_1 = decode_record(rec_1, vocab)
    decoded_2 = decode_record(rec_2, vocab)

    # arch aliases collapse in the encoded token; that is by design.
    assert decoded_1["arch"] == "x64"
    assert decoded_1["compiler"] == "gcc"
    assert decoded_1["compilerversion"] == "13.2.0"
    assert decoded_1["opt"] == "O2"
    assert decoded_1["hardening"] == ["full"]

    assert decoded_2["arch"] == "arm64"
    assert decoded_2["compiler"] == "clang"
    assert decoded_2["opt"] == "Os"
    # Multi-valued metadata: each value decodes as its own list entry,
    # values sorted alphabetically (matches encoder.encode_record).
    assert decoded_2["sanitizer"] == ["address", "undefined"]


# ---------------------------------------------------------------------------
# Slim CSV format — filename,offset (no other columns)
# ---------------------------------------------------------------------------


def test_slim_csv_has_three_columns(tmp_path):
    """The slim CSV is exactly three columns:
    ``filename, variant_id, offset``. No ``arch``, ``compiler``,
    ``version``, ``opt``, ``pkg``, ``flags``, ``length`` — those were
    the verbose-CSV columns the bin replaces."""
    v1 = _make_version(
        variant_id=0x12345678,
        filename="hello-x64-gcc-13.2.0-O2",
        extra_metadata={"hardening": "full"},
    )
    v2 = _make_version(
        opt="-Os",
        variant_id=0xCAFEBABE,
        filename="hello-x64-gcc-13.2.0-Os",
        extra_metadata={},
    )
    vocab = _make_vocab_for([v1, v2])

    registry = VariantRegistry.from_versions([v1, v2], vocab)
    registry.write_sidecar(tmp_path, "hello")

    csv_path = tmp_path / "hello_variants.csv"
    rows = _read_slim_csv_rows(csv_path)

    assert rows[0] == ["filename", "variant_id", "offset"]
    assert len(rows) == 3
    # Data rows: exactly three columns each.
    for row in rows[1:]:
        assert len(row) == 3
    # variant_id is zero-padded 8-hex.
    assert rows[1][1] == "12345678"
    assert rows[2][1] == "cafebabe"


def test_slim_csv_offsets_match_ref(tmp_path):
    """The slim CSV's ``offset`` column equals the value ``ref(vkey)``
    returns — they are the same byte offset, just sourced via a
    different surface."""
    v1 = _make_version(filename="a", extra_metadata={"k": "v"})
    v2 = _make_version(opt="-Os", filename="b", extra_metadata={})
    v3 = _make_version(
        opt="-O3", filename="c",
        extra_metadata={"hardening": ["fortify", "stack-clash"]},
    )
    vocab = _make_vocab_for([v1, v2, v3])

    registry = VariantRegistry.from_versions([v1, v2, v3], vocab)
    registry.write_sidecar(tmp_path, "trio")

    csv_path = tmp_path / "trio_variants.csv"
    rows = _read_slim_csv_rows(csv_path)

    # filename -> offset from CSV (third column now).
    csv_filename_to_offset = {row[0]: row[2] for row in rows[1:]}

    for version in (v1, v2, v3):
        vkey = VersionKey(
            arch=version.arch, compiler=version.compiler,
            compilerversion=version.compilerversion,
            opt=version.opt, variant_id=version.variant_id,
        )
        assert csv_filename_to_offset[version.filename] == registry.ref(vkey)


def test_no_versions_json_emitted(tmp_path):
    """The v3 layout removes ``<bin>_versions.json``. The registry
    must not emit one as a side effect of ``write_sidecar``."""
    v1 = _make_version(filename="x")
    vocab = _make_vocab_for([v1])
    registry = VariantRegistry.from_versions([v1], vocab)
    registry.write_sidecar(tmp_path, "x")

    assert not (tmp_path / "x_versions.json").exists()


def test_bin_record_count_matches_unique_vkeys(tmp_path):
    """Walk the bin sequentially using each record's u16 size header
    and count records — must equal the number of unique vkeys
    registered."""
    v1 = _make_version(filename="a")
    v2 = _make_version(filename="dup-of-a")  # same vkey
    v3 = _make_version(opt="-Os", filename="c")
    vocab = _make_vocab_for([v1, v3])

    registry = VariantRegistry.from_versions([v1, v2, v3], vocab)
    bin_path = registry.write_sidecar(tmp_path, "g")

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    record_count = 0
    offset = 0
    while offset < len(mmap):
        record = read_record(mmap, offset)
        n_tokens = int(record[0])
        record_count += 1
        offset += 2 + 2 * n_tokens
    assert record_count == 2  # v1 and v3; v2 deduped
    assert offset == len(mmap)  # walked the whole file exactly
