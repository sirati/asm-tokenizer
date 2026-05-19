"""Tests for ``tokenizer.aligned_data.loader.variant_resolver``.

End-to-end round-trip:
    1. Build a ``FakeVocab`` with known variant-axis tokens.
    2. ``write_record`` each variant into a synthetic ``_variants.bin``.
    3. Emit a slim CSV (``filename,offset``) matching the writer order.
    4. ``load_variants_offset_to_filename`` reads the CSV back.
    5. ``get_variant_by_ref`` resolves a hex ref to the full identity
       dict and we assert every promised field.

The fake vocab + version-info dataclasses are reused from the
``variant_tokens`` test suite to avoid hand-rolling them again.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.loader.variant_resolver import (
    get_variant_by_ref,
    load_variants_offset_to_filename,
)
from tokenizer.variant_tokens.prefixes import build_axis_strings
from tokenizer.variant_tokens.record import write_record
from tokenizer.variant_tokens.tests._fakes import (
    FakeVersionInfo,
    FakeVocab,
)


def _write_corpus(
    tmp_path: Path,
    versions: list[tuple[FakeVersionInfo, str]],
) -> tuple[Path, Path, FakeVocab, list[int]]:
    """Write a synthetic ``_variants.bin`` + slim CSV for ``versions``.

    Each entry in ``versions`` is ``(version_info, filename)``. Returns
    ``(bin_path, slim_csv_path, vocab, offsets)`` where ``offsets[i]``
    is the byte offset of the i-th record (parallel to ``versions``).
    """
    vocab = FakeVocab()
    for vi, _fname in versions:
        for token_string in build_axis_strings(vi):
            vocab.register(token_string)

    bin_path = tmp_path / "synthetic_variants.bin"
    offsets: list[int] = []
    with open(bin_path, "wb") as f:
        for vi, _fname in versions:
            offsets.append(write_record(f, vi, vocab))

    slim_csv_path = tmp_path / "synthetic_variants.csv"
    with open(slim_csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("# format=1\n")
        writer = csv.writer(f)
        writer.writerow(["filename", "offset"])
        for (vi, fname), off in zip(versions, offsets):
            writer.writerow([fname, f"{off:x}"])

    return bin_path, slim_csv_path, vocab, offsets


def test_roundtrip_single_variant(tmp_path):
    """One variant, scalar metadata → identity dict matches."""
    vi = FakeVersionInfo(
        arch="x86_64",
        compiler="gcc",
        compilerversion="13.2.0",
        opt="-O2",
        extra_metadata={"hardening": "full"},
    )
    bin_path, slim_csv_path, vocab, offsets = _write_corpus(
        tmp_path, [(vi, "hello-x64-gcc-13.2.0-O2")]
    )

    offset_to_filename = load_variants_offset_to_filename(slim_csv_path)
    assert offset_to_filename == {0: "hello-x64-gcc-13.2.0-O2"}

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    result = get_variant_by_ref(f"{offsets[0]:x}", vocab, mmap, offset_to_filename)

    # Positional axes — note arch is alias-collapsed to "x64".
    assert result["arch"] == "x64"
    assert result["compiler"] == "gcc"
    assert result["compilerversion"] == "13.2.0"
    assert result["opt"] == "O2"
    # Scalar metadata round-trips as a length-1 list (always-list
    # contract).
    assert result["hardening"] == ["full"]
    assert result["filename"] == "hello-x64-gcc-13.2.0-O2"
    # ``variant_tokens`` is the [*ids] slice (no header).
    assert isinstance(result["variant_tokens"], np.ndarray)
    assert result["variant_tokens"].dtype == np.uint16
    # n_tokens = 4 positional + 1 metadata = 5.
    assert result["variant_tokens"].shape == (5,)


def test_roundtrip_multiple_variants_distinct_offsets(tmp_path):
    """Two variants → each ref resolves to its own filename + identity.

    Also confirms the slim CSV's offset column genuinely keys the dict
    by integer (not by hex string)."""
    vi_a = FakeVersionInfo(
        arch="aarch64", compiler="clang", compilerversion="17.0.6",
        opt="-Os", extra_metadata={},
    )
    vi_b = FakeVersionInfo(
        arch="x86_64", compiler="gcc", compilerversion="13.2.0",
        opt="-O3",
        extra_metadata={"hardening": ["full", "fortify"], "lto": "thin"},
    )
    bin_path, slim_csv_path, vocab, offsets = _write_corpus(
        tmp_path, [(vi_a, "hello-arm64-clang"), (vi_b, "hello-x64-gcc")]
    )

    offset_to_filename = load_variants_offset_to_filename(slim_csv_path)
    # Both offsets present and distinct.
    assert set(offset_to_filename.keys()) == set(offsets)
    assert offset_to_filename[offsets[0]] == "hello-arm64-clang"
    assert offset_to_filename[offsets[1]] == "hello-x64-gcc"

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")

    result_a = get_variant_by_ref(f"{offsets[0]:x}", vocab, mmap, offset_to_filename)
    assert result_a["arch"] == "arm64"
    assert result_a["compiler"] == "clang"
    assert result_a["opt"] == "Os"
    assert result_a["filename"] == "hello-arm64-clang"
    # Empty metadata → no metadata keys (plan §"Verification" 7).
    assert "hardening" not in result_a
    assert result_a["variant_tokens"].shape == (4,)

    result_b = get_variant_by_ref(f"{offsets[1]:x}", vocab, mmap, offset_to_filename)
    assert result_b["arch"] == "x64"
    assert result_b["compiler"] == "gcc"
    assert result_b["opt"] == "O3"
    # Multi-valued key decodes to a sorted list of strings.
    assert result_b["hardening"] == ["fortify", "full"]
    assert result_b["lto"] == ["thin"]
    assert result_b["filename"] == "hello-x64-gcc"
    # 4 positional + 2 hardening + 1 lto = 7 tokens.
    assert result_b["variant_tokens"].shape == (7,)


def test_variant_tokens_drops_header_for_stream_concat(tmp_path):
    """``variant_tokens`` excludes the leading ``n_tokens`` size header.

    The header is record-layout metadata; it would be nonsense if
    concatenated into the instruction token stream as a vocab ID.
    Verifies the deliberate design choice documented at the top of
    ``variant_resolver.py``."""
    vi = FakeVersionInfo(extra_metadata={})
    bin_path, slim_csv_path, vocab, offsets = _write_corpus(
        tmp_path, [(vi, "minimal")]
    )
    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    result = get_variant_by_ref("0", vocab, mmap, load_variants_offset_to_filename(slim_csv_path))

    # Each ID in ``variant_tokens`` must be a real vocab entry — the
    # round-trip ``vocab.get_token_str(id)`` must yield a non-empty
    # string. If the header (value 4) had been left in, it would map
    # to "" because no token was registered at ID 4.
    for token_id in result["variant_tokens"]:
        assert vocab.get_token_str(int(token_id)) != "", (
            f"token id {token_id} not in vocab — header likely leaked "
            "into variant_tokens"
        )


def test_load_variants_offset_to_filename_missing_returns_empty(tmp_path):
    """Legacy datasets without the slim CSV → empty dict, no crash."""
    assert load_variants_offset_to_filename(tmp_path / "absent.csv") == {}


def test_load_variants_offset_to_filename_hex_offset_parse(tmp_path):
    """Slim CSV's hex offsets parse correctly across the byte range
    (catches a sloppy ``int(s)`` instead of ``int(s, 16)``)."""
    slim_csv_path = tmp_path / "wide.csv"
    with open(slim_csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("# format=1\n")
        writer = csv.writer(f)
        writer.writerow(["filename", "offset"])
        writer.writerow(["a", "0"])
        writer.writerow(["b", "a"])         # 10
        writer.writerow(["c", "100"])       # 256
        writer.writerow(["d", "deadbeef"])  # large

    table = load_variants_offset_to_filename(slim_csv_path)
    assert table == {0: "a", 10: "b", 256: "c", 0xDEADBEEF: "d"}


def test_get_variant_by_ref_missing_filename_raises(tmp_path):
    """An offset present in the bin but not in the slim CSV is a
    corruption signal — KeyError surfaces it instead of silently
    returning ``None``/empty."""
    vi = FakeVersionInfo(extra_metadata={})
    bin_path, _slim_csv_path, vocab, _offsets = _write_corpus(
        tmp_path, [(vi, "present")]
    )
    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    # Deliberately empty filename table.
    with pytest.raises(KeyError):
        get_variant_by_ref("0", vocab, mmap, {})


def test_variant_tokens_survives_memmap_close(tmp_path):
    """Returned ``variant_tokens`` must own its buffer.

    ``read_record`` slices a memmap view; if we returned that view
    directly, the array would dangle once the enclosing
    ``BinarySession`` closes the memmap and any post-batch
    dereference (e.g. ``FunctionData.full_token_stream()`` in a
    training loop) would segfault on freed pages. The resolver
    therefore copies into an owning buffer at the session-boundary
    handoff; this test pins that contract."""
    vi = FakeVersionInfo(extra_metadata={})
    bin_path, slim_csv_path, vocab, _offsets = _write_corpus(
        tmp_path, [(vi, "lonely")]
    )
    offset_to_filename = load_variants_offset_to_filename(slim_csv_path)

    mmap = np.memmap(bin_path, dtype=np.uint8, mode="r")
    result = get_variant_by_ref("0", vocab, mmap, offset_to_filename)
    tokens = result["variant_tokens"]
    # Releasing the memmap MUST not invalidate tokens.
    mmap._mmap.close()
    del mmap
    # If tokens were a view, the next line would touch freed memory.
    assert int(tokens.sum()) >= 0
    assert tokens.flags.owndata
