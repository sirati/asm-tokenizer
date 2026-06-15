"""#27 catalog<->vocab fingerprint safety net.

The memmap builder stamps an 8-byte identity fingerprint of the unified
vocab into each ``_data.bin`` prelude (the reserved slot); the loader
recomputes it from the vocab it actually loaded and HARD-FAILS when a
catalog is decoded with a DIFFERENT (same-format-version) vocab -- the bin
stores unified-vocab token ids for the whole stream, so a wrong vocab
silently mis-decodes EVERY token, not just the variant axes.
``NO_FINGERPRINT`` (all-zero) is the backward-compatible soft-skip for
bins / vocabs that predate the fingerprint.

Covers: the prelude codec round-trip, the file-hash helper's
distinguishing power, and the session-level verify (match passes,
mismatch raises, unfingerprinted soft-skips).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tokenizer.aligned_data.memmap_format import (
    DATA_BIN_PRELUDE_SIZE,
    NO_FINGERPRINT,
    assert_data_bin_prelude,
    encode_data_bin_prelude,
    read_bin_prelude_reserved,
)
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.loader.unified_vocab_gate import (
    compute_vocab_fingerprint,
)

# Reuse the on-disk synthetic-binary fixture the session exception-safety
# suite uses (full loadable _data.bin + sections + metadata).
from tokenizer.aligned_data.loader.tests.test_session_exception_safety import (  # noqa: F401
    synthetic_binary,
)

_FP_A = b"\xaa\xbb\xcc\xdd\x01\x02\x03\x04"
_FP_B = b"\x11\x22\x33\x44\x55\x66\x77\x88"


# --- prelude codec -------------------------------------------------------


def test_prelude_carries_fingerprint_round_trip():
    prelude = encode_data_bin_prelude(_FP_A)
    assert len(prelude) == DATA_BIN_PRELUDE_SIZE
    assert_data_bin_prelude(prelude)  # still a valid DATA prelude (magic+version)
    assert read_bin_prelude_reserved(prelude) == _FP_A


def test_prelude_default_is_no_fingerprint():
    assert read_bin_prelude_reserved(encode_data_bin_prelude()) == NO_FINGERPRINT


# --- file-hash helper ----------------------------------------------------


def test_compute_vocab_fingerprint_stable_and_distinguishing(tmp_path: Path):
    a = tmp_path / "a.csv"
    a.write_bytes(b"vocabulary,one,two\nformat_version,1\n")
    b = tmp_path / "b.csv"
    b.write_bytes(b"vocabulary,one,three\nformat_version,1\n")
    fa = compute_vocab_fingerprint(a)
    fb = compute_vocab_fingerprint(b)
    assert len(fa) == 8 == len(fb)
    assert fa != fb  # different vocab content -> different fingerprint
    assert compute_vocab_fingerprint(a) == fa  # stable across calls
    assert fa != NO_FINGERPRINT


# --- session-level verify ------------------------------------------------


def _restamp_data_bin(base_path: Path, binary_name: str, fingerprint: bytes) -> None:
    """Overwrite the first 16 prelude bytes of the binary's _data.bin with a
    fingerprinted prelude (records + trailer untouched)."""
    path = base_path / f"{binary_name}_data.bin"
    raw = bytearray(path.read_bytes())
    raw[:DATA_BIN_PRELUDE_SIZE] = encode_data_bin_prelude(fingerprint)
    path.write_bytes(bytes(raw))


def test_session_accepts_matching_vocab_fingerprint(synthetic_binary):
    fb = synthetic_binary
    _restamp_data_bin(fb["base_path"], fb["binary_name"], _FP_A)
    fb["vocab"]._vocab_fingerprint = _FP_A
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        sess.load_matched(0)  # opens _data.bin -> verify passes (match)


def test_session_rejects_wrong_vocab_fingerprint(synthetic_binary):
    fb = synthetic_binary
    _restamp_data_bin(fb["base_path"], fb["binary_name"], _FP_A)
    fb["vocab"]._vocab_fingerprint = _FP_B  # mismatched vocab
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            sess.load_matched(0)


def test_session_soft_skips_unfingerprinted_catalog(synthetic_binary):
    """A pre-#27 _data.bin (NO_FINGERPRINT in the prelude) loads even when
    the vocab IS fingerprinted -- backward compatibility."""
    fb = synthetic_binary
    # fixture's _data.bin is written with the default (NO_FINGERPRINT) prelude.
    fb["vocab"]._vocab_fingerprint = _FP_A
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        sess.load_matched(0)  # no raise: catalog unfingerprinted -> soft-skip
