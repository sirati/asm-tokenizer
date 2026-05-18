"""Tests for ``tokenizer.variant_tokens.encoder``.

Covers the round-trip property, the uint16-overflow guard, the
empty-metadata edge case, sorted-key stability across runs, and the
split-on-first-colon decoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.variant_tokens.encoder import (
    decode_record,
    encode_record,
)
from tokenizer.variant_tokens.prefixes import build_axis_strings

from ._fakes import FakeVersionInfo, FakeVocab, make_vocab_with


def test_encode_emits_uint16_with_correct_header():
    vi = FakeVersionInfo(extra_metadata={"hardening": "full"})
    vocab = make_vocab_with(vi)
    record = encode_record(vi, vocab)
    assert record.dtype == np.uint16
    # n_tokens = 4 positional + 1 metadata = 5; array length = 1 + n.
    assert record[0] == 5
    assert record.shape == (6,)


def test_encode_empty_metadata_n_is_4():
    """Plan §"Verification" step 7: variant with empty extra_metadata
    is a 10-byte record (2 header + 4*2 axes); n=4 with NO tail."""
    vi = FakeVersionInfo(extra_metadata={})
    vocab = make_vocab_with(vi)
    record = encode_record(vi, vocab)
    assert record[0] == 4
    assert record.shape == (5,)
    # 5 u16 entries = 10 bytes on disk.
    assert record.tobytes().__len__() == 10


def test_roundtrip_no_metadata():
    vi = FakeVersionInfo(arch="x86_64", compiler="gcc",
                         compilerversion="13.2.0", opt="-O2",
                         extra_metadata={})
    vocab = make_vocab_with(vi)
    decoded = decode_record(encode_record(vi, vocab), vocab)
    assert decoded["compiler"] == "gcc"
    assert decoded["compilerversion"] == "13.2.0"
    assert decoded["opt"] == "O2"  # leading dash stripped
    # arch may be ``x64`` (1C merged) or ``x86_64`` (placeholder).
    assert decoded["arch"] in ("x64", "x86_64")
    # No metadata keys at all — plan §"Empty / edge cases".
    assert set(decoded.keys()) == {"arch", "compiler", "compilerversion", "opt"}


def test_roundtrip_multi_valued_metadata():
    vi = FakeVersionInfo(
        extra_metadata={"hardening": ["full", "fortify"]}
    )
    vocab = make_vocab_with(vi)
    decoded = decode_record(encode_record(vi, vocab), vocab)
    # Always-list shape per plan §"Always-list metadata" scope cut.
    assert decoded["hardening"] == ["fortify", "full"]


def test_roundtrip_single_valued_metadata_is_list():
    vi = FakeVersionInfo(
        extra_metadata={"sanitizer": "address"}
    )
    vocab = make_vocab_with(vi)
    decoded = decode_record(encode_record(vi, vocab), vocab)
    assert decoded["sanitizer"] == ["address"]


def test_decoder_splits_metadata_on_first_colon_only():
    """A metadata value containing ``:`` (e.g. URL-like) must round-
    trip — the decoder uses str.partition(":") which splits on the
    first colon only. Encoder doesn't restrict value chars; only key
    chars are restricted (by ``inventory.add()``)."""
    # Manually craft a record without going through encode_record
    # because we want to control the exact token string.
    vocab = FakeVocab()
    arch_id = vocab.register("arch:x64")
    comp_id = vocab.register("comp:gcc")
    cver_id = vocab.register("cver:gcc:13.2.0")
    opt_id = vocab.register("opt:O2")
    # Value contains a colon — should land entirely in the value.
    meta_id = vocab.register("flag_set:fortify:strict")
    record = np.array([5, arch_id, comp_id, cver_id, opt_id, meta_id],
                      dtype=np.uint16)
    decoded = decode_record(record, vocab)
    assert decoded["flag_set"] == ["fortify:strict"]


def test_encode_then_decode_then_encode_byte_identical():
    """Stable across runs — a re-encode of the decoded dict must
    produce the same byte stream (sorted-key metadata stability)."""
    vi = FakeVersionInfo(
        extra_metadata={"hardening": ["full", "fortify"], "sanitizer": "address"}
    )
    vocab = make_vocab_with(vi)
    first = encode_record(vi, vocab)
    second = encode_record(vi, vocab)
    assert np.array_equal(first, second)


def test_uint16_overflow_at_lookup_fires():
    """A token ID > 65535 must trigger ``AssertionError`` at lookup
    time, BEFORE any silent truncation can corrupt the record."""
    vi = FakeVersionInfo(extra_metadata={})
    vocab = FakeVocab()
    # Manually place the arch token at an out-of-range ID.
    arch_strings = build_axis_strings(vi)
    vocab.register_at(arch_strings[0], 0x10000)  # 65536
    for token in arch_strings[1:]:
        vocab.register(token)
    with pytest.raises(AssertionError, match=r"uint16 ceiling"):
        encode_record(vi, vocab)


def test_missing_token_id_fires():
    """If the unifier missed a discovery pass, the encoder must fail
    loudly instead of silently emitting a sentinel ID."""
    vi = FakeVersionInfo(extra_metadata={"hardening": "full"})
    vocab = FakeVocab()
    # Register only positional axes; omit metadata token.
    for token in build_axis_strings(vi)[:4]:
        vocab.register(token)
    with pytest.raises(AssertionError, match=r"not registered"):
        encode_record(vi, vocab)


def test_decode_rejects_non_uint16():
    vocab = FakeVocab()
    bad = np.array([4, 1, 2, 3, 4], dtype=np.uint32)
    with pytest.raises(AssertionError, match=r"uint16"):
        decode_record(bad, vocab)


def test_decode_rejects_truncated_record():
    vocab = FakeVocab()
    vocab.register("arch:x64")
    vocab.register("comp:gcc")
    # Claims n=4 but array only has 3 trailing IDs.
    bad = np.array([4, 256, 257, 258], dtype=np.uint16)
    with pytest.raises(AssertionError, match=r"smaller than header"):
        decode_record(bad, vocab)


def test_decode_rejects_truncated_header():
    vocab = FakeVocab()
    # Claims n=3 — below the 4 positional axes the layout requires.
    bad = np.array([3, 256, 257, 258], dtype=np.uint16)
    with pytest.raises(AssertionError, match=r"below"):
        decode_record(bad, vocab)


def test_mixed_type_values_encode_decode_roundtrip():
    """Plan: mixed-type values get ``str()`` coerced before sort, so
    a ``[1, "a", True]`` value list encodes deterministically."""
    vi = FakeVersionInfo(extra_metadata={"mix": [1, "a", True]})
    vocab = make_vocab_with(vi)
    decoded = decode_record(encode_record(vi, vocab), vocab)
    # str(1)="1", str(True)="True", str("a")="a" — alphabetical:
    # "1" < "True" < "a".
    assert decoded["mix"] == ["1", "True", "a"]


def test_token_ids_in_record_match_vocab_lookups():
    """Every payload u16 must equal ``vocab.get_token_id`` for the
    corresponding axis string — protects against off-by-one indexing
    in ``encode_record``'s loop."""
    vi = FakeVersionInfo(extra_metadata={"hardening": "full"})
    vocab = make_vocab_with(vi)
    record = encode_record(vi, vocab)
    expected_strings = build_axis_strings(vi)
    for i, token_str in enumerate(expected_strings, start=1):
        assert record[i] == vocab.get_token_id(token_str)
