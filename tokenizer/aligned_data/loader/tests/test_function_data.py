"""Tests for FunctionData.variant_tokens + full_token_stream()."""

import numpy as np

from tokenizer.aligned_data.loader.function_data import FunctionData


def _make(
    tokens: np.ndarray,
    variant_tokens: np.ndarray,
) -> FunctionData:
    return FunctionData(
        func_name="fn",
        metadata={"arch": "x64", "compiler": "gcc", "opt": "O2"},
        tokens=tokens,
        insn_runlength=np.array([len(tokens)], dtype=np.uint16),
        block_runlength=np.array([len(tokens)], dtype=np.uint16),
        variant_tokens=variant_tokens,
    )


def test_variant_tokens_field_populates():
    variant = np.array([256, 257, 258, 259], dtype=np.uint16)
    tokens = np.array([1024, 1025, 1026], dtype=np.uint16)
    fd = _make(tokens, variant)

    assert fd.variant_tokens is variant
    assert fd.variant_tokens.dtype == np.uint16
    assert len(fd.variant_tokens) == 4
    # Instruction tokens unaffected; __len__ still measures the instruction
    # stream, not the variant prefix.
    assert len(fd) == 3


def test_full_token_stream_is_concatenation():
    variant = np.array([256, 257, 258, 259], dtype=np.uint16)
    tokens = np.array([1024, 1025, 1026], dtype=np.uint16)
    fd = _make(tokens, variant)

    stream = fd.full_token_stream()
    expected = np.concatenate([variant, tokens])
    assert np.array_equal(stream, expected)
    assert stream.tolist() == [256, 257, 258, 259, 1024, 1025, 1026]


def test_full_token_stream_uint16_contiguous():
    variant = np.array([256, 257, 258, 259], dtype=np.uint16)
    tokens = np.array([1024, 1025], dtype=np.uint16)
    fd = _make(tokens, variant)

    stream = fd.full_token_stream()
    assert stream.dtype == np.uint16
    assert stream.flags["C_CONTIGUOUS"]


def test_full_token_stream_is_method_not_property():
    # Cost-signalling contract: full_token_stream is a method because the
    # concat copies memory; a property would suggest free access. Callable
    # check guards against an accidental @property regression.
    variant = np.array([256], dtype=np.uint16)
    tokens = np.array([1024], dtype=np.uint16)
    fd = _make(tokens, variant)
    assert callable(fd.full_token_stream)


def test_full_token_stream_copies_memory():
    # Concatenation must produce an independent buffer so that mutating the
    # returned stream does not corrupt the source variant_tokens or tokens
    # arrays (which are typically memmap-backed views).
    variant = np.array([256, 257], dtype=np.uint16)
    tokens = np.array([1024, 1025], dtype=np.uint16)
    fd = _make(tokens, variant)

    stream = fd.full_token_stream()
    stream[0] = 999
    assert fd.variant_tokens[0] == 256
    assert fd.tokens[0] == 1024


def test_empty_variant_tokens_round_trips():
    # Zero-length variant_tokens is the only legal degenerate shape (per
    # plan: "Zero-length only on a corrupt dataset"); the method must not
    # crash on it.
    variant = np.array([], dtype=np.uint16)
    tokens = np.array([1024, 1025, 1026], dtype=np.uint16)
    fd = _make(tokens, variant)

    stream = fd.full_token_stream()
    assert stream.dtype == np.uint16
    assert np.array_equal(stream, tokens)
