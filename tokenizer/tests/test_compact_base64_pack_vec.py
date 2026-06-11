"""Byte-identity + round-trip pins for the vectorised bit packer.

``_pack_bits_vec`` historically scattered shifted values into 32-bit
words and corrupted any width that straddled a word boundary — exactly
11, and everything above 12 (its ``assert bits <= 12`` ceiling forced
big-binary packs of bits == 11 or >= 13 onto a per-token Python loop).
The rewrite expands values into an MSB-first bit matrix and folds it
with ``np.packbits``, mirroring the 5-byte-gather decoder.

The wire format MUST NOT change: existing CSVs were written by the
scalar writer, so the vec output is pinned byte-for-byte against a
frozen copy of that scalar implementation (``_pack_bits_scalar_ref``
below), at every width the helper accepts (1..64) — a superset of the
2..33 range the compact header can express — plus pack→unpack identity
through both public decoders at every header-expressible width.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from tokenizer.compact_base64_utils import (
    _pack_bits,
    _pack_bits_vec,
    base64_to_ndarray,
    base64_to_ndarray_vec,
    ndarray_to_base64,
)


def _pack_bits_scalar_ref(
    values: np.ndarray,
    bits_per_val: int,
    prefix: List[Tuple[int, int]] | None = None,
) -> bytes:
    """FROZEN copy of the pre-rewrite scalar packer (the path every
    existing CSV was written through for bits == 11 and >= 13). Reference
    implementation only — do not "fix" or modernise."""
    buf = 0
    buf_bits = 0
    out = bytearray()

    def _write(val: int, n: int):
        nonlocal buf, buf_bits
        buf = (buf << n) | val
        buf_bits += n
        while buf_bits >= 8:
            shift = buf_bits - 8
            out.append((buf >> shift) & 0xFF)
            buf_bits -= 8
            buf &= (1 << buf_bits) - 1 if buf_bits else 0

    if prefix:
        for val, n in prefix:
            if val >= (1 << n):
                raise ValueError(f"{val} will not fit in {n} bits")
            _write(val, n)

    for v in values:
        _write(int(v), bits_per_val)

    if buf_bits:  # tail (pad right with zeros)
        out.append(buf << (8 - buf_bits) & 0xFF)

    return bytes(out)


def _values_for(bits: int, size: int, seed: int) -> np.ndarray:
    """Random uint64 values that exactly need ``bits`` (max forced in)."""
    max_val = (1 << bits) - 1
    rng = np.random.default_rng(seed)
    values = rng.integers(0, max_val, size=size, dtype=np.uint64, endpoint=True)
    if size:
        values[0] = max_val  # force the full width
    return values


# A header-shaped prefix (5 + 3 + variable length bits) keeps the payload
# start NON-byte-aligned for every len_bits not divisible by 8 — the same
# alignment regime ``ndarray_to_base64`` produces.
def _header_prefix(bits: int, n: int) -> List[Tuple[int, int]]:
    len_bits = 4
    while (1 << len_bits) <= n:
        len_bits += 4
    # bits_code clamped: widths > 33 are not header-expressible but the
    # raw packers are still identity-tested there; the prefix only needs
    # to provide a realistic (mis)alignment, not a valid header.
    return [(min(31, max(0, bits - 2)), 5), (len_bits // 4 - 1, 3), (n, len_bits)]


@pytest.mark.parametrize("bits", list(range(1, 65)))
@pytest.mark.parametrize("size", [0, 1, 2, 3, 17, 393, 1000])
def test_vec_packer_byte_identical_to_scalar(bits: int, size: int) -> None:
    """The vec output equals the frozen scalar writer bit-for-bit, with
    and without a (non-byte-aligned) prefix."""
    values = _values_for(bits, size, seed=bits * 10007 + size)

    assert _pack_bits_vec(values, bits) == _pack_bits_scalar_ref(values, bits)

    prefix = _header_prefix(bits, size)
    assert _pack_bits_vec(values, bits, prefix) == _pack_bits_scalar_ref(
        values, bits, prefix
    )
    # The dispatcher must route this width to the vec path AND stay
    # byte-identical through it.
    assert _pack_bits(values, bits, prefix) == _pack_bits_scalar_ref(
        values, bits, prefix
    )


@pytest.mark.parametrize("bits", list(range(2, 34)))
@pytest.mark.parametrize("size", [0, 1, 3, 17, 393, 1000])
def test_public_round_trip_every_header_width(bits: int, size: int) -> None:
    """pack -> unpack identity through BOTH decoders at every width the
    compact header can express (2..33), incl. empty / single / odd sizes."""
    values = _values_for(bits, size, seed=bits * 31 + size)

    encoded = ndarray_to_base64(values)
    np.testing.assert_array_equal(base64_to_ndarray(encoded), values)
    np.testing.assert_array_equal(base64_to_ndarray_vec(encoded), values)


@pytest.mark.parametrize("bits", list(range(2, 34)))
def test_encoder_output_byte_identical_to_legacy(bits: int) -> None:
    """``ndarray_to_base64`` emits the SAME base64 string the legacy
    scalar-packed encoder produced — the on-disk CSV format is unchanged."""
    values = _values_for(bits, 393, seed=bits)

    prefix = _header_prefix(bits, values.size)
    legacy_raw = _pack_bits_scalar_ref(values, bits, prefix)

    import base64

    assert ndarray_to_base64(values) == base64.b64encode(legacy_raw).decode("ascii")


def test_bits_11_cross_word_regression_values() -> None:
    """The exact value pattern the old vec path corrupted at bits=11
    (272 -> 16, 1000 -> 488) now packs correctly through the vec path."""
    values = np.zeros(393, dtype=np.uint64)
    values[:256] = np.arange(256)
    values[256] = 264
    values[257] = 272
    values[258] = 530
    values[262] = 1000

    assert _pack_bits_vec(values, 11) == _pack_bits_scalar_ref(values, 11)
    decoded = base64_to_ndarray_vec(ndarray_to_base64(values))
    np.testing.assert_array_equal(decoded.astype(np.uint64), values)
