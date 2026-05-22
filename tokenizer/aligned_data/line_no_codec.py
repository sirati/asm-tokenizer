"""Compact base64 codec for 1-indexed line numbers.

Used by the function-names indirection: each occurrence of a function
name inside the section CSVs is replaced with the base64 of its line
number in ``<binary>_function_names.txt``. Encoded values are the
shortest possible byte representation of the integer (leading zero
bytes stripped) urlsafe-base64'd with ``=`` padding stripped.

Both encoder and decoder live in this module so writer and reader
agree on the exact wire string. ``n >= 1`` is enforced because line
numbers are 1-indexed; the empty CSV string round-trips to ``[]``.
"""

from __future__ import annotations

import base64
from typing import List, Sequence


def encode_line_no(n: int) -> str:
    """Compact urlsafe-base64 of ``n`` (no leading zeros, no padding).

    ``n`` must be ``>= 1`` (line numbers are 1-indexed). The byte
    representation uses the minimum number of bytes needed
    (``(n.bit_length() + 7) // 8`` or ``1`` for ``n == 1``).
    """
    if n < 1:
        raise ValueError(f"line number must be >= 1, got {n}")
    n_bytes = (n.bit_length() + 7) // 8 or 1
    raw = n.to_bytes(n_bytes, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_line_no(s: str) -> int:
    """Inverse of :func:`encode_line_no`.

    Pads ``=`` back to a multiple of 4 before urlsafe-base64 decoding,
    then interprets the bytes as a big-endian unsigned integer.
    """
    pad = (-len(s)) % 4
    raw = base64.urlsafe_b64decode(s.encode("ascii") + b"=" * pad)
    return int.from_bytes(raw, "big")


def encode_line_nos_csv(ns: Sequence[int]) -> str:
    """Comma-joined :func:`encode_line_no` for a sequence of line nos."""
    return ",".join(encode_line_no(n) for n in ns)


def decode_line_nos_csv(s: str) -> List[int]:
    """Inverse of :func:`encode_line_nos_csv`; empty string -> ``[]``."""
    if not s:
        return []
    return [decode_line_no(part) for part in s.split(",")]
