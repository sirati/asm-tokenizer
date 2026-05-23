"""Shared occurrence iterator for the decode pass.

Single concern of this module: provide the
``mask -> positions -> per-position (position, payload_bytes)`` walk used
by the identity arm.  Per-position payload decoding (identity-vs-number
value semantics) stays in each arm; only the raw-stream traversal is
shared here.

Post-vectorization of the number arm only the identity arm consumes this
iterator; it remains a separate module both to keep the identity-arm
file focused and to leave the door open for any future consumer that
needs the same raw-stream walk.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def _iter_token_occurrences(
    raw_tokens: np.ndarray,
    runlen: np.ndarray,
    token_id: int,
) -> Iterator[Tuple[int, bytes]]:
    """Yield ``(position, payload_bytes)`` for every occurrence of ``token_id``.

    ``payload_bytes`` is the contiguous run of inline-digit bytes
    immediately following each occurrence, materialised from the uint16
    stream as a big-endian byte sequence.  Tail-position occurrences
    (no room for a trailing inline run) yield ``b''``.

    Pure read pass over ``raw_tokens`` -- never mutates the caller's
    array.  ``np.nonzero`` returns sorted ascending indices, so the
    yielded ``position`` values are stream-ascending.
    """
    n = raw_tokens.shape[0]
    mask = raw_tokens == token_id
    positions = np.nonzero(mask)[0]
    for position in positions:
        position_int = int(position)
        if position_int + 1 < n:
            inline_len = int(runlen[position_int + 1])
            payload_end = position_int + 1 + inline_len
            payload = bytes(raw_tokens[position_int + 1 : payload_end].tolist())
        else:
            payload = b""
        yield position_int, payload
