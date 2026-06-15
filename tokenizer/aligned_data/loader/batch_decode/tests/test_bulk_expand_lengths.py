"""Equivalence: bulk contributing-length scan vs scalar ``expand_tokens``.

The expansion semantics are owned by :mod:`.._expand_tokens`; the bulk
scan (:mod:`.._bulk_expand_lengths`) must agree with it on every
stream shape -- plain real tokens, digit runs, sign markers, VC2
multi-chunk sources (incl. the empty-payload edge), and finite vs
NaN/Inf F128 sources.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import _bulk_expand_lengths
from tokenizer.aligned_data.loader.batch_decode._bulk_expand_lengths import (
    bulk_contributing_body_lengths,
    bulk_contributing_geometry,
)
from tokenizer.aligned_data.loader.batch_decode._expand_tokens import (
    expand_tokens,
)
from tokenizer.aligned_data.loader.batch_decode._surviving_counts import (
    count_surviving,
)
from tokenizer.aligned_data.loader.decoded._inline_decode_state import (
    build_inline_decode_state,
)
from tokenizer.tokens import Category


_VC2 = 257
_F128 = 263
_IDENTITY = 264  # first IDENTITY-block carrier (BLOCK_V2)
_REAL = 300  # arbitrary id above the eager block start


def _scalar_body_length(raw: np.ndarray) -> int:
    """Contributing body length via the scalar source of truth."""
    state = build_inline_decode_state(raw, format_version=1)
    stub = SimpleNamespace(state=state, encounter_category=Category.LOCAL_FUNC)
    return expand_tokens(stub).predicted_full_length - 1


def _scalar_body_geometry(raw: np.ndarray) -> tuple:
    """Decode-and-count oracle: ``(body_len, id_count, value_chunk_count)``.

    Actually expands the record via the scalar :func:`expand_tokens`
    path, then counts the IDENTITY-band and NUMBER-band entries the real
    decode produces -- using :func:`count_surviving`, the decode path's
    own band counter -- over the BODY (slot 0 is the prepended self-token,
    which the bulk scan deliberately excludes). This is independent of
    the bulk derivation: it counts produced entries, not raw carriers.
    """
    state = build_inline_decode_state(raw, format_version=1)
    stub = SimpleNamespace(state=state, encounter_category=Category.LOCAL_FUNC)
    expanded = expand_tokens(stub)
    body = expanded.expanded_token_ids[1:]
    counts = count_surviving(body, body.shape[0])
    return (
        expanded.predicted_full_length - 1,
        counts.surviving_identity_count,
        counts.surviving_number_chunk_count,
    )


def _digits(rng: np.random.Generator, n: int) -> List[int]:
    return [int(d) for d in rng.integers(0, 256, size=n)]


def _random_stream(rng: np.random.Generator) -> np.ndarray:
    """One synthetic raw stream; first position is always a real token."""
    toks: List[int] = [_REAL]
    for _ in range(int(rng.integers(2, 30))):
        kind = int(rng.integers(0, 7))
        if kind == 6:
            # IDENTITY carrier with a 0/1/2-byte inline payload (the only
            # widths the v2 codec emits for identity tokens; never
            # promoted -- one expanded position each).
            toks.append(_IDENTITY + int(rng.integers(0, 8)))
            toks.extend(_digits(rng, int(rng.integers(0, 3))))
        elif kind == 0:
            toks.append(_REAL + int(rng.integers(0, 50)))
        elif kind == 1:
            # Real carrier-ish token followed by a digit run + optional
            # sign marker (id 256).
            toks.append(_REAL)
            toks.extend(_digits(rng, int(rng.integers(1, 12))))
            if rng.integers(0, 2):
                toks.append(256)
        elif kind == 2:
            # VC2 with a payload run of 1..20 digits (multi-chunk above 8).
            toks.append(_VC2)
            toks.extend(_digits(rng, int(rng.integers(1, 21))))
        elif kind == 3:
            # VC2 empty-payload edge: carrier directly followed by a
            # real token (still 1 chunk; no extra slots).
            toks.append(_VC2)
            toks.append(_REAL)
        elif kind == 4:
            # Finite F128: 16 payload digits, high u16 not all-ones.
            toks.append(_F128)
            toks.extend([0x3F, 0xF0] + _digits(rng, 14))
        else:
            # NaN/Inf F128: high u16 (sign-stripped) all-ones.
            toks.append(_F128)
            toks.extend([0xFF, 0xFF] + _digits(rng, 14))
    toks.append(_REAL)  # keep carriers away from the record tail
    return np.array(toks, dtype=np.uint16)


def _pack(streams: List[np.ndarray]):
    """Serialize streams into one buffer with junk gaps between them."""
    buf = bytearray()
    starts, counts = [], []
    rng = np.random.default_rng(99)
    for s in streams:
        buf += bytes(rng.integers(0, 256, size=int(rng.integers(0, 5)) * 2))
        starts.append(len(buf))
        counts.append(s.size)
        buf += s.astype("<u2").tobytes()
    data = np.frombuffer(bytes(buf), dtype=np.uint8)
    return data, np.asarray(starts, dtype=np.int64), np.asarray(counts)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_bulk_matches_scalar_expand_tokens(seed: int) -> None:
    rng = np.random.default_rng(seed)
    streams = [_random_stream(rng) for _ in range(25)]
    data, starts, counts = _pack(streams)

    got = bulk_contributing_body_lengths(data, starts, counts)
    expected = np.array([_scalar_body_length(s) for s in streams])
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_bulk_geometry_matches_decode_and_count(seed: int) -> None:
    """All three geometry axes match the decode-and-count oracle."""
    rng = np.random.default_rng(seed)
    streams = [_random_stream(rng) for _ in range(25)]
    data, starts, counts = _pack(streams)

    geom = bulk_contributing_geometry(data, starts, counts)
    expected = np.array([_scalar_body_geometry(s) for s in streams])

    assert np.array_equal(geom.body_len, expected[:, 0])
    assert np.array_equal(geom.id_count, expected[:, 1])
    assert np.array_equal(geom.value_chunk_count, expected[:, 2])
    # The thin wrapper stays byte-identical to the geometry body axis.
    assert np.array_equal(
        bulk_contributing_body_lengths(data, starts, counts), geom.body_len
    )


def test_zero_token_records_contribute_zero() -> None:
    rng = np.random.default_rng(3)
    streams = [_random_stream(rng), _random_stream(rng)]
    data, starts, counts = _pack(streams)
    # Splice empty records between, before, and AFTER the real ones
    # (the trailing empty record exercises rec_start == total).
    starts2 = np.array([starts[0], starts[0], starts[1], starts[1]])
    counts2 = np.array([0, counts[0], counts[1], 0])
    got = bulk_contributing_body_lengths(data, starts2, counts2)
    assert got[0] == 0 and got[3] == 0
    assert got[1] == _scalar_body_length(streams[0])
    assert got[2] == _scalar_body_length(streams[1])


def test_chunking_is_transparent(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(11)
    streams = [_random_stream(rng) for _ in range(20)]
    data, starts, counts = _pack(streams)
    whole = bulk_contributing_body_lengths(data, starts, counts)
    monkeypatch.setattr(_bulk_expand_lengths, "_CHUNK_TOKENS", 16)
    chunked = bulk_contributing_body_lengths(data, starts, counts)
    assert np.array_equal(whole, chunked)


def test_vc2_at_tail_raises() -> None:
    raw = np.array([_REAL, _VC2], dtype=np.uint16)
    data, starts, counts = _pack([raw])
    with pytest.raises(AssertionError, match="VC2 carrier at the last"):
        bulk_contributing_body_lengths(data, starts, counts)


def test_f128_near_tail_raises() -> None:
    raw = np.array([_REAL, _F128, 0x3F], dtype=np.uint16)
    data, starts, counts = _pack([raw])
    with pytest.raises(AssertionError, match="F128 carrier within 2"):
        bulk_contributing_body_lengths(data, starts, counts)
