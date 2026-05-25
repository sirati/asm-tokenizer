"""Multi-chunk NUMBER-band accumulator (per-instruction grouping).

Single concern: buffer wire-form chunk pairs that belong to the same
multi-chunk NUMBER source (VC2 with K_visible > 1, F128 finite with
two chunks) and emit ONE rendered text + optional precision-entry
pair at flush time. The discriminator is the encoder invariant
"``K`` consecutive NUMBER-band tokens with the same shifted id all
live within ONE instruction and all belong to the SAME source."

API:

* :meth:`_NumberAccumulator.feed` accepts one ``(token_type, shifted_id,
  chunk_pair, value_negative)`` tuple per NUMBER token. If the
  incoming ``shifted_id`` matches the buffered source it appends;
  otherwise it auto-flushes the prior source first.
* :meth:`_NumberAccumulator.flush` empties the buffer and returns the
  rendered ``(short_text, optional precision_entry)`` of the buffered
  source. Returns ``None`` when the buffer is empty (idempotent —
  the row walker can flush at every instruction boundary unconditionally).
* :meth:`_NumberAccumulator.has_pending` exposes the
  "buffer non-empty" flag for the row walker's intra-instruction
  invariant assert (W3-17).

The accumulator does NOT own its caller's row-walker state; it owns
only the per-source chunk buffer. Flush triggers (band-switch,
instruction boundary, end-of-row) are the caller's concern.

Module boundary: depends only on
:mod:`tokenizer.aligned_data.loader.decoded._number_render` (the
renderer) and :mod:`tokenizer.tokens` (``TokenType``). No inspector
dependency — this module is reusable by any future caller that walks
a NUMBER-band stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.loader.decoded._number_render import (
    InlineNumberPrecisionEntry,
    needs_precision_expand,
    render_float_full,
    render_float_short,
    render_vc2_full,
    render_vc2_short,
)
from tokenizer.tokens import TokenType


__all__ = ["_NumberAccumulator", "AccumulatorEmission"]


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


_ChunkPair = Tuple[np.uint64, np.uint32]


@dataclass(frozen=True)
class AccumulatorEmission:
    """One flushed source: the short text + optional expansion entry.

    The text field is what the caller appends to the per-instruction
    asm-text buffer; the ``precision_entry`` is what (if non-None)
    the caller appends to the per-instruction openables buffer.
    """

    short_text: str
    precision_entry: Optional[InlineNumberPrecisionEntry]


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


@dataclass
class _NumberAccumulator:
    """Per-instruction buffer for multi-chunk NUMBER sources.

    The accumulator tracks ONE in-flight source identified by its
    ``(token_type, shifted_id)`` pair; consecutive feeds with the
    SAME shifted id extend the same source. A feed with a DIFFERENT
    shifted id auto-flushes the in-flight source first (the prior
    source is complete) and starts a new buffer for the incoming id.

    ``value_negative`` is the sign marker latched on the source's
    first chunk and applied at flush time (VC2 only — float chunks
    carry their own sign in the chunk's sign-exponent word).

    The accumulator does NOT cap K (the number of chunks in a single
    source); the renderer's reconstruction kernel
    (:func:`reconstruct_chunks`) handles any K >= 1.
    """

    _pending_token_type: Optional[TokenType] = None
    _pending_shifted_id: Optional[int] = None
    _pending_chunks: List[_ChunkPair] = field(default_factory=list)
    _pending_value_negative: bool = False

    def has_pending(self) -> bool:
        """``True`` iff at least one chunk has been fed since the last flush."""
        return bool(self._pending_chunks)

    def pending_shifted_id(self) -> Optional[int]:
        """Shifted id of the in-flight source (``None`` when empty).

        Exposed so callers can decide -- without inspecting the chunk
        buffer -- whether an upcoming NUMBER token would EXTEND the
        same source (matching shifted id) or START a new one
        (different shifted id; auto-flushed on the next :meth:`feed`).
        The row walker uses this to decide whether to pre-drain at an
        instruction boundary: a NUMBER token with a different shifted
        id at the boundary means the prior source is complete, so the
        drain is legitimate (not a W3-17 invariant violation).
        """
        return self._pending_shifted_id

    def feed(
        self,
        *,
        token_type: TokenType,
        shifted_id: int,
        chunk: _ChunkPair,
        value_negative: bool = False,
    ) -> Optional[AccumulatorEmission]:
        """Append one chunk; auto-flush if the shifted id changed.

        Returns the auto-flushed prior emission (if any) so the
        caller can append its text + precision entry before
        continuing. Returns ``None`` when the chunk extends the same
        in-flight source.

        The ``value_negative`` flag is captured on the source's first
        chunk; subsequent chunks of the same source MUST agree on the
        sign (defensive assert: encoder invariant says the postfix
        marker fires at most once per source).
        """
        prior: Optional[AccumulatorEmission] = None
        if self._pending_chunks and self._pending_shifted_id != shifted_id:
            prior = self.flush()
        if not self._pending_chunks:
            self._pending_token_type = token_type
            self._pending_shifted_id = shifted_id
            self._pending_value_negative = value_negative
        else:
            # Same source: the token type MUST match (the shifted id
            # already gates this, but the assert pins the invariant
            # against a future encoder-side drift).
            assert self._pending_token_type == token_type, (
                f"accumulator: token_type drift within same shifted_id "
                f"{shifted_id}: {self._pending_token_type!r} vs {token_type!r}"
            )
            assert self._pending_value_negative == value_negative, (
                f"accumulator: value_negative drift within same source "
                f"(shifted_id={shifted_id}): {self._pending_value_negative} "
                f"vs {value_negative}"
            )
        self._pending_chunks.append(chunk)
        return prior

    def flush(self) -> Optional[AccumulatorEmission]:
        """Render the buffered source and clear the buffer.

        Returns ``None`` when there is nothing pending — the caller
        can flush at every instruction boundary unconditionally
        without needing to guard on :meth:`has_pending`.

        Flush triggers (caller's concern):

        1. Band switch — the wire stream emitted a non-NUMBER token.
        2. Instruction boundary — the row walker finalised an
           instruction; multi-chunk sources never span instructions.
        3. End-of-row — the wire stream hit the row terminator.
        """
        if not self._pending_chunks:
            return None
        token_type = self._pending_token_type
        assert token_type is not None
        chunks = tuple(self._pending_chunks)
        value_negative = self._pending_value_negative

        if token_type is TokenType.VALUED_CONST_V2:
            short_text = _render_vc2_short_from_chunks(chunks, value_negative)
            full_text = render_vc2_full(chunks, value_negative)
        else:
            short_text = render_float_short(token_type, chunks)
            full_text = render_float_full(token_type, chunks)

        precision_entry: Optional[InlineNumberPrecisionEntry]
        if needs_precision_expand(short_text, full_text):
            precision_entry = InlineNumberPrecisionEntry(
                token_type=token_type,
                full_text=full_text,
            )
        else:
            precision_entry = None

        self._reset()
        return AccumulatorEmission(
            short_text=short_text, precision_entry=precision_entry
        )

    def _reset(self) -> None:
        self._pending_token_type = None
        self._pending_shifted_id = None
        self._pending_chunks = []
        self._pending_value_negative = False


def _render_vc2_short_from_chunks(
    chunks: Tuple[_ChunkPair, ...], value_negative: bool
) -> str:
    """VC2 short rendering from chunk sequence (multi-chunk-aware).

    The single-chunk short renderer (:func:`render_vc2_short`) takes
    the magnitude directly; for multi-chunk VC2 we reconstruct the
    full magnitude first, then format via the same renderer. This
    keeps the "short" text consistent with the "full" text whenever
    the value fits losslessly — they SHOULD diverge only when the
    short form's character width is the limiting factor (VC2
    integers do not lose precision under hex formatting at any
    magnitude, so the short / full texts ARE identical and
    :func:`needs_precision_expand` returns ``False``; W3-15 conservative
    expansion).
    """
    from tokenizer.aligned_data.loader.decoded.custom_float import (
        reconstruct_chunks,
    )

    fraction = reconstruct_chunks(list(chunks))
    if fraction.denominator != 1:
        raise ValueError(
            f"_render_vc2_short_from_chunks: VC2 reconstruction yielded "
            f"non-integer fraction {fraction}"
        )
    magnitude = int(fraction)
    if magnitude < 0:
        magnitude = -magnitude
    return render_vc2_short(magnitude, value_negative)
