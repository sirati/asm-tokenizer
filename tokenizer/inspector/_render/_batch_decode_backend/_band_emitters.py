"""Per-band emit primitives for the BatchDecodeBackend row walker.

Single concern: shifted-id -> :class:`LineItem` emission for the
INSTR_REP and NUMBER bands. The IDENTITY band lives in
:mod:`._row_walk` (it owns the per-Category dispatch + block /
function-category branching that need walker state). Each emitter
mutates the supplied :class:`_WalkState`-shaped accumulator in place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._number_decode._band_constants import (
    _NUMBER_BAND_LO_SHIFTED,
    _NUMBER_BLOCK_TOKEN_TYPES,
)
from tokenizer.aligned_data.loader.decoded.number_hex_format import (
    chunks_to_hex_bits,
)
from tokenizer.inspector._render._protocol import AsmLine, LineItem
from tokenizer.inspector._render._token_text import substitute_mem_chars
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

from ._arch_prefix import strip_arch_prefix


__all__ = [
    "emit_instr_rep",
    "emit_number",
    "MULTI_CHUNK_SHIFTED_IDS",
]


# Stage-2 strip-shift (``-256``) applied at token assembly; reverse via
# ``+_V2_RESERVED_DIGIT_COUNT`` to recover the unified-vocab id.
_V2_RESERVED_DIGIT_COUNT: int = VocabularyManager._V2_RESERVED_DIGIT_COUNT


# Shifted ids whose source can span multiple NUMBER-band tokens on the
# wire (VC2 with K_visible chunks, F128 finite with 2 chunks). Derived
# from ``_NUMBER_BLOCK_TOKEN_TYPES`` so a NUMBER-block layout shift
# updates this set without touching the walker. Phase-1 trailing-chunk
# detection (plan #17): a NUMBER token whose shifted id matches the
# immediately-preceding NUMBER token's id AND lives in this set is a
# trailing chunk of the same source -- the walker emits ``"..."`` for
# it instead of feeding the chunk pair to :func:`chunks_to_hex_bits`.
# Single-chunk types (F16/BF16/F32/F64/F80) are excluded; consecutive
# tokens of those types come from distinct sources and each renders.
MULTI_CHUNK_SHIFTED_IDS: frozenset[int] = frozenset(
    1 + i
    for i, tt in enumerate(_NUMBER_BLOCK_TOKEN_TYPES)
    if tt.name in ("VALUED_CONST_V2", "FLOAT128")
)


def emit_instr_rep(
    items: List[LineItem],
    *,
    shifted_id: int,
    vocab_manager: VocabularyManager,
    arch_prefixes: tuple[str, ...] = (),
) -> None:
    """INSTR_REP band: vocab lookup -> polished display -> :class:`AsmLine`.

    Two display transforms after the raw vocab-string lookup:

    1. ``substitute_mem_chars`` collapses the six
       :class:`MemoryOperandSymbol` vocab strings (``MEM_OPEN_BRACKET``
       etc.) to their display chars (``[`` etc.) -- matches the FTL
       backend's display per cluster #3 of the W3-3 W4-amended plan.
    2. ``strip_arch_prefix`` removes the most-specific matching arch
       prefix (per-ISA, family, then unified) so BatchDecode display
       mirrors :meth:`PlatformTokenInner.to_asm_like`'s prefix-stripped
       form.

    ``arch_prefixes`` defaults to ``()``: callers that haven't plumbed
    the arch through (test fixtures, legacy paths) get the substitution
    transform without arch elision.
    """
    original_id = int(shifted_id) + _V2_RESERVED_DIGIT_COUNT
    raw = vocab_manager.get_token_str(original_id)
    display = strip_arch_prefix(substitute_mem_chars(raw), arch_prefixes)
    items.append(AsmLine(text=display))


def emit_number(
    items: List[LineItem],
    *,
    shifted_id: int,
    numbers_sig: np.ndarray,
    numbers_se: np.ndarray,
    num_cursor: int,
    last_number_shifted_id: int,
) -> int:
    """NUMBER band: consume one chunk pair, emit hex or trailing slot.

    Returns the post-emit ``num_cursor``; callers update their state's
    cursor + ``last_number_shifted_id`` accordingly.

    Phase-1 trailing-chunk detection (plan #17): for shifted ids in
    :data:`MULTI_CHUNK_SHIFTED_IDS` (VC2, F128), consecutive same-id
    slots on the wire belong to the SAME source and only the lead
    chunk goes through :func:`chunks_to_hex_bits`; every trailing slot
    emits ``AsmLine("...")``. This is the encoder's stream order: a
    multi-chunk source emits K consecutive tokens of the same shifted
    id before any other token interrupts. Phase 2 (plan section 11)
    will replace this stream-position heuristic with the chunk-count
    sidecar so full multi-chunk reconstruction is possible.
    """
    band_index = int(shifted_id) - _NUMBER_BAND_LO_SHIFTED
    token_type: TokenType = _NUMBER_BLOCK_TOKEN_TYPES[band_index]
    sig = numbers_sig[num_cursor]
    se = numbers_se[num_cursor]
    is_trailing_chunk = (
        shifted_id == last_number_shifted_id
        and shifted_id in MULTI_CHUNK_SHIFTED_IDS
    )
    if is_trailing_chunk:
        items.append(AsmLine(text="..."))
    else:
        items.append(AsmLine(text=chunks_to_hex_bits(token_type, sig, se)))
    return num_cursor + 1
