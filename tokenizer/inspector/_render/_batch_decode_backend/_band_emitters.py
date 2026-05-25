"""Per-band emit primitives for the BatchDecodeBackend row walker.

Single concern: shifted-id -> :class:`LineItem` emission for the
INSTR_REP and NUMBER bands. The IDENTITY band lives in
:mod:`._row_walk` (it owns the per-Category dispatch + block /
function-category branching that need walker state). Each emitter
mutates the supplied :class:`_WalkState`-shaped accumulator in place.
"""

from __future__ import annotations

from typing import List

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._number_decode._band_constants import (
    _NUMBER_BAND_LO_SHIFTED,
    _NUMBER_BLOCK_TOKEN_TYPES,
)
from tokenizer.aligned_data.loader.decoded._number_render_collector import (
    AccumulatorEmission,
    _NumberAccumulator,
)
from tokenizer.inspector._render._protocol import AsmLine, LineItem
from tokenizer.inspector._render._token_text import substitute_mem_chars
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

from ._arch_prefix import strip_arch_prefix


__all__ = [
    "emit_instr_rep",
    "emit_number",
    "flush_accumulator_into",
    "MULTI_CHUNK_SHIFTED_IDS",
]


# Stage-2 strip-shift (``-256``) applied at token assembly; reverse via
# ``+_V2_RESERVED_DIGIT_COUNT`` to recover the unified-vocab id.
_V2_RESERVED_DIGIT_COUNT: int = VocabularyManager._V2_RESERVED_DIGIT_COUNT


# Shifted ids whose source CAN span multiple NUMBER-band tokens on the
# wire (VC2 with K_visible chunks, F128 finite with 2 chunks). Derived
# from ``_NUMBER_BLOCK_TOKEN_TYPES`` so a NUMBER-block layout shift
# updates this set without touching the walker. Single-chunk types
# (F16/BF16/F32/F64/F80) are EXCLUDED; for those types consecutive
# same-shifted-id tokens are DISTINCT sources and the walker MUST
# force-flush the accumulator between feeds so they do not collapse
# into one mis-grouped multi-chunk source.
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
    accumulator: _NumberAccumulator,
) -> int:
    """NUMBER band: feed one chunk pair into the accumulator.

    Drives :class:`_NumberAccumulator` (the SSOT for multi-chunk
    grouping + short / full text rendering). Multi-chunk-capable
    shifted ids (VC2, F128; :data:`MULTI_CHUNK_SHIFTED_IDS`) extend
    the accumulator across consecutive feeds with the same shifted id;
    a shifted-id change auto-flushes the prior source and returns its
    emission for in-place appending.

    Single-chunk shifted ids (F16/BF16/F32/F64/F80) are force-flushed
    immediately after the feed so two consecutive same-shifted-id
    feeds do NOT collapse into a mis-grouped multi-chunk source. The
    force-flush is the correctness gate for cluster #21 / R1-Audit
    L-4: the accumulator's API groups by shifted id alone (it cannot
    distinguish "next single-chunk source" from "trailing chunk"); the
    caller injects that knowledge via the force-flush after every
    non-multi-chunk feed.

    Returns the post-emit ``num_cursor``; callers update their state's
    cursor accordingly. The caller is responsible for flushing the
    accumulator at instruction boundaries (W3-17) and at end-of-row
    (cluster #21 H-4 cut-variant tolerance); see
    :func:`flush_accumulator_into`.
    """
    band_index = int(shifted_id) - _NUMBER_BAND_LO_SHIFTED
    token_type: TokenType = _NUMBER_BLOCK_TOKEN_TYPES[band_index]
    chunk = (numbers_sig[num_cursor], numbers_se[num_cursor])
    prior = accumulator.feed(
        token_type=token_type, shifted_id=int(shifted_id), chunk=chunk,
    )
    if prior is not None:
        _append_emission(items, prior)
    if int(shifted_id) not in MULTI_CHUNK_SHIFTED_IDS:
        # Single-chunk source: force-flush so a subsequent same-id
        # feed starts a fresh source instead of extending this one.
        forced = accumulator.flush()
        if forced is not None:
            _append_emission(items, forced)
    return num_cursor + 1


def flush_accumulator_into(
    items: List[LineItem], *, accumulator: _NumberAccumulator,
) -> None:
    """Flush the accumulator and append any emission to ``items``.

    Idempotent: a no-pending accumulator yields ``None`` and the
    function returns without touching ``items``. Callers invoke this
    at:

    1. Band switches off NUMBER (INSTR_REP / IDENTITY tokens):
       within-instruction NUMBER groupings end at the next non-NUMBER
       token; every instruction starts with an INSTR_REP mnemonic, so
       a band switch off NUMBER is equivalent to an instruction
       boundary in practice. This matches W3-17's flush trigger #2
       (instruction boundary) without the row walker needing a full
       instruction-runlength state machine (R2a's concern).
    2. End-of-row (the ``shifted_id == 0`` break or end of columns):
       cut-variant tolerance per cluster #21 H-4 -- pending multi-
       chunk source's lead-chunk contribution is emitted as
       best-effort.
    """
    emission = accumulator.flush()
    if emission is not None:
        _append_emission(items, emission)


def _append_emission(
    items: List[LineItem], emission: AccumulatorEmission,
) -> None:
    """Convert an :class:`AccumulatorEmission` to an :class:`AsmLine`.

    The short text becomes :attr:`AsmLine.text`; a non-``None``
    :attr:`AccumulatorEmission.precision_entry` rides along on the
    line's :attr:`AsmLine.openables` tuple so the tree-model's lazy
    expansion path (R2e) can surface the full-precision form as a
    child row.
    """
    openables: tuple
    if emission.precision_entry is not None:
        openables = (emission.precision_entry,)
    else:
        openables = ()
    items.append(AsmLine(text=emission.short_text, openables=openables))
