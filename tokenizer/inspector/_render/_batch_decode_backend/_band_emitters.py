"""Per-band slot emitters for the BatchDecodeBackend row walker.

Single concern: shifted-id -> per-instruction-buffer routing for the
INSTR_REP and NUMBER bands. The IDENTITY band lives in
:mod:`._row_walk._dispatch` (it owns the per-Category dispatch + block
/ function-category branching that need walker state). Each emitter
mutates the in-flight instruction's text-parts / openables buffers on
:class:`._row_walk._state._WalkState` via the per-instruction collector
helpers (:mod:`._row_walk._instruction`), keeping the single-emit-site
discipline (W3-16 W4-AMENDED + A-L1 H2): one :class:`AsmLine` per
:func:`._row_walk._instruction._finalize_instruction`, not one per
slot.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._number_decode._band_constants import (
    _NUMBER_BAND_LO_SHIFTED,
    _NUMBER_BLOCK_TOKEN_TYPES,
)
from tokenizer.aligned_data.loader.decoded._number_render_collector import (
    AccumulatorEmission,
    _NumberAccumulator,
)
from tokenizer.inspector._render._token_text import substitute_display_chars
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

from ._arch_prefix import strip_arch_prefix
from ._row_walk._instruction import _consume_text_slot
from ._row_walk._state import _WalkState


__all__ = [
    "emit_instr_rep",
    "emit_number",
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
    state: _WalkState,
    *,
    shifted_id: int,
    vocab_manager: VocabularyManager,
    arch_prefixes: tuple[str, ...] = (),
) -> None:
    """INSTR_REP band: vocab lookup -> polished display -> text-buffer.

    Two display transforms after the raw vocab-string lookup:

    1. ``substitute_display_chars`` collapses the six
       :class:`MemoryOperandSymbol` vocab strings (``MEM_OPEN_BRACKET``
       etc.) to their display chars (``[`` etc.) -- matches the FTL
       backend's display per cluster #3 of the W3-3 W4-amended plan.
    2. ``strip_arch_prefix`` removes the most-specific matching arch
       prefix (per-ISA, family, then unified) so BatchDecode display
       mirrors :meth:`PlatformTokenInner.to_asm_like`'s prefix-stripped
       form.

    The rendered atom lands on the in-flight instruction's
    :attr:`_WalkState.current_insn_text_parts` via
    :func:`._row_walk._instruction._consume_text_slot`; the eventual
    :class:`AsmLine` emission happens at instruction finalize.

    ``arch_prefixes`` defaults to ``()``: callers that haven't plumbed
    the arch through (test fixtures, legacy paths) get the substitution
    transform without arch elision.
    """
    original_id = int(shifted_id) + _V2_RESERVED_DIGIT_COUNT
    raw = vocab_manager.get_token_str(original_id)
    display = strip_arch_prefix(substitute_display_chars(raw), arch_prefixes)
    _consume_text_slot(state, text=display)


def emit_number(
    state: _WalkState,
    *,
    shifted_id: int,
    numbers_sig: np.ndarray,
    numbers_se: np.ndarray,
) -> None:
    """NUMBER band: feed one chunk pair into the accumulator.

    Drives :attr:`_WalkState.number_accumulator` (the SSOT for
    multi-chunk grouping + short / full text rendering). Multi-chunk-
    capable shifted ids (VC2, F128; :data:`MULTI_CHUNK_SHIFTED_IDS`)
    extend the accumulator across consecutive feeds with the same
    shifted id; a shifted-id change auto-flushes the prior source and
    its emission flows into the in-flight instruction's text buffer
    + openables list.

    Single-chunk shifted ids (F16/BF16/F32/F64/F80) are force-flushed
    immediately after the feed so two consecutive same-shifted-id
    feeds do NOT collapse into a mis-grouped multi-chunk source. The
    force-flush is the correctness gate for cluster #21 / R1-Audit
    L-4: the accumulator's API groups by shifted id alone (it cannot
    distinguish "next single-chunk source" from "trailing chunk"); the
    caller injects that knowledge via the force-flush after every
    non-multi-chunk feed.

    Advances :attr:`_WalkState.num_cursor`. The caller is responsible
    for the instruction-boundary flush (W3-17, done by
    :func:`._row_walk._instruction._finalize_instruction`) and the
    end-of-row flush (cluster #21 H-4 cut-variant tolerance, also done
    by ``_finalize_instruction(end_of_row=True)``).
    """
    band_index = int(shifted_id) - _NUMBER_BAND_LO_SHIFTED
    token_type: TokenType = _NUMBER_BLOCK_TOKEN_TYPES[band_index]
    chunk = (numbers_sig[state.num_cursor], numbers_se[state.num_cursor])
    prior = state.number_accumulator.feed(
        token_type=token_type, shifted_id=int(shifted_id), chunk=chunk,
    )
    if prior is not None:
        _absorb_emission(state, prior)
    if int(shifted_id) not in MULTI_CHUNK_SHIFTED_IDS:
        # Single-chunk source: force-flush so a subsequent same-id
        # feed starts a fresh source instead of extending this one.
        forced = state.number_accumulator.flush()
        if forced is not None:
            _absorb_emission(state, forced)
    state.num_cursor += 1


def _absorb_emission(
    state: _WalkState, emission: AccumulatorEmission,
) -> None:
    """Route an :class:`AccumulatorEmission` into the in-flight
    instruction's text + openables buffers.

    Short text becomes a text atom; a non-``None`` precision entry
    rides on the openables list so the tree-model's lazy expansion
    surfaces the full-precision form as a child row.
    """
    state.current_insn_text_parts.append(emission.short_text)
    if emission.precision_entry is not None:
        state.current_insn_openables.append(emission.precision_entry)
