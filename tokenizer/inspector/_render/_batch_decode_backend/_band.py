"""Band classification for the BatchDecodeBackend row walker.

Single concern: classify a (shifted) vocab id into one of the four
disjoint bands the per-token dispatch reads off the wire format. The
band layout itself is the unified vocab's canonical partition (see
:class:`~tokenizer.token_manager.VocabularyManager` docstring lines
116-122 + the ``_V2_*`` slot anchors):

* shifted id ``0`` -- padding / end-of-stream.
* shifted id ``[_NUMBER_BAND_LO_SHIFTED, _NUMBER_BAND_HI_SHIFTED)`` --
  NUMBER block (VC2 / F16 / BF16 / F32 / F64 / F80 / F128).
* shifted id ``[_IDENTITY_BAND_LO_SHIFTED, _IDENTITY_BAND_HI_SHIFTED)``
  -- IDENTITY block (8 :class:`Category` slots).
* shifted id ``>= _IDENTITY_BAND_HI_SHIFTED`` -- instruction-rep block.

The thresholds are imported (NUMBER bounds) or derived (IDENTITY
bounds) from :class:`VocabularyManager` slot anchors; this module
never inlines the literal values ``1`` / ``8`` / ``16`` so a vocab-
layout shift surfaces in one place (token_manager.py) and ripples
through unchanged consumers.

Lives under :mod:`._batch_decode_backend` because the row walker is
the sole consumer; the prior cross-backend "shared dispatch table"
scaffolding was never populated and got dropped in Phase D cleanup.
"""

from __future__ import annotations

from enum import IntEnum

from tokenizer.aligned_data.loader.batch_decode._number_decode._band_constants import (
    _NUMBER_BAND_HI_SHIFTED,
    _NUMBER_BAND_LO_SHIFTED,
)
from tokenizer.token_manager import VocabularyManager


__all__ = [
    "Band",
    "classify_shifted_id",
]


# ---------------------------------------------------------------------------
# Band boundaries
#
# NUMBER bounds come from ``_band_constants.py`` (the existing single-source
# for the NUMBER-block layout). IDENTITY upper bound derives from the
# ``_V2_EAGER_BLOCK_END`` slot anchor on :class:`VocabularyManager` -- the
# first instruction-rep slot. No literal ``1`` / ``8`` / ``16`` lives in
# this module; per audit B-MED-10 / B-LOW-14 a vocab-layout shift updates
# the central anchors and every consumer (this module included) sees the
# new bounds on next import.
# ---------------------------------------------------------------------------
_IDENTITY_BAND_LO_SHIFTED: int = _NUMBER_BAND_HI_SHIFTED  # NUMBER ends, IDENTITY begins
_IDENTITY_BAND_HI_SHIFTED: int = (
    VocabularyManager._V2_EAGER_BLOCK_END - VocabularyManager._V2_RESERVED_DIGIT_COUNT
)


class Band(IntEnum):
    """The four disjoint shifted-id bands the per-token dispatch reads.

    Integer values are arbitrary; bands are discriminated by enum
    identity. ``PADDING`` is the row-end / pad-slot sentinel (shifted
    id ``0``); ``INSTR_REP`` covers every shifted id at or above
    :data:`_IDENTITY_BAND_HI_SHIFTED` (the instruction-rep block grows
    beyond ``_V2_EAGER_BLOCK_END`` on the unified vocab as more
    instructions are seen).
    """

    PADDING = 0
    NUMBER = 1
    IDENTITY = 2
    INSTR_REP = 3


def classify_shifted_id(t: int) -> Band:
    """Map a shifted id to its :class:`Band`.

    The branch order matches expected per-row frequency on the wire
    (INSTR_REP > IDENTITY > NUMBER > PADDING under typical assembly
    streams) so the hot path bails on the first compare. Branch-free
    arithmetic forms exist but would obscure the boundary semantics;
    this is a per-token classifier on the inspector path, not the
    Stage 1 cutter, so correctness clarity wins over micro-throughput.
    """
    if t >= _IDENTITY_BAND_HI_SHIFTED:
        return Band.INSTR_REP
    if t >= _IDENTITY_BAND_LO_SHIFTED:
        return Band.IDENTITY
    if t >= _NUMBER_BAND_LO_SHIFTED:
        return Band.NUMBER
    return Band.PADDING
