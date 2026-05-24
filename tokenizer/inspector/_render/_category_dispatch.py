"""IDENTITY-band per-:class:`Category` emitter registry.

Single concern: publish the shared dispatch table both rendering
backends register into for IDENTITY-band tokens. Once the
:mod:`._band` dispatch routes a shifted id into :class:`Band.IDENTITY`,
the band handler reads the resolved :class:`Category` (via
``_SHIFTED_ID_TO_CATEGORY`` -- the inverse map pinned alongside the
forward in :mod:`tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants`)
and forwards to :data:`CATEGORY_EMITTERS[cat]`.

The eight-Category partition splits into two sub-groups, sourced from
the existing tuples in ``_dedup_walk/_constants.py`` (audit B-LOW-12 —
no inline enum names here):

* :data:`FUNCTION_CATEGORIES` = (LOCAL_FUNC, PLT_FUNC, EXT_FUNC):
  emit one :class:`InlineCallEntry` per token (callee resolution
  via per-backend FID tables).
* :data:`COUNTER_CATEGORIES` = (BLOCK, STRING_PTR, RO_DATA_PTR,
  RW_DATA_PTR, JUMP_TABLE): emit an :class:`AsmLine` placeholder
  (or, for BLOCK, dispatch via the stage-1 boundary table to
  block-header vs :class:`InlineJumpEntry` per plan section 6 +
  decisions 29 + 30).

Both backends (FtlBackend, BatchDecodeBackend) register their own
per-Category emitter at module load. The walker raises
:class:`KeyError` on a missing entry — same fail-loud contract as the
:data:`tokenizer.inspector._render._band.BAND_HANDLERS` registry.

Plan reference: ``inspector-render-backends.md`` §6 + decision 27 +
audits B-LOW-12, B-MED-10.
"""

from __future__ import annotations

from typing import Any, MutableMapping, Protocol

# FUNCTION_CATEGORIES + COUNTER_CATEGORIES are the v2 partition's
# single source of truth (plan D4). Importing them here keeps the
# Category-group lists pinned in ONE module; a partition change
# (e.g. promoting a new identity Category) ripples in without
# re-stamping the group enums per consumer.
from tokenizer.aligned_data.loader.batch_decode._dedup_walk._constants import (
    COUNTER_CATEGORIES,
    FUNCTION_CATEGORIES,
)
from tokenizer.tokens import Category


__all__ = [
    "CATEGORY_EMITTERS",
    "COUNTER_CATEGORIES",
    "CategoryEmitter",
    "FUNCTION_CATEGORIES",
]


# ---------------------------------------------------------------------------
# Per-Category emitter Protocol
# ---------------------------------------------------------------------------


class CategoryEmitter(Protocol):
    """Per-Category IDENTITY-band emitter.

    The IDENTITY-band band-handler (registered in
    :data:`tokenizer.inspector._render._band.BAND_HANDLERS`) resolves
    the shifted id to a :class:`Category`, looks up the emitter via
    :data:`CATEGORY_EMITTERS`, and invokes it with the walker state
    plus per-backend keyword extras.

    State mutation contract: emitters append :class:`LineItem`-typed
    entries onto the walker's current-block accumulator and advance
    any per-Category cursors (FID counters, block-id offsets, ...).
    Return value is :data:`None`; the walker reads the appended items
    from ``state`` after the call.

    FUNCTION-Category emitters typically emit one
    :class:`InlineCallEntry`. COUNTER-Category emitters emit either
    an :class:`AsmLine` placeholder (STRING_PTR / RO_DATA_PTR /
    RW_DATA_PTR / JUMP_TABLE) or, for BLOCK, an
    :class:`InlineJumpEntry` plus optional block-boundary side-effect.
    The dispatch table never knows or branches on which form the
    emitter chose -- that's the per-Category emitter's concern.
    """

    def __call__(self, state: Any, /, **kwargs: Any) -> None:
        ...


# ---------------------------------------------------------------------------
# CATEGORY_EMITTERS — mutable registry, populated by the rendering backends
# ---------------------------------------------------------------------------
#
# Mutable so the two backend submodules can register their per-Category
# emitters at import time. The IDENTITY-band handler raises
# :class:`KeyError` on a missing entry — a registration gap surfaces
# immediately rather than silently rendering an empty line.
CATEGORY_EMITTERS: MutableMapping[Category, CategoryEmitter] = {}
