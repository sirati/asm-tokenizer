"""Shared helper for extracting per-call-target in-stream identity ids.

Single concern: parallel-to-identity-slice extraction of the surviving
in-stream identity-band token ids for one call_target. Used by BOTH
:mod:`._function_remap` (ALG-3) and :mod:`._counter_bump` (ALG-4) to
build the Category-mask for the in-stream slice.

Lives in its own submodule because it's the only shared algorithmic
helper between the two per-Category dispatch paths — the constants in
:mod:`._constants` are pure-data; this helper reads from a
:class:`Stage3CallTarget`'s stage-2 state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from .._types import Stage3CallTarget


__all__ = ["_surviving_in_stream_token_ids"]


def _surviving_in_stream_token_ids(
    call_target: "Stage3CallTarget",
) -> np.ndarray:
    """Token ids parallel to the call_target's in-stream identity slots.

    The call_target's ``identity_slice`` includes the prepend slot at
    its start (ALG-9); the in-stream slots are ``identity_slice``
    minus that first slot. The parallel token ids come from the
    SURVIVING in-stream IDENTITY-band tokens in
    ``expanded_token_ids[:partial_cut_length]`` — the prepend's row in
    that mask is dropped here.

    Returns a u16 ndarray of length
    ``identity_slice.stop - identity_slice.start - 1``.
    """
    stage2 = call_target.stage2
    surviving_expanded = stage2.expanded_token_ids[: stage2.partial_cut_length]
    # IDENTITY block shifted span: [8, 16) per the vocab table.
    identity_band_mask = (surviving_expanded >= np.uint16(8)) & (
        surviving_expanded < np.uint16(16)
    )
    identity_token_ids = surviving_expanded[identity_band_mask]
    # The first entry in ``identity_token_ids`` corresponds to the
    # prepend slot (slot 0 of ``expanded_token_ids`` is the calling-
    # category self-token, an IDENTITY-band id). Drop it so the
    # remaining ids parallel the call_target's in-stream identity
    # slice.
    return identity_token_ids[1:]
