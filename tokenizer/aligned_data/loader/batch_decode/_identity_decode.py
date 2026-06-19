"""Stage 3b -- identity idx_2d construction + view-cast to caller-local u16.

Single concern: given Stage 2's per-call-target expanded streams and
Stage 3a's already-allocated ``inline_bytes`` buffer (one u8 array with a
leading zero pad at index 0 + per-call-target byte slices computed by 3a),
build the per-position 2D byte-offset table that, when used as a fancy
index into ``inline_bytes``, yields the big-endian u16 caller-local
identity ids in stream order. The view-cast helper does the final
``.view('>u2').reshape(-1)`` step that turns the gathered u8[N, 2] pairs
into the model-facing u16[N] caller-local id sequence.

What this module owns -- and ONLY this:

* The mapping from "surviving in-stream identity carrier in raw_tokens"
  to "(hi_offset, lo_offset) pair into inline_bytes" (per ALG-5).
* The view-cast step (a single ``np.ndarray.view`` + ``reshape``).
* The per-call-target ``identity_slice`` that records WHERE this
  call_target's identities live in the level-1 ``identities_flat_caller_local``
  array (INCLUDING the leading prepend slot stage 4 writes per ALG-9).

What this module does NOT own:

* The ``inline_bytes`` buffer allocation -- stage 3a does that (ALG-1).
* The prepend slot's caller-local id -- stage 4 writes it (ALG-9).
* Per-Category remap of caller-local ids -- stage 4 owns ALG-3 + ALG-4.
* Mapping from "in-stream identity position in expanded_token_ids" to
  "raw_tokens position" -- inferred from
  :class:`InlineDecodeState`'s ``carries_inline_mask`` + ``raw_tokens``
  directly (identity carriers are never promoted; per-stream encounter
  order matches expanded-stream order, modulo the cut prefix).

Plan reference: ``batch_decode_plan.md`` ALG-5 + Stage 3 step 3.

ALG-5 payload-width recap (from the plan):

* 2-byte identity payload: ``idx_2d row = [hi_offset, lo_offset]`` with
  ``lo_offset = hi_offset + 1`` (consecutive bytes in ``inline_bytes``).
* 1-byte identity payload: ``idx_2d row = [0, lo_offset]`` -- ``0`` is
  the leading zero pad at ``inline_bytes[0]`` which supplies the high
  byte of the big-endian u16.
* 0-byte identity payload: ``idx_2d row = [0, 0]`` -- read as u16 ``0``
  via the leading zero pad twice. The v2 encoder reserves caller-local
  id ``0`` for this case so no real callee collides.

In all three cases the row is read as a 2-byte big-endian u16, which is
exactly what a ``.view('>u2')`` of the gathered u8[N, 2] block performs
in a single vectorised pass.
"""

from __future__ import annotations

import numpy as np

from dedup_hashmap import build_identity_carriers_kernel

from tokenizer.token_manager import VocabularyManager

from ._dense_columns import DenseColumns


__all__ = ["build_identity_idx_2d", "view_cast_identities"]


# ---------------------------------------------------------------------------
# Band constants -- derived from the VocabularyManager source of truth.
#
# Identity carriers in raw_tokens (pre-shift) sit in [264, 272). In the
# expanded stream (post-shift, post-strip) the same tokens sit in [8, 16).
# We use the RAW-stream band here because the per-carrier byte-offset
# computation walks ``state.raw_tokens`` + ``state.number_mask``, not the
# expanded stream.
# ---------------------------------------------------------------------------
_V2_IDENTITY_BLOCK_START = VocabularyManager._V2_IDENTITY_BLOCK_START  # 264
_V2_EAGER_BLOCK_END = VocabularyManager._V2_EAGER_BLOCK_END  # 272


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_identity_idx_2d(
    dense: DenseColumns,
    inline_bytes: np.ndarray,
    inline_byte_slices: list[slice],
) -> tuple[np.ndarray, list[slice]]:
    """Build the identity-token idx_2d table per ALG-5.

    Reads the shared :class:`DenseColumns` front-matter over the DFS
    (== emitted-node) node axis. For each SURVIVING identity carrier in
    each call_target's raw token stream emits one row into
    ``identity_idx_2d`` that points (in ``inline_bytes`` coordinates) at
    the carrier's payload bytes.

    Excludes prepend slots: stage 4 writes them directly per ALG-9.

    Parameters
    ----------
    dense
        The shared dense front-matter (:class:`DenseColumns`). The walk
        reads each node's ``raw_tokens`` / ``runlen_number`` /
        ``real_mask`` / ``digit_cumsum`` and its per-node
        ``surviving_token_count`` / ``surviving_identity_count``.
    inline_bytes
        The u8 array stage 3a produced (size ``1 + total_surviving_bytes``;
        index 0 is the leading zero pad). Only its shape is consulted
        (the byte values are gathered later via fancy-indexing during
        the view-cast step).
    inline_byte_slices
        Stage 3a's per-call-target byte slices, ONE entry per level-4
        call_target in DFS encounter order. Slice length 0 is valid
        (drop / no surviving inline bytes); the corresponding identity
        slice will have length equal to ``surviving_identity_count``
        (zero when the call_target is fully dropped, ``1 + in_stream``
        otherwise).

    Returns
    -------
    identity_idx_2d : np.ndarray
        ``u32[total_in_stream_identity_tokens, 2]``. Each row is a pair
        of byte offsets into ``inline_bytes``. Prepend slots are NOT
        included here (stage 4 writes them into
        ``identities_flat_caller_local`` directly per ALG-9).
    identity_slices : list[slice]
        One entry per level-4 call_target (matching the order of
        ``inline_byte_slices``). Each slice points into the level-1
        ``identities_flat_caller_local`` array and INCLUDES the
        prepend slot at ``slice.start``. Slice length =
        ``surviving_identity_count`` of the call_target (which equals
        ``1 + in_stream_id_count`` when the call_target survives at
        all, else ``0``).
    """

    # ------------------------------------------------------------------
    # B-S2 batched form: instead of one byte-offset compute pass per
    # call_target, EVERY surviving in-stream identity carrier of EVERY
    # call_target is gathered into flat arrays and the ALG-5 row build
    # runs in a SINGLE vectorised pass. The per-call_target
    # ``identity_slices`` (a pure cumsum over ``surviving_identity_count``)
    # and the per-carrier ``inline_byte_slice.start`` base are derived as
    # CSR arrays. Output ``identity_idx_2d`` rows stay in DFS-then-stream
    # encounter order -- byte-identical to the per-call_target walk.
    # ------------------------------------------------------------------
    identity_slices = _identity_slices(dense)

    (
        carrier_offsets,
        carrier_L,
        carrier_raw_positions,
    ) = _gather_identity_carriers(dense, inline_byte_slices)

    identity_idx_2d = _identity_rows_from_carriers(
        carrier_offsets, carrier_L, carrier_raw_positions
    )

    return identity_idx_2d, identity_slices


def _identity_slices(dense: DenseColumns) -> list[slice]:
    """Per-call_target ``identity_slice`` list (DFS order, all targets).

    Each slice covers ``surviving_identity_count`` entries (INCLUDING the
    prepend slot) into the level-1 ``identities_flat_caller_local`` array;
    fully-dropped call_targets get a zero-length slice at the running
    offset. This is a plain cumsum over the per-node surviving identity
    counts -- the same offsets the per-call_target walk produced, one
    ``slice`` object per node.
    """
    slices: list[slice] = []
    level1_offset = 0
    for e in range(dense.n_nodes):
        sic = int(dense.surviving_identity_count[e])
        if int(dense.surviving_token_count[e]) == 0:
            # Defensive: a fully-dropped call_target also has zero
            # surviving identity tokens (the prepend at expanded
            # position 0 is itself an identity token).
            assert sic == 0, (
                "Stage 2 invariant violated: a call_target with "
                "surviving_token_count == 0 must also have "
                "surviving_identity_count == 0 (the prepend at "
                "expanded position 0 is itself an identity token)."
            )
        slices.append(slice(level1_offset, level1_offset + sic))
        level1_offset += sic
    return slices


def _gather_identity_carriers(
    dense: DenseColumns,
    inline_byte_slices: list[slice],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat per-carrier ``(first_payload_offset, L, raw_position)``.

    For each SURVIVING in-stream identity carrier (the first
    ``surviving_identity_count - 1`` identity-band real tokens of each
    surviving node's raw stream), the carrier's absolute first-payload
    byte offset into ``inline_bytes`` + its payload length ``L`` (per the
    ALG-5 width table) + its raw position, in DFS-then-stream order.

    GIL-released (B-S3): the per-node Python gather loop (per-node
    ``np.nonzero`` of the identity-band real mask + the ``runlen_number``
    / ``digit_cumsum`` gathers) is a single ``py.detach`` Rust kernel
    reading the flat :class:`DenseColumns` columns directly. The ALG-5
    row build (:func:`_identity_rows_from_carriers`) stays in numpy --
    it is already one vectorised pass over these flat triples.
    """
    inline_slice_start = np.fromiter(
        (sl.start for sl in inline_byte_slices),
        dtype=np.int64,
        count=len(inline_byte_slices),
    )
    return build_identity_carriers_kernel(
        np.ascontiguousarray(dense.raw_tokens, dtype=np.uint16),
        np.ascontiguousarray(dense.real_mask, dtype=np.bool_),
        np.ascontiguousarray(dense.runlen_number, dtype=np.uint16),
        np.ascontiguousarray(dense.raw_offsets, dtype=np.int64),
        np.ascontiguousarray(dense.digit_cumsum, dtype=np.uint32),
        np.ascontiguousarray(dense.digit_offsets, dtype=np.int64),
        np.ascontiguousarray(dense.surviving_token_count, dtype=np.int64),
        np.ascontiguousarray(dense.surviving_identity_count, dtype=np.int64),
        inline_slice_start,
        int(_V2_IDENTITY_BLOCK_START),
        int(_V2_EAGER_BLOCK_END),
    )


def _identity_rows_from_carriers(
    first_payload_offset: np.ndarray,
    L: np.ndarray,
    raw_positions: np.ndarray,
) -> np.ndarray:
    """ALG-5 ``u32[K, 2]`` rows for the flat carrier population.

    A 2-byte carrier emits ``[hi, hi + 1]``; a 1-byte carrier ``[0, lo]``
    (the leading zero pad supplies the high byte); a 0-byte carrier stays
    ``[0, 0]``. Any other width is a v2-codec violation. One vectorised
    pass over all carriers -- no per-call_target iteration.
    """
    rows = np.zeros((int(first_payload_offset.shape[0]), 2), dtype=np.uint32)
    two_byte_mask = L == 2
    one_byte_mask = L == 1

    if two_byte_mask.any():
        hi_2 = first_payload_offset[two_byte_mask].astype(np.uint32)
        rows[two_byte_mask, 0] = hi_2
        rows[two_byte_mask, 1] = hi_2 + np.uint32(1)

    if one_byte_mask.any():
        rows[one_byte_mask, 1] = first_payload_offset[one_byte_mask].astype(
            np.uint32
        )

    other_width_mask = ~(two_byte_mask | one_byte_mask | (L == 0))
    if other_width_mask.any():
        bad_positions = raw_positions[other_width_mask]
        bad_lengths = L[other_width_mask]
        raise AssertionError(
            f"Identity carriers at raw positions "
            f"{bad_positions.tolist()} declared payload "
            f"lengths {bad_lengths.tolist()} -- v2 spec "
            "restricts identity payloads to {0, 1, 2} bytes."
        )

    return rows


def view_cast_identities(
    identity_idx_2d: np.ndarray,
    inline_bytes: np.ndarray,
) -> np.ndarray:
    """Gather + view-cast per ALG-5.

    Two-stage vectorised decode:

    1. ``gathered = inline_bytes[identity_idx_2d]`` -- u8[N, 2]; a fresh
       contiguous copy (numpy fancy-indexing always copies). Each row
       holds the big-endian byte pair for the carrier's caller-local
       u16 id, with the leading zero pad at ``inline_bytes[0]``
       supplying the high byte for 1-byte payloads and both bytes for
       0-byte payloads.
    2. ``gathered.view('>u2').reshape(-1)`` -- a zero-copy reinterpret
       of the same buffer as big-endian u16 followed by a flatten. The
       reshape collapses the trailing length-1 axis (each pair becomes
       one u16) so the output is the requested ``u16[N]``.

    Returns
    -------
    np.ndarray
        ``u16[N]`` of caller-local identity ids in stream encounter
        order, EXCLUDING prepend slots. The native byte order is
        whatever ``inline_bytes`` was stored in -- after the view-cast
        from ``>u2`` numpy presents the values as little-endian-
        compatible Python integers on read, the underlying buffer
        remains big-endian; consumers that compare against ``np.uint16``
        scalars see the correct numeric value.

    Notes
    -----
    Prepend slots are not in ``identity_idx_2d`` -- stage 4 writes them
    directly into the level-1 ``identities_flat_caller_local`` array at
    each call_target's ``identity_slice.start`` index per ALG-9.
    """
    # The empty path matters for two reasons: (1) numpy fancy-indexing
    # with an empty index array works but produces an empty result of
    # the SAME dtype as the indexed array (u8 here), (2) the view-cast
    # itself is well-defined on a (0, 2) u8 array -- it produces a
    # (0, 1) u16 array which reshape(-1) collapses to (0,). So no
    # special case is required; the path is exercised by the
    # zero-identity-tokens test.
    gathered = inline_bytes[identity_idx_2d]
    return gathered.view(">u2").reshape(-1)
