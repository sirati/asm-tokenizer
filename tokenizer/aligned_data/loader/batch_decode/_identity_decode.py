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

from typing import TYPE_CHECKING

import numpy as np

from tokenizer.token_manager import VocabularyManager

if TYPE_CHECKING:
    from ._types import Stage2Batch


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
    stage2_batch: "Stage2Batch",
    inline_bytes: np.ndarray,
    inline_byte_slices: list[slice],
) -> tuple[np.ndarray, list[slice]]:
    """Build the identity-token idx_2d table per ALG-5.

    Walks sections -> variants -> call_targets in DFS encounter order
    (the linearisation stage 1 produced + stage 2 mirrors). For each
    SURVIVING identity carrier in each call_target's raw token stream
    emits one row into ``identity_idx_2d`` that points (in
    ``inline_bytes`` coordinates) at the carrier's payload bytes.

    Excludes prepend slots: stage 4 writes them directly per ALG-9.

    Parameters
    ----------
    stage2_batch
        The level-1 stage-2 result. The walk only reads
        ``stage1.state.raw_tokens`` / ``runlen_number`` /
        ``number_mask`` / ``carries_inline_mask`` (via the level-4
        ``stage1.state`` back-pointer) and the per-call-target
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
    # Phase 1 -- single linear walk to collect per-carrier byte offsets.
    #
    # We size the output exactly via the per-variant aggregate counts
    # already computed at stage 2:
    #   total_in_stream = total_surviving_identity - per-call-target_count_with_at_least_one_surviving_token
    # ...but that's a slightly awkward derivation. The simpler -- and
    # equally vectorised -- approach is to collect each call_target's
    # per-carrier u32[K, 2] block into a Python list of arrays and
    # concatenate at the end. The Python-list overhead is per-call-target
    # not per-token, so it stays out of the hot path.
    # ------------------------------------------------------------------

    rows_chunks: list[np.ndarray] = []
    identity_slices: list[slice] = []
    level1_offset: int = 0

    call_target_iter_idx = 0

    for stage2_section in stage2_batch.sections:
        for stage2_variant in stage2_section.variants:
            for stage2_ct in stage2_variant.call_targets:
                inline_byte_slice = inline_byte_slices[call_target_iter_idx]
                call_target_iter_idx += 1

                surviving_token_count = stage2_ct.surviving_token_count
                surviving_identity_count = stage2_ct.surviving_identity_count

                # A call_target that fully dropped under the cut
                # contributes nothing to either the level-1 identity
                # array OR the idx_2d table. The slice is empty at the
                # current level1_offset (length 0, NO advance).
                if surviving_token_count == 0:
                    # Defensive: a fully-dropped call_target also has
                    # zero surviving identity tokens (the prepend at
                    # expanded position 0 is itself an identity token,
                    # so it can't survive if surviving_token_count == 0).
                    assert surviving_identity_count == 0, (
                        "Stage 2 invariant violated: a call_target with "
                        "surviving_token_count == 0 must also have "
                        "surviving_identity_count == 0 (the prepend at "
                        "expanded position 0 is itself an identity "
                        "token)."
                    )
                    identity_slices.append(
                        slice(level1_offset, level1_offset)
                    )
                    continue

                # Surviving call_target -- at least the prepend slot is
                # in [8, 16). in_stream_id_count is the count of
                # identity carriers AFTER the prepend that landed in
                # the surviving expanded prefix.
                in_stream_id_count = surviving_identity_count - 1

                # Reserve the full slice (prepend + in-stream entries).
                identity_slices.append(
                    slice(level1_offset, level1_offset + surviving_identity_count)
                )
                level1_offset += surviving_identity_count

                if in_stream_id_count == 0:
                    # The call_target survives but no in-stream
                    # identity carrier did (only the prepend at
                    # expanded[0] is an identity, stage 4 writes that).
                    # Nothing to add to identity_idx_2d.
                    continue

                # Locate the surviving in-stream identity carriers in
                # raw_tokens. Identity tokens are NEVER promoted (only
                # VC2 + F128 promote per stage 2a), so their raw-stream
                # positions are in 1:1 encounter-order correspondence
                # with the in-stream identity positions in the expanded
                # stream. The cut may chop off LATER identity carriers
                # but never reorders -- so the surviving carriers are
                # exactly the FIRST in_stream_id_count identity-band
                # entries of state.carries_inline_mask.
                state = stage2_ct.stage1.state
                raw_tokens = state.raw_tokens

                identity_carrier_mask = state.real_mask & (
                    (raw_tokens >= _V2_IDENTITY_BLOCK_START)
                    & (raw_tokens < _V2_EAGER_BLOCK_END)
                )
                identity_carrier_positions = np.nonzero(
                    identity_carrier_mask
                )[0]
                surviving_carrier_positions = identity_carrier_positions[
                    :in_stream_id_count
                ]

                # ----------------------------------------------------------
                # Byte-offset computation -- vectorised over the K
                # surviving identity carriers of this call_target.
                #
                # For each carrier at raw position p:
                #   * Payload length L = runlen_number[p+1] when
                #     p+1 < n (and 0 otherwise -- a carrier at the
                #     last raw position has no payload). The runlen
                #     array is 0 at non-run-start slots, so
                #     ``runlen_number[p+1] == 0`` whenever
                #     ``number_mask[p+1] == False`` -- the conditional
                #     reduces to "is p+1 in-bounds?".
                #   * First payload-byte's offset in the call_target's
                #     surviving-inline-byte region equals the count of
                #     number_mask=True positions in raw_tokens[:p+1]
                #     (= raw_tokens[:p] since number_mask[p] is False
                #     at identity carriers; using [:p+1] is also
                #     correct and the cumsum interface is cleaner).
                #     Add inline_byte_slice.start for the absolute
                #     offset in inline_bytes.
                #
                # Per-call-target cumsum is O(n) for the function; we
                # do it once and then index it by the K carrier
                # positions.
                # ----------------------------------------------------------
                runlen_number = state.runlen_number
                number_mask = state.number_mask
                n = raw_tokens.shape[0]

                # cum_number[i] = sum(number_mask[0:i+1]) = count of
                # number_mask=True in raw_tokens[0..i] inclusive. We
                # need count strictly before p+1 for an identity
                # carrier at p; since number_mask[p] is False, that
                # count equals cum_number[p].
                cum_number = number_mask.cumsum(dtype=np.int64)

                # Payload length per carrier. p+1 is in-bounds when
                # p < n - 1; the carrier at position n-1 (if any) has
                # no payload (L=0). We compute L for ALL carriers via
                # a conditional fetch -- using np.where to avoid an
                # out-of-bounds gather.
                p = surviving_carrier_positions.astype(np.int64)
                has_p1 = p < (n - 1)
                # For carriers with p == n-1 (no p+1 slot) we read
                # runlen_number[0] as a safe dummy index; the np.where
                # below overrides those entries to 0.
                safe_p1 = np.where(has_p1, p + 1, np.int64(0))
                L_raw = runlen_number[safe_p1].astype(np.int64)
                L = np.where(has_p1, L_raw, np.int64(0))

                # First payload byte offset within the call_target's
                # inline-byte region (0-based) = cum_number[p].
                # Absolute offset in inline_bytes adds inline_byte_slice.start.
                first_payload_offset = (
                    cum_number[p] + np.int64(inline_byte_slice.start)
                )

                # idx_2d rows per ALG-5 payload-width table.
                # We produce a fresh u32[K, 2] array per call_target;
                # the final concatenate at the bottom does the only
                # large copy.
                rows = np.zeros(
                    (surviving_carrier_positions.shape[0], 2),
                    dtype=np.uint32,
                )
                two_byte_mask = L == 2
                one_byte_mask = L == 1
                # 0-byte rows stay [0, 0] from the zero-allocation.

                # 2-byte: [hi_offset, lo_offset] with lo = hi + 1.
                if two_byte_mask.any():
                    hi_2 = first_payload_offset[two_byte_mask].astype(
                        np.uint32
                    )
                    rows[two_byte_mask, 0] = hi_2
                    rows[two_byte_mask, 1] = hi_2 + np.uint32(1)

                # 1-byte: [0, lo_offset]. hi stays at 0 from
                # zero-allocation; only write lo.
                if one_byte_mask.any():
                    rows[one_byte_mask, 1] = first_payload_offset[
                        one_byte_mask
                    ].astype(np.uint32)

                # Defensive: any other payload width is a v2-codec
                # violation for identity tokens (per the v2 spec
                # identity payloads are 0 / 1 / 2 bytes only).
                other_width_mask = ~(two_byte_mask | one_byte_mask | (L == 0))
                if other_width_mask.any():
                    bad_positions = surviving_carrier_positions[
                        other_width_mask
                    ]
                    bad_lengths = L[other_width_mask]
                    raise AssertionError(
                        f"Identity carriers at raw positions "
                        f"{bad_positions.tolist()} declared payload "
                        f"lengths {bad_lengths.tolist()} -- v2 spec "
                        "restricts identity payloads to {0, 1, 2} bytes."
                    )

                rows_chunks.append(rows)

    # ------------------------------------------------------------------
    # Phase 2 -- single concatenate. Producing a (0, 2) shape when no
    # in-stream identity tokens exist anywhere keeps the dtype contract
    # uniform regardless of the batch's identity-token density.
    # ------------------------------------------------------------------
    if rows_chunks:
        identity_idx_2d = np.concatenate(rows_chunks, axis=0)
    else:
        identity_idx_2d = np.zeros((0, 2), dtype=np.uint32)

    return identity_idx_2d, identity_slices


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
