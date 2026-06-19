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

import os

import numpy as np

from dedup_hashmap import build_identity_carriers_kernel

from tokenizer.token_manager import VocabularyManager

from ._dense_columns import DenseColumns


__all__ = [
    "IdentitySlicesCSR",
    "build_identity_idx_2d",
    "view_cast_identities",
]


# When set, derive the per-call_target identity CSR via the original
# per-node Python loop (the byte-identity oracle) instead of the
# vectorised cumsum. Equivalence-gate / mutation-test hook only -- the
# production path is the vectorised one.
_USE_PYTHON_IDENTITY_SLICES = bool(
    os.environ.get("ASM_PYTHON_IDENTITY_SLICES")
)


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
    inline_byte_starts: np.ndarray,
) -> tuple[np.ndarray, "IdentitySlicesCSR"]:
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
    inline_byte_starts
        Stage 3a's per-call-target byte start offsets (``int64``), ONE
        entry per level-4 call_target in DFS encounter order. A zero-byte
        call_target is valid (drop / no surviving inline bytes); the
        corresponding identity slice will have length equal to
        ``surviving_identity_count`` (zero when the call_target is fully
        dropped, ``1 + in_stream`` otherwise).

    Returns
    -------
    identity_idx_2d : np.ndarray
        ``u32[total_in_stream_identity_tokens, 2]``. Each row is a pair
        of byte offsets into ``inline_bytes``. Prepend slots are NOT
        included here (stage 4 writes them into
        ``identities_flat_caller_local`` directly per ALG-9).
    identity_slices : IdentitySlicesCSR
        Lazy per-call_target identity CSR (one entry per level-4
        call_target, matching the order of ``inline_byte_starts``).
        Indexing yields a ``slice`` into the level-1
        ``identities_flat_caller_local`` array that INCLUDES the prepend
        slot at ``slice.start``; the range length =
        ``surviving_identity_count`` of the call_target (which equals
        ``1 + in_stream_id_count`` when the call_target survives at all,
        else ``0``). The hot path reads ``.starts`` / ``.stops`` for the
        vectorised level-1 scatter without ever materialising slices.
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
    ) = _gather_identity_carriers(dense, inline_byte_starts)

    identity_idx_2d = _identity_rows_from_carriers(
        carrier_offsets, carrier_L, carrier_raw_positions
    )

    return identity_idx_2d, identity_slices


class IdentitySlicesCSR:
    """Lazy per-call_target identity CSR over the level-1 array.

    Holds the per-node ``starts`` / ``stops`` cumsum arrays (one entry per
    DFS node, INCLUDING the prepend slot at ``start``) and materialises a
    ``slice`` only on element access. The vector hot path reads the CSR
    arrays directly (``.starts`` / ``.stops``) for the vectorised level-1
    scatter; the staged tree-walk indexes per call_target and gets a real
    ``slice`` back -- so the per-node ``slice``-object Python loop only
    runs when the tree is actually assembled, never on the hot path.

    Supports the read-only sequence protocol the consumers need:
    ``len()``, positive/negative ``__getitem__`` (returning ``slice``),
    iteration, and truthiness. It is NOT a ``list`` -- there is no
    materialised slice list anywhere on the hot path.
    """

    __slots__ = ("starts", "stops")

    def __init__(self, starts: np.ndarray, stops: np.ndarray) -> None:
        self.starts = starts
        self.stops = stops

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, i: int) -> slice:
        return slice(int(self.starts[i]), int(self.stops[i]))

    def __iter__(self):
        for start, stop in zip(self.starts.tolist(), self.stops.tolist()):
            yield slice(start, stop)

    @property
    def total_length(self) -> int:
        """Total level-1 length (== ``stops[-1]`` / ``[-1].stop``, 0 empty)."""
        return int(self.stops[-1]) if self.stops.shape[0] else 0


def _identity_slices(dense: DenseColumns) -> IdentitySlicesCSR:
    """Per-call_target identity CSR (DFS order, all targets).

    Each entry covers ``surviving_identity_count`` slots (INCLUDING the
    prepend slot) into the level-1 ``identities_flat_caller_local`` array;
    fully-dropped call_targets get a zero-length range at the running
    offset. This is a plain cumsum over the per-node surviving identity
    counts -- the same offsets the per-call_target walk produced.

    Vectorised: ``stops = cumsum(surviving_identity_count)`` and
    ``starts = stops - surviving_identity_count``; the
    ``surviving_token_count == 0 => surviving_identity_count == 0``
    invariant is a single boolean reduction. No per-node Python loop.
    """
    if _USE_PYTHON_IDENTITY_SLICES:
        return _identity_slices_python_oracle(dense)

    sic = np.ascontiguousarray(dense.surviving_identity_count, dtype=np.int64)
    stc = np.ascontiguousarray(dense.surviving_token_count, dtype=np.int64)
    # Defensive: a fully-dropped call_target (surviving_token_count == 0)
    # also has zero surviving identity tokens -- the prepend at expanded
    # position 0 is itself an identity token.
    if np.any((stc == 0) & (sic != 0)):
        raise AssertionError(
            "Stage 2 invariant violated: a call_target with "
            "surviving_token_count == 0 must also have "
            "surviving_identity_count == 0 (the prepend at "
            "expanded position 0 is itself an identity token)."
        )
    stops = np.cumsum(sic)
    starts = stops - sic
    return IdentitySlicesCSR(starts, stops)


def _identity_slices_python_oracle(dense: DenseColumns) -> IdentitySlicesCSR:
    """Original per-node Python prefix-sum -- byte-identity oracle.

    Pinned verbatim from the pre-port loop: one running offset, one
    ``slice`` per node, the ``surviving_token_count == 0 => sic == 0``
    assert per node. Materialises the CSR arrays from the produced
    slices so it returns the same :class:`IdentitySlicesCSR` type as the
    vectorised path (the equivalence gate compares the produced
    ``starts`` / ``stops``).
    """
    slices: list[slice] = []
    level1_offset = 0
    for e in range(dense.n_nodes):
        sic = int(dense.surviving_identity_count[e])
        if int(dense.surviving_token_count[e]) == 0:
            assert sic == 0, (
                "Stage 2 invariant violated: a call_target with "
                "surviving_token_count == 0 must also have "
                "surviving_identity_count == 0 (the prepend at "
                "expanded position 0 is itself an identity token)."
            )
        slices.append(slice(level1_offset, level1_offset + sic))
        level1_offset += sic
    starts = np.fromiter(
        (sl.start for sl in slices), dtype=np.int64, count=len(slices)
    )
    stops = np.fromiter(
        (sl.stop for sl in slices), dtype=np.int64, count=len(slices)
    )
    return IdentitySlicesCSR(starts, stops)


def _gather_identity_carriers(
    dense: DenseColumns,
    inline_byte_starts: np.ndarray,
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
    it is already one vectorised pass over these flat triples. The
    per-call-target byte start offsets arrive as the stage-3a CSR
    ``inline_byte_starts`` array directly (no slice-list re-extraction).
    """
    inline_slice_start = np.ascontiguousarray(
        inline_byte_starts, dtype=np.int64
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


def scatter_in_stream_identities(
    identity_slices: "IdentitySlicesCSR",
    identities_in_stream: np.ndarray,
) -> np.ndarray:
    """Allocate the level-1 identities array + scatter the in-stream ids.

    Builds ``identities_flat_caller_local`` (``u16`` of length
    ``identity_slices.total_length``) with prepend slots left 0 (stage 4
    fills them per ALG-9), and scatters ``identities_in_stream`` (the
    view-cast caller-local u16 ids, in DFS-then-stream order) into the
    POST-PREPEND sub-range ``[start + 1 : stop]`` of every surviving
    call_target.

    Vectorised: the destination indices are
    ``concat over nodes of arange(start + 1, stop)`` -- the classic
    cumulative-offset arange (``np.repeat`` of the per-node first
    post-prepend index + a within-node ``arange`` derived from the
    per-node in-stream lengths), then ONE fancy-index assignment. No
    per-node Python loop.

    The flat emission order of ``identities_in_stream`` is "all in-stream
    identities of call_target 0, then 1, ..." with per-call_target length
    ``surviving_identity_count - 1`` for surviving targets and 0 for
    fully-dropped ones -- exactly the lengths
    ``max(stop - start - 1, 0)`` recover, so the global arange consumes
    ``identities_in_stream`` left-to-right in the same order.
    """
    starts = identity_slices.starts
    stops = identity_slices.stops
    flat = np.zeros(identity_slices.total_length, dtype=np.uint16)

    # Per-node in-stream length: surviving targets have slot 0 = prepend,
    # slots 1..end = in-stream ids; fully-dropped (zero-length) targets
    # contribute nothing. ``max(.., 0)`` keeps a zero-length range at 0.
    in_stream_len = np.maximum(stops - starts - 1, 0)
    total = int(in_stream_len.sum())
    if total != int(identities_in_stream.shape[0]):
        raise AssertionError(
            f"identity in-stream count mismatch: per-call_target lengths "
            f"sum to {total} u16 ids but the view-cast produced "
            f"{int(identities_in_stream.shape[0])}"
        )
    if total == 0:
        return flat

    # First POST-PREPEND destination index per node (start + 1), repeated
    # by the node's in-stream length, plus a within-node 0..len arange.
    first_dest = starts + 1
    base = np.repeat(first_dest, in_stream_len)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(in_stream_len) - in_stream_len, in_stream_len
    )
    flat[base + within] = identities_in_stream
    return flat
