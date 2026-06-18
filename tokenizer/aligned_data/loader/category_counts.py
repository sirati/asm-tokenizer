"""Per-function COUNTER-Category unique-id counts.

Single concern: given a function's raw token stream (the same shape
:class:`FunctionData.tokens` carries -- v2 wire form, carriers + inline
digits interleaved), return the per-COUNTER-Category count of distinct
caller-local ids the encoder emitted.

The result is consumed by Stage-4 of the batch-decode pipeline (plan
ALG-4 ``_per_call_target_counter_count``) as the per-function "how many
ids of this Category do I contribute to the row-global offset". Because
caller-local ids are emitted DENSE 0..K-1 by the encoder, K = (max id
seen) + 1; this module materializes K via a distinct-count which is
equivalent under the dense invariant and stays correct even if the
invariant is ever relaxed.

The FUNCTION categories (LOCAL_FUNC / PLT_FUNC / EXT_FUNC) deliberately
DO NOT appear in the returned mapping -- they dedup by
``function_name_ptr`` (ALG-3), not by per-row offset, so no per-function
counter count is meaningful for them.

Plan reference: ``batch_decode_plan.md`` -- ALG-4 + ALG-5 + the loader-
side ``FunctionData.metadata["category_counts"]`` contract.
"""

from __future__ import annotations

import numpy as np

from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import Category

from .decoded._inline_decode_state import build_inline_decode_state


__all__ = [
    "COUNTER_CATEGORIES",
    "compute_category_counts",
    "category_counts_from_runlen",
    "category_counts_from_runlen_batched",
]


# ---------------------------------------------------------------------------
# Category -> raw (pre-shift) identity vocab id under the unified vocab.
#
# Source of truth: :class:`VocabularyManager`'s canonical IDENTITY-block
# layout (slots 264..271 in user-canonical-then-alphabetical order;
# see the table in :class:`VocabularyManager`'s class docstring). The
# offsets mirror ``_dedup_walk._CATEGORY_TO_SHIFTED_ID`` but stay in the
# pre-shift (raw_tokens) space because the COUNTER walk reads from the
# raw stream, not the strip-and-shift expanded stream.
# ---------------------------------------------------------------------------
_V2_IDENTITY_BLOCK_START = VocabularyManager._V2_IDENTITY_BLOCK_START


# IDENTITY-block layout offsets (per VocabularyManager's canonical table).
# Only the COUNTER categories appear here -- FUNCTION categories
# (LOCAL_FUNC / PLT_FUNC / EXT_FUNC at offsets 1/2/3) are intentionally
# omitted because they dedup by function_name_ptr, not by counter count.
_COUNTER_CATEGORY_TO_RAW_ID: dict[Category, int] = {
    Category.BLOCK: _V2_IDENTITY_BLOCK_START + 0,
    Category.STRING_PTR: _V2_IDENTITY_BLOCK_START + 4,
    Category.JUMP_TABLE: _V2_IDENTITY_BLOCK_START + 5,
    Category.RO_DATA_PTR: _V2_IDENTITY_BLOCK_START + 6,
    Category.RW_DATA_PTR: _V2_IDENTITY_BLOCK_START + 7,
}


# Public tuple of the categories this module produces counts for. Mirrors
# ``_dedup_walk.COUNTER_CATEGORIES`` -- the two tables share the same
# partition of :class:`Category` but live in different modules to keep
# each module's import surface tight.
COUNTER_CATEGORIES: tuple[Category, ...] = tuple(_COUNTER_CATEGORY_TO_RAW_ID)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def compute_category_counts(raw_tokens: np.ndarray) -> dict[Category, int]:
    """Count distinct caller-local ids per COUNTER Category in ``raw_tokens``.

    Builds an :class:`InlineDecodeState` once and walks each COUNTER
    Category's identity carriers, decoding the inline-byte payload per
    ALG-5 (0-byte -> id 0; 1-byte -> id = byte; 2-byte -> id = u16
    big-endian). Returns the count of distinct ids per Category.

    The returned dict ALWAYS contains an entry for every COUNTER
    Category (value 0 when the Category has no carriers in the stream)
    so consumers can ``[]`` index without a key-presence dance. The
    contract is "fully populated dense mapping; missing keys are an
    upstream bug, not a benign absence".

    Parameters
    ----------
    raw_tokens
        ``uint16[N]`` v2 wire-form token stream (carriers + inline-digit
        bytes interleaved; see plan ``### Vocab + wire format reference``).
        May be zero-length: an empty stream contributes zero counts in
        every Category, returned as the all-zeros mapping.

    Returns
    -------
    dict[Category, int]
        One entry per :data:`COUNTER_CATEGORIES`. Values are
        non-negative ints (zero when the Category has no carriers).
    """
    # Zero-length stream: every Category gets 0. Building an
    # InlineDecodeState on an empty array is well-defined (all masks are
    # empty), but short-circuiting avoids the work and keeps the
    # contract explicit.
    if raw_tokens.shape[0] == 0:
        return {cat: 0 for cat in COUNTER_CATEGORIES}

    state = build_inline_decode_state(raw_tokens, format_version=1)
    return category_counts_from_runlen(raw_tokens, state.runlen_number)


def category_counts_from_runlen(
    raw_tokens: np.ndarray, runlen_number: np.ndarray
) -> dict[Category, int]:
    """Count distinct caller-local ids reusing a precomputed ``runlen_number``.

    Identical contract + output to :func:`compute_category_counts`, but
    the caller supplies the ``number_mask`` run-length array
    (``InlineDecodeState.runlen_number``) it already computed for the
    same ``raw_tokens`` stream, so this path skips the full
    :class:`InlineDecodeState` rebuild. The distinct-id counting logic is
    shared with :func:`compute_category_counts` (which delegates here),
    so the two cannot drift.

    Parameters
    ----------
    raw_tokens
        ``uint16[N]`` v2 wire-form token stream.
    runlen_number
        ``InlineDecodeState.runlen_number`` for the SAME ``raw_tokens``:
        per-position run length of the inline-digit (``< 256``) mask.
        Must be the aligned, same-length array for ``raw_tokens``.
    """
    if raw_tokens.shape[0] == 0:
        return {cat: 0 for cat in COUNTER_CATEGORIES}

    n = int(raw_tokens.shape[0])
    counts: dict[Category, int] = {}
    for category, carrier_id in _COUNTER_CATEGORY_TO_RAW_ID.items():
        counts[category] = _count_distinct_caller_local_ids(
            raw_tokens, runlen_number, carrier_id, n
        )
    return counts


def category_counts_from_runlen_batched(
    raw_flat: np.ndarray,
    runlen_number_flat: np.ndarray,
    record_offsets: np.ndarray,
) -> dict[Category, np.ndarray]:
    """Per-node distinct caller-local id counts across a whole batch.

    The batched twin of :func:`category_counts_from_runlen`: instead of
    one scalar :func:`_count_distinct_caller_local_ids` call per node per
    COUNTER Category, this counts every node's distinct ids in ONE
    numpy reduction per Category over the flat batch arrays. The distinct
    COUNT is an order-independent per-node reduction, so the per-node
    result is byte-identical to looping :func:`category_counts_from_runlen`
    over each node's ``raw_flat[lo:hi]`` / ``runlen_number_flat[lo:hi]``
    slice.

    Parameters
    ----------
    raw_flat
        ``uint16[total_raw]`` -- every node's v2 wire-form token stream
        concatenated in node order (the body gather's flat ``raw``).
    runlen_number_flat
        ``InlineDecodeState.runlen_number`` for the SAME flat stream:
        per-position run length of the inline-digit (``< 256``) mask,
        aligned to ``raw_flat``.
    record_offsets
        ``int64[n_nodes + 1]`` CSR jump table: node ``i`` owns
        ``raw_flat[record_offsets[i] : record_offsets[i + 1]]``. The
        per-node ``p + 1`` / ``p + 2`` payload reads are clamped to the
        owning node's end so a carrier at a node boundary never reads
        into the next node (exactly the per-node ``p < n - 1`` guard).

    Returns
    -------
    dict[Category, np.ndarray]
        One ``int64[n_nodes]`` entry per :data:`COUNTER_CATEGORIES`;
        ``out[category][i]`` equals
        ``category_counts_from_runlen(node_i_raw, node_i_runlen)[category]``.
    """
    raw_flat = np.asarray(raw_flat, dtype=np.uint16).reshape(-1)
    runlen_number_flat = np.asarray(runlen_number_flat).reshape(-1)
    rec = np.asarray(record_offsets, dtype=np.int64).reshape(-1)
    n_nodes = int(rec.size) - 1
    return {
        category: _count_distinct_caller_local_ids_per_node(
            raw_flat, runlen_number_flat, rec, n_nodes, carrier_id
        )
        for category, carrier_id in _COUNTER_CATEGORY_TO_RAW_ID.items()
    }


def _count_distinct_caller_local_ids_per_node(
    raw_flat: np.ndarray,
    runlen_number_flat: np.ndarray,
    record_offsets: np.ndarray,
    n_nodes: int,
    carrier_id: int,
) -> np.ndarray:
    """``int64[n_nodes]`` distinct-id count for ONE COUNTER Category.

    Vectorized twin of :func:`_count_distinct_caller_local_ids` evaluated
    over every node at once. Carrier positions are located in the flat
    stream, attributed to their owning node via the CSR ``record_offsets``,
    decoded per the ALG-5 payload-width table (0/1/2 bytes), and the
    distinct ``(node, id)`` pairs are reduced to a per-node count by a
    single ``unique`` + ``bincount``.
    """
    out = np.zeros(n_nodes, dtype=np.int64)
    carrier_pos = np.flatnonzero(raw_flat == np.uint16(carrier_id))
    if carrier_pos.size == 0:
        return out

    # Owning node per carrier (CSR bucket) + that node's exclusive end.
    node = np.searchsorted(record_offsets, carrier_pos, side="right") - 1
    node_end = record_offsets[node + 1]

    # Payload length per carrier. A carrier whose ``p + 1`` slot falls in
    # the NEXT node (or past the stream) has no in-node payload slot --
    # treat its length as 0 (the 0-byte branch decodes id 0 reading no
    # byte). This is the flat-space form of the per-node ``p < n - 1``
    # guard. A safe in-bounds dummy index avoids an out-of-bounds gather.
    has_p1 = (carrier_pos + 1) < node_end
    safe_p1 = np.where(has_p1, carrier_pos + 1, np.int64(0))
    L = np.where(
        has_p1, runlen_number_flat[safe_p1].astype(np.int64), np.int64(0)
    )

    # Decode caller-local ids per ALG-5 payload-width table. 2-byte
    # carriers' ``p + 2`` slot is guaranteed in-node (the 2-long inline
    # run that produced ``L == 2`` lives within the node), so the gather
    # stays in-bounds.
    ids = np.zeros(carrier_pos.shape, dtype=np.uint16)

    one_byte_mask = L == 1
    if one_byte_mask.any():
        ids[one_byte_mask] = raw_flat[
            carrier_pos[one_byte_mask] + np.int64(1)
        ].astype(np.uint16)

    two_byte_mask = L == 2
    if two_byte_mask.any():
        hi = raw_flat[carrier_pos[two_byte_mask] + np.int64(1)].astype(
            np.uint16
        )
        lo = raw_flat[carrier_pos[two_byte_mask] + np.int64(2)].astype(
            np.uint16
        )
        ids[two_byte_mask] = (hi << np.uint16(8)) | lo

    # Defensive: any width other than 0/1/2 is a v2-codec violation for
    # identity carriers (ALG-5 restricts identity payloads to those
    # widths). Surface it here, mirroring the per-node scalar path.
    other_width_mask = ~(one_byte_mask | two_byte_mask | (L == 0))
    if other_width_mask.any():
        bad_positions = carrier_pos[other_width_mask]
        bad_lengths = L[other_width_mask]
        raise AssertionError(
            f"Identity carrier id {carrier_id} at raw positions "
            f"{bad_positions.tolist()} declared payload lengths "
            f"{bad_lengths.tolist()} -- v2 spec restricts identity "
            "payloads to {0, 1, 2} bytes."
        )

    # Distinct count per node = number of unique ``(node, id)`` pairs that
    # land in each node. Dedup the pairs, then bincount the surviving node
    # labels. ``unique`` over a single composite key is order-independent,
    # so the per-node count matches the scalar ``np.unique(ids).size``.
    key = node.astype(np.int64) * np.int64(1 << 16) + ids.astype(np.int64)
    distinct_keys = np.unique(key)
    distinct_node = distinct_keys >> np.int64(16)
    counts = np.bincount(distinct_node, minlength=n_nodes)
    out[: counts.size] = counts.astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Internal helper -- per-Category distinct-id count.
# ---------------------------------------------------------------------------


def _count_distinct_caller_local_ids(
    raw_tokens: np.ndarray,
    runlen_number: np.ndarray,
    carrier_id: int,
    n: int,
) -> int:
    """Count distinct caller-local ids for one COUNTER Category.

    Carrier positions are ``raw_tokens == carrier_id``. For each carrier
    at position ``p`` the payload length is ``runlen_number[p + 1]``
    (0 / 1 / 2 bytes -- ALG-5 restricts identity payloads to that set).
    The caller-local id decodes to:

    * 0-byte: id = 0 (the encoder reserves caller-local id 0 for this
      case so no real callee collides -- plan ``_identity_decode`` 0-byte
      table row).
    * 1-byte: id = ``raw_tokens[p + 1]`` (the low byte; high byte 0).
    * 2-byte: id = ``(raw_tokens[p + 1] << 8) | raw_tokens[p + 2]``
      (big-endian u16).

    Returns the number of distinct decoded ids across all carriers of
    this ``carrier_id``. Returns 0 when no carriers of this Category
    exist in the stream.
    """
    carrier_positions = np.flatnonzero(raw_tokens == np.uint16(carrier_id))
    if carrier_positions.size == 0:
        return 0

    # Payload length per carrier. A carrier at the last raw position has
    # no p+1 slot -- treat its payload length as 0 (the 0-byte branch
    # produces id 0 without reading any byte). ``np.where`` with a safe
    # in-bounds dummy index avoids an out-of-bounds gather.
    p = carrier_positions.astype(np.int64)
    has_p1 = p < (n - 1)
    safe_p1 = np.where(has_p1, p + 1, np.int64(0))
    L = np.where(has_p1, runlen_number[safe_p1].astype(np.int64), np.int64(0))

    # Decode caller-local ids per ALG-5 payload-width table.
    ids = np.zeros(p.shape, dtype=np.uint16)

    one_byte_mask = L == 1
    if one_byte_mask.any():
        ids[one_byte_mask] = raw_tokens[
            (p[one_byte_mask] + np.int64(1))
        ].astype(np.uint16)

    two_byte_mask = L == 2
    if two_byte_mask.any():
        hi = raw_tokens[(p[two_byte_mask] + np.int64(1))].astype(np.uint16)
        lo = raw_tokens[(p[two_byte_mask] + np.int64(2))].astype(np.uint16)
        ids[two_byte_mask] = (hi << np.uint16(8)) | lo

    # 0-byte rows stay at 0 from the zero-allocation -- the encoder's
    # reserved-id-0 contract makes that the correct caller-local id for
    # the 0-byte payload case.

    # Defensive: any other payload width is a v2-codec violation for
    # identity carriers (ALG-5 restricts identity payloads to 0/1/2
    # bytes). Surfacing here keeps the diagnostic local to the loader
    # rather than letting a malformed stream silently produce wrong
    # counts.
    other_width_mask = ~(one_byte_mask | two_byte_mask | (L == 0))
    if other_width_mask.any():
        bad_positions = carrier_positions[other_width_mask]
        bad_lengths = L[other_width_mask]
        raise AssertionError(
            f"Identity carrier id {carrier_id} at raw positions "
            f"{bad_positions.tolist()} declared payload lengths "
            f"{bad_lengths.tolist()} -- v2 spec restricts identity "
            "payloads to {0, 1, 2} bytes."
        )

    return int(np.unique(ids).size)
