"""Per-row block / call-target boundary computation.

Single concern: given a row's ``partial_cut_lengths`` + runlength
sidecars, derive the column positions that drive the walker's
section transitions:

* :func:`call_target_starts` -- per-call-target start cols (one per
  CT), used so the FUNCTION-band resolver advances
  ``current_call_target_idx`` into the right
  ``call_targets_section`` table and so the section accumulator can
  open the FUNCTION_ID section at the root CT's leading slot.
* :func:`header_trigger_cols` -- ``pending_header``-latch trigger
  cols (CT boundaries + in-CT block boundaries). The latch suppresses
  the ``Block_Def`` INSTR_REP and consumes the ``block_v2`` IDENTITY
  silently so the BODY section's first item is the first real
  instruction.

The helpers are pure-input -> pure-output (no dependencies on the
walker's mutable state); separated from :mod:`._row_walk` so the
boundary algorithm can be unit-tested + reused without standing up
the full walker. Owned by the BatchDecode backend (the FTL backend
uses :func:`FunctionTokenList.iter_blocks` directly).
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "call_target_starts",
    "header_trigger_cols",
]


def call_target_starts(
    *, n_axis: int, partial_cut_lengths: list[int]
) -> list[int]:
    """Per-call-target start cols, in encounter order.

    Row layout per :mod:`tokenizer.aligned_data.loader.batch_decode._token_assembly`:
    ``row[0:n_axis] = variant_tokens_prefix``; then each call_target's
    ``expanded_token_ids[:partial_cut_length]`` lands consecutively.
    The first start is ``n_axis`` (after variant prefix); subsequent
    starts increment by each prior CT's ``partial_cut_length``.
    """
    starts: list[int] = []
    running = n_axis
    for pcl in partial_cut_lengths:
        starts.append(running)
        running += pcl
    return starts


def header_trigger_cols(
    *,
    n_axis: int,
    partial_cut_lengths: list[int],
    block_runlength_row: np.ndarray,
    insn_runlength_row: np.ndarray,
) -> frozenset[int]:
    """Compute the row-column positions that latch ``pending_header``.

    Two trigger sources:

    * CT-boundary columns (one per non-empty call_target). The first
      BLOCK_V2 token at-or-after such a column is the call_target's
      entry header. In production the self-prepend slot at
      ``ct_start_i`` is a LOCAL_FUNC / PLT_FUNC IDENTITY token that
      latches through to col ``ct_start_i + 1`` (the actual BLOCK_V2
      header).
    * Runlength-computed in-CT block-start columns. For each
      call_target body, cumulative ``insn_runlength`` sums map block
      index ``k+1`` to the column where the block's BLOCK_V2 header
      lands; that column is added as a trigger so multi-block bodies
      open new blocks at the right boundaries.

    Empty CTs (``partial_cut_lengths[i] == 0``) contribute nothing.
    ``block_runlength_row`` / ``insn_runlength_row`` are sliced per
    :class:`BatchDecodeResult`'s ``block_runlength_row_offsets`` /
    ``insn_runlength_row_offsets`` so they cover exactly this row's
    surviving content (per stage 2's cutoff walk).
    """
    triggers: set[int] = set()
    block_cursor = 0
    insn_cursor = 0
    col = n_axis
    n_blocks = int(block_runlength_row.size)
    n_insns = int(insn_runlength_row.size)
    for pcl in partial_cut_lengths:
        if pcl <= 0:
            continue
        # CT-boundary trigger: latches through the self-prepend slot to
        # the first BLOCK_V2 header.
        triggers.add(col)
        # Self-prepend slot at col; body starts at col+1.
        body_start_col = col + 1
        block_col = body_start_col
        ct_body_slot_budget = int(pcl) - 1
        first_in_ct = True
        while block_cursor < n_blocks and ct_body_slot_budget > 0:
            block_insn_count = int(block_runlength_row[block_cursor])
            if block_insn_count <= 0:
                # Defensive: skip empty blocks (the runlength emit
                # should not produce them, but iterating doesn't
                # break either).
                block_cursor += 1
                continue
            if not first_in_ct:
                # In-CT block boundary: latches the next BLOCK_V2
                # header (which lands AT block_col by the runlength
                # contract).
                triggers.add(block_col)
            first_in_ct = False
            block_slot_sum = 0
            for _ in range(block_insn_count):
                if insn_cursor >= n_insns:
                    break
                block_slot_sum += int(insn_runlength_row[insn_cursor])
                insn_cursor += 1
            block_col += block_slot_sum
            ct_body_slot_budget -= block_slot_sum
            block_cursor += 1
        # Advance the running col past this CT.
        col += int(pcl)
    return frozenset(triggers)
