"""Skip predicates shared by pass-1 walkers.

Row decoding + hash precomputation now live on the per-CSV iterator
(:mod:`tokenizer.aligned_data.parsed_record_iter`); the binary-record
writer is in :mod:`tokenizer.aligned_data._writers`; the global dedup
helper is in :mod:`tokenizer.memmap_builder._dedup`. This module is
intentionally small — it just owns the two predicates that the matched
and unmatched walkers gate function-survival on.
"""

from __future__ import annotations

import numpy as np

# Block-runlength cap (in sum units, NOT block count). Functions whose
# summed block runlengths reach this number are dropped from the
# matched arm — they would otherwise push the encoder past the
# ``block_word_count`` cap on at least one variant.
_BLOCK_RUNLENGTH_SUM_CAP: int = 4096


def should_skip_for_matched(block_runlength: np.ndarray) -> bool:
    """``True`` when this variant's block runlength reaches the cap.

    Pass-1 matched calls this for every per-variant block runlength
    and skips the function (whole, across variants) if any one of
    them reaches the cap. Mirrors the pre-refactor
    ``should_skip_function_for_matched`` semantics one variant at a
    time so the caller can short-circuit on the first hit.
    """
    return int(block_runlength.sum()) >= _BLOCK_RUNLENGTH_SUM_CAP


def should_skip_for_unmatched(_block_runlength: np.ndarray) -> bool:
    """No-op preserved from the legacy walker.

    The pre-refactor ``should_skip_function_for_unmatched`` looked up
    ``row["block_runlength"]`` instead of ``row["block_runlength_base64"]``
    so the check never fired in practice. The behavior is preserved here
    verbatim to keep the dedup refactor scope-clean; the buggy intent of
    skipping small unmatched functions is a separate question worth
    raising with the user before changing.
    """
    return False
