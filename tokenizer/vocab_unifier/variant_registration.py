"""Variant-axis sidecar discovery helper for the unifier.

Single concern: walk a corpus of per-binary vocab CSVs and yield one
``VariantInfo`` per CSV, skipping (with a warning) any CSV whose
filename + optional ``_meta.json`` sidecar fails to parse.

Registration of variant tokens onto the unified vocabulary lives in
``tokenizer.vocab_unifier.unifier`` because the variant block lands at
the TAIL of the unified vocab (after all instruction representatives)
and the order is axis-grouped via
``VariantInventory.iter_tokens_axis_grouped`` — both invariants the
unifier owns, not this helper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from tokenizer.variant_info import VariantInfo

logger = logging.getLogger(__name__)


def _iter_variant_infos(csv_files: Iterable[Path]) -> Iterable[VariantInfo]:
    """Yield one ``VariantInfo`` per CSV; skip + warn on parse errors.

    A single corrupt sidecar must not abort the discovery walk — the
    unifier already tolerates per-file load failures in its instruction
    merge and the same robustness applies here. The caller decides what
    to do with a zero-variants outcome.
    """
    for csv_path in csv_files:
        try:
            yield VariantInfo.from_csv(csv_path)
        except (ValueError, OSError) as exc:
            logger.warning(
                "variant discovery: skipping %s (%s)", csv_path, exc,
            )
