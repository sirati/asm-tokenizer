"""Variant-axis token discovery + registration for the unifier.

Single concern: walk a corpus of per-binary vocab CSVs, derive every
distinct variant-axis token string (via the existing
``VariantInfo.from_csv`` + ``VariantInventory`` chain), and register
each one into a unified ``VocabularyManager`` in a deterministic
order.

This module sits between ``tokenizer.variant_tokens`` (which knows
the prefix grammar and the inventory) and
``tokenizer.vocab_unifier.unifier`` (which orchestrates the two
passes). It does NOT touch CSV vocab contents, mapping arrays, or
instruction tokens — those are pass 2's concern.

Order discipline: ``VariantInventory.iter_tokens`` yields strings in
alphabetical order. The unifier registers them on a freshly
constructed v3 ``VocabularyManager`` whose first registerable ID is
``_V2_RESERVED_DIGIT_COUNT`` (256). Variant tokens therefore land at
contiguous IDs ``[256, 256 + n_variants)``. Two runs over the same
corpus produce byte-identical unified vocabs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from tokenizer.token_manager import VocabularyManager
from tokenizer.variant_info import VariantInfo
from tokenizer.variant_tokens import VariantInventory

logger = logging.getLogger(__name__)


def _iter_variant_infos(csv_files: Iterable[Path]) -> Iterable[VariantInfo]:
    """Yield one ``VariantInfo`` per CSV; skip + warn on parse errors.

    A single corrupt sidecar must not abort the discovery pass — the
    unifier already tolerates per-file load failures in pass 2 and the
    same robustness applies here. The caller decides what to do with
    a zero-variants outcome.
    """
    for csv_path in csv_files:
        try:
            yield VariantInfo.from_csv(csv_path)
        except (ValueError, OSError) as exc:
            logger.warning(
                "variant discovery: skipping %s (%s)", csv_path, exc,
            )


def discover_and_register_variants(
    csv_files: list[Path],
    unified_vm: VocabularyManager,
) -> int:
    """Pass-1 variant discovery + registration.

    Sidecar-only walk: each CSV is mapped to a ``VariantInfo`` via
    ``VariantInfo.from_csv`` (which reads only the filename + the
    optional ``_meta.json`` sidecar — never the CSV body). Every
    distinct prefixed axis string the corpus implies is registered
    into ``unified_vm`` via ``unified_vm.Variant_Axis(token_str)``.

    The Inner-class constructor calls ``_private_add_token`` under
    the hood and is idempotent on duplicate strings, so the per-token
    registration cost is one dict lookup after the first occurrence.

    ``unified_vm.format_version`` MUST be 3 — variant tokens are
    meaningful only in the additive v3 unified vocab. The assert
    is here (not inside the Inner class) because the Inner class is
    layout-agnostic and a future format may legitimately reuse it.

    Returns the count of distinct variant tokens registered — the
    unifier logs this and tests assert it.
    """
    assert unified_vm.format_version == 3, (
        f"discover_and_register_variants requires a v3 unified VM; "
        f"got format_version={unified_vm.format_version}"
    )

    inventory = VariantInventory()
    inventory.update(_iter_variant_infos(csv_files))

    # `iter_tokens` yields alphabetically — deterministic across runs
    # on the same corpus. Each call to `unified_vm.Variant_Axis(s)`
    # registers `s` at the next free vocab id (starting at 256 on a
    # fresh v3 VM); duplicates are no-ops.
    n_registered = 0
    for token_str in inventory.iter_tokens():
        unified_vm.Variant_Axis(token_str)
        n_registered += 1

    logger.info(
        "variant discovery: registered %d distinct variant-axis tokens "
        "(from %d CSV files)",
        n_registered,
        len(csv_files),
    )
    return n_registered
