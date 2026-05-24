"""Top-level length-bucketed batch helper (plan D7).

:func:`open_length_bucketed_batch` samples ``batch_size`` section
pointers via a :class:`MultiBinarySortedIndexSampler`, groups them by
binary, opens one :class:`BinarySession` per binary via
``session_factory``, runs :func:`batch_decode` over each group, and
concatenates the per-binary results via :func:`_concat_results`.

Binary ordering: per-binary results are concatenated in alphabetical
``binary_name`` order (the same order
:attr:`MultiBinarySortedIndexSampler.binary_names` exposes). This
determines the ``binary_id_per_row`` numbering and is stable across
runs.

Imports from ``loader/batch_decode`` here are the typed handoff
classes plus the public :func:`batch_decode` entry; we do NOT touch
any internal stage helpers.
"""

from __future__ import annotations

from typing import Callable, ContextManager, Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._entry import batch_decode
from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.session import BinarySession

from .._types import MultiBinaryBatchDecodeResult
from ._concat import _concat_results
from ._sample import MultiBinarySortedIndexSampler


__all__ = [
    "open_length_bucketed_batch",
]


def open_length_bucketed_batch(
    session_factory: Callable[[str], ContextManager[BinarySession]],
    sampler: MultiBinarySortedIndexSampler,
    target_length: int,
    batch_size: int,
    *,
    context_len: int,
    num_variants_per_section: int,
    max_depth: int,
    rng: np.random.Generator,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    inlined_equivalent_call_targets_only: bool = False,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
) -> MultiBinaryBatchDecodeResult:
    """Length-bucketed batch helper (plan D7).

    Samples ``batch_size`` section pointers via ``sampler`` at
    ``target_length``, groups by binary, opens one session per binary
    via ``session_factory``, runs :func:`batch_decode` over each
    group, and concatenates the per-binary results via
    :func:`_concat_results`.

    Binary ordering: per-binary results are concatenated in
    alphabetical ``binary_name`` order (the same order
    :attr:`MultiBinarySortedIndexSampler.binary_names` exposes). This
    determines the ``binary_id_per_row`` numbering and is stable
    across runs.

    Raises
    ------
    ValueError
        When the sampler returns 0 pointers (empty pool at
        ``target_length`` across every binary). The caller is
        expected to handle this -- either skip the training step or
        pick a different ``target_length``.
    ValueError
        When ``keep_intermediate=True``. The cross-binary
        :func:`_concat_results` boundary inherently drops per-binary
        :class:`Stage3Batch` intermediates (each is single-binary-
        scoped and not stitchable), so the helper rejects this flag
        rather than silently producing inconsistent state.
    """
    if keep_intermediate:
        raise ValueError(
            "open_length_bucketed_batch: keep_intermediate=True is not "
            "supported -- per-binary Stage3Batch intermediates cannot "
            "cross the concat boundary",
        )

    pointers = sampler.sample_section_pointers(
        target_length, batch_size, rng,
    )
    if not pointers:
        raise ValueError(
            "open_length_bucketed_batch: empty sampler pool at "
            f"target_length={target_length}",
        )

    # Group section pointers by binary_name. Iterate the sampled list
    # rather than the sampler's full binary set so binaries with zero
    # samples are skipped (no empty BinarySession opens).
    per_binary_pointers: Dict[str, List[SectionPointerSpec]] = {}
    for ptr in pointers:
        per_binary_pointers.setdefault(ptr.binary_name, []).append(
            ptr.section_pointer,
        )

    # Iterate per-binary work in alphabetical order so the concat
    # input list is canonical and the resulting binary_id_per_row
    # numbering is stable.
    per_binary_results: List[Tuple[str, BatchDecodeResult]] = []
    for binary_name in sampler.binary_names:
        section_pointers = per_binary_pointers.get(binary_name)
        if section_pointers is None:
            continue
        with session_factory(binary_name) as session:
            result = batch_decode(
                session,
                section_pointers,
                num_variants_per_section=num_variants_per_section,
                context_len=context_len,
                max_depth=max_depth,
                variant_padding=variant_padding,
                inlined_equivalent_call_targets_only=(
                    inlined_equivalent_call_targets_only
                ),
                include_fid_sidecar=include_fid_sidecar,
                keep_intermediate=False,
                rng=rng,
            )
        per_binary_results.append((binary_name, result))

    return _concat_results(per_binary_results)
