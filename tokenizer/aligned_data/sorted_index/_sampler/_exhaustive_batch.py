"""Exhaustive dataloader entry: enumerate -> fixed-group RAGGED decode.

Single concern of this module: own the SESSION LIFECYCLE around the
deterministic :class:`ExhaustiveSectionSampler`. It supplies the sampler's
injected ``count_provider`` (a body-free per-binary variant-count read),
groups the exhaustive all-variants pointer sequence into fixed-size
pointer groups, and decodes each group through the EXISTING
:func:`decode_pointer_batch` core with ``variant_padding=RAGGED`` -- the
all-variants subset rides inside each pointer's
:class:`ExplicitIndicesSelection`, so decode is reused verbatim.

This is the modular SIBLING to :func:`open_validation_batches`: same
session-lifecycle concern, same verbatim decode reuse, but it groups WHOLE
pointers (sections) into fixed bunches rather than chunking one section's
shuffled variants. RAGGED gives contiguous per-section variant blocks with
no padding, so ``batch_idx_to_section_variant`` maps each row back to its
(section-in-group, variant) directly.

Concern boundaries:

* The sampler (:mod:`._exhaustive`) owns the deterministic enumeration +
  pointer construction; it is decode/session-agnostic.
* This entry owns ONLY session lifecycle: a per-binary session cache
  (opened lazily, closed together on exit via a single
  :class:`contextlib.ExitStack`) backs both the count_provider reads and
  the per-group decode. Each ``decode_pointer_batch`` result reads
  session-backed numpy views, so the owning sessions stay open across the
  whole iteration.
* Decode is the unchanged :func:`decode_pointer_batch`; this module adds no
  decode logic.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Callable, ContextManager, Dict, Iterator, List, Sequence

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import VariantPadding
from tokenizer.aligned_data.loader.session import BinarySession

from .._types import MultiBinaryBatchDecodeResult, MultiBinarySectionPointer
from ._batch import decode_pointer_batch
from ._exhaustive import ExhaustiveSectionSampler


__all__ = ["open_exhaustive_batches"]


def open_exhaustive_batches(
    session_factory: Callable[[str], ContextManager[BinarySession]],
    sampler: ExhaustiveSectionSampler,
    *,
    group_size: int,
    context_len: int,
    max_depth: int,
    rng: np.random.Generator,
    include_fid_sidecar: bool = False,
    inlined_equivalent_call_targets_only: bool = True,
) -> Iterator[MultiBinaryBatchDecodeResult]:
    """Decode the exhaustive pointer sequence in fixed-size RAGGED groups.

    Builds the deterministic whole-corpus pointer sequence
    (:meth:`ExhaustiveSectionSampler.all_pointers`), then groups it into
    consecutive bunches of at most ``group_size`` pointers (the last group
    may be smaller -- NO drop, every section is decoded). Opens one
    :class:`BinarySession` per binary lazily (cached + closed together via
    an :class:`ExitStack` spanning the whole iteration), threaded into BOTH
    the sampler's variant-count provider AND the per-group decode.

    Each group decodes through the unchanged :func:`decode_pointer_batch`
    with ``variant_padding=RAGGED`` (contiguous per-section variant blocks,
    no padding). ``rng`` is threaded into the decode core for its own draws
    (callee-walk sampling etc.); the VARIANT selection itself is fully
    deterministic via each pointer's :class:`ExplicitIndicesSelection` and
    never consumes ``rng``.

    Yields one :class:`MultiBinaryBatchDecodeResult` per group, in the
    sampler's deterministic canonical order. The generator keeps every
    session open until exhausted (or the caller closes it), since each
    yielded result reads session-backed numpy views.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")

    with ExitStack() as session_stack:
        session_cache: Dict[str, BinarySession] = {}

        def get_session(binary_name: str) -> BinarySession:
            session = session_cache.get(binary_name)
            if session is None:
                session = session_stack.enter_context(
                    session_factory(binary_name)
                )
                session_cache[binary_name] = session
            return session

        def count_provider(
            binary_name: str, section_indices: np.ndarray
        ) -> np.ndarray:
            return get_session(
                binary_name
            )._matched_section_variant_counts(section_indices)

        pointers = sampler.all_pointers(count_provider)
        for group in _group(pointers, group_size):
            sessions = {
                ptr.binary_name: get_session(ptr.binary_name)
                for ptr in group
            }
            yield decode_pointer_batch(
                sessions,
                list(group),
                context_len=context_len,
                num_variants_per_section=group_size,
                max_depth=max_depth,
                rng=rng,
                variant_padding=VariantPadding.RAGGED,
                inlined_equivalent_call_targets_only=(
                    inlined_equivalent_call_targets_only
                ),
                include_fid_sidecar=include_fid_sidecar,
            )


def _group(
    pointers: Sequence[MultiBinarySectionPointer], group_size: int
) -> Iterator[List[MultiBinarySectionPointer]]:
    """Consecutive at-most-``group_size`` slices of ``pointers`` (no drop)."""
    for start in range(0, len(pointers), group_size):
        yield list(pointers[start : start + group_size])
