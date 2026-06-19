"""Validation dataloader entry: drive the sequential sampler -> decode.

Single concern of this module: own the SESSION LIFECYCLE around the
deterministic :class:`SequentialValidationSampler`. It supplies the
sampler's injected ``count_provider`` (a body-free per-binary
variant-count read) and, for each emitted :class:`ValidationBatch`, opens
the binary's session and decodes the bunch through the EXISTING
:func:`decode_pointer_batch` core -- the explicit variant subset rides
inside the pointer's :class:`ExplicitIndicesSelection`, so decode is reused
verbatim.

Concern boundaries:

* The sampler (:mod:`._validation`) owns the deterministic shuffle/chunk
  stream + pointer construction; it is decode/session-agnostic.
* This entry owns ONLY session lifecycle: a per-binary session cache
  (opened lazily, closed together on exit) backs both the count_provider
  reads and the per-bunch decode. The cache is a single
  :class:`contextlib.ExitStack`-scoped concern -- the result of each
  ``decode_pointer_batch`` reads session-backed numpy views, so the owning
  session must stay open across that decode (and a cached session is reused
  across every bunch of that binary).
* Decode is the unchanged :func:`decode_pointer_batch`; this module adds no
  decode logic.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Callable, ContextManager, Dict, Iterator

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import VariantPadding
from tokenizer.aligned_data.loader.session import BinarySession

from .._types import MultiBinaryBatchDecodeResult
from ._batch import decode_pointer_batch
from ._validation import SequentialValidationSampler


__all__ = ["open_validation_batches"]


def open_validation_batches(
    session_factory: Callable[[str], ContextManager[BinarySession]],
    sampler: SequentialValidationSampler,
    *,
    context_len: int,
    max_depth: int,
    rng: np.random.Generator,
    include_fid_sidecar: bool = False,
    inlined_equivalent_call_targets_only: bool = True,
) -> Iterator[MultiBinaryBatchDecodeResult]:
    """Decode the sampler's deterministic bunch stream, one result per bunch.

    Opens one :class:`BinarySession` per binary lazily (cached + closed
    together via an :class:`ExitStack` that spans the whole iteration), and
    threads that cache into BOTH the sampler's variant-count provider AND
    the per-bunch decode. For each :class:`ValidationBatch` the bunch's
    pinned variant subset rides inside the pointer's
    :class:`ExplicitIndicesSelection`, so the decode is the unchanged
    :func:`decode_pointer_batch` with ``variant_padding=RAGGED`` (exactly
    ``batch_size`` rows, no padding) and
    ``num_variants_per_section=batch_size`` (inert under an explicit
    selection -- the count path never runs for these pointers).

    ``rng`` is threaded into ``decode_pointer_batch`` for the decode core's
    own draws (callee-walk sampling etc.); the VARIANT selection itself is
    fully deterministic via the sampler's kernel stream and never consumes
    ``rng``.

    Yields one :class:`MultiBinaryBatchDecodeResult` per bunch, in the
    sampler's deterministic emission order. The generator keeps every
    session open until exhausted (or the caller closes it), since each
    yielded result reads session-backed numpy views.
    """
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

        for batch in sampler.iter_batches(count_provider):
            session = get_session(batch.binary_name)
            yield decode_pointer_batch(
                {batch.binary_name: session},
                [batch.as_pointer()],
                context_len=context_len,
                num_variants_per_section=sampler.batch_size,
                max_depth=max_depth,
                rng=rng,
                variant_padding=VariantPadding.RAGGED,
                inlined_equivalent_call_targets_only=(
                    inlined_equivalent_call_targets_only
                ),
                include_fid_sidecar=include_fid_sidecar,
            )
