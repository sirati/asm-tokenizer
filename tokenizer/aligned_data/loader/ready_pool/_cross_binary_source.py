"""The CROSS-BINARY decode seam: a ``produce`` over ``load_batch_cross_depth``.

Single concern: build the OPAQUE no-arg ``produce`` callable a
:class:`PoolConfig` registers for the CROSS-binary x cross-depth training
distribution -- one that, per call, runs the full produce PIPELINE on the
worker thread: drive the EXISTING cross-binary primitive
:meth:`IndexedMemmapCollection.load_batch_cross_depth` (the
``CrossSpecSortedIndexSampler`` urn draw + per-binary VECTOR_BATCH decode +
alphabetical concat, all internal) and run the user ``postprocess`` (the
FINAL stage) on its FULL :class:`MultiBinaryBatchDecodeResult` -- the
cross-binary result that carries per-row binary identity
(``binary_id_per_row`` / ``binary_names``) the downstream training REQUIRES.

WHY this WRAPS, never reimplements: the production data mix is a
cross-(binary x spec) draw; reproducing it byte-identically means calling
``load_batch_cross_depth`` on a production-configured collection VERBATIM.
This seam owns only the threading (a thread-local collection + RNG) and the
parameter bundle threaded into that one call. It is the cross-binary twin of
:mod:`._vector_batch_source` (single-binary, ``vector_batch_tokens``); the
two are first-class peers behind the same :class:`CloseableProduce` seam and
neither :class:`ReadyPool` nor :class:`GpuReadyPool` knows either exists.

WHY ``postprocess`` runs HERE (the worker thread, the final produce stage):
same rationale as the single-binary seam -- the decode result is CPU numpy
and Layer 2 uploads torch-tensor leaves, so running the user
numpy->upload-ready transform as the last produce stage keeps that adapt OFF
the train loop (overlapped with compute). ``postprocess`` defaults to
identity (the CPU-only path hands back the raw
:class:`MultiBinaryBatchDecodeResult`).

WHY ``collection_factory`` (not building the collection here): constructing
an :class:`IndexedMemmapCollection` needs the production ``readers_by_spec``
config, which the CONSUMER (ml-project) owns -- and byte-identity depends on
it being built EXACTLY as production. The factory is the consumer's no-arg
thunk returning a freshly-opened production-configured collection; this seam
opens ONE per thread and closes it on shutdown. It never builds the
collection itself.

THREAD-SAFETY (critical): an :class:`IndexedMemmapCollection`'s session +
handle caches, its shared :class:`ExitStack`, and the RNG are NOT
thread-safe. A single :class:`PoolConfig.produce` may be driven by SEVERAL
refill threads (``threads_per_config > 1``), so the collection AND the RNG
are THREAD-LOCAL -- each refill thread builds its OWN collection via
``collection_factory()`` (once, then reuses) and draws from its OWN RNG
stream. Per-thread RNG seeds are derived from the config seed via the
:class:`numpy.random.SeedSequence` spawn mechanism (guarded by a lock on the
shared parent), so distinct threads draw distinct, reproducible streams --
mirroring :mod:`._vector_batch_source` exactly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    VariantPadding,
)
from tokenizer.aligned_data.sorted_index._collection._collection import (
    IndexedMemmapCollection,
)
from tokenizer.aligned_data.sorted_index._types import (
    MultiBinaryBatchDecodeResult,
)

from ._produce import CloseableProduce


__all__ = [
    "CrossDecodeParams",
    "make_cross_binary_produce",
]


@dataclass(frozen=True)
class CrossDecodeParams:
    """The ``load_batch_cross_depth`` knobs shared by every draw of a config.

    Bundles exactly the cross-binary primitive's parameters threaded into
    every :meth:`IndexedMemmapCollection.load_batch_cross_depth` call.
    Unlike the single-binary :class:`DecodeParams` (where B is a property of
    the sampler's draw, ``len(section_pointers)``), the cross-binary B is a
    DRAW parameter: ``batch_size`` is how many pointers the cross-(binary x
    spec) urn draws, and ``target_length`` / ``band`` select the length
    universe sampled over. ``context_len`` is the sequence length L; the
    rest mirror the primitive's same-named flags.
    """

    target_length: int
    batch_size: int
    context_len: int
    num_variants_per_section: int = 1
    band: Optional[Tuple[int, int]] = None
    variant_padding: VariantPadding = VariantPadding.PAD_NULL
    inlined_equivalent_call_targets_only: bool = True
    include_fid_sidecar: bool = False


def _identity(
    result: MultiBinaryBatchDecodeResult,
) -> MultiBinaryBatchDecodeResult:
    """Default ``postprocess``: hand back the raw decode result (CPU-only)."""
    return result


def make_cross_binary_produce(
    *,
    collection_factory: Callable[[], IndexedMemmapCollection],
    params: CrossDecodeParams,
    postprocess: Callable[[MultiBinaryBatchDecodeResult], Any] = _identity,
    seed: Optional[int] = None,
) -> "CloseableProduce":
    """Build the no-arg cross-binary ``produce`` callable for a :class:`PoolConfig`.

    The returned object is callable (``produce()`` runs the pipeline) AND
    honours the :class:`CloseableProduce` seam: its ``close()`` releases the
    CALLING thread's collection (closing its sessions + handles via the
    collection's own :meth:`IndexedMemmapCollection.close`). The pool's
    refill loop calls ``close()`` in a ``finally`` when each worker stops, so
    the per-thread collection is released deterministically (mirroring the
    repo's explicit-close lifecycle) instead of waiting on thread exit.

    Each call runs the full produce pipeline: ensure THIS thread's collection
    is open (built once via ``collection_factory`` then reused), drive
    :meth:`IndexedMemmapCollection.load_batch_cross_depth` with the bundled
    :class:`CrossDecodeParams` + this thread's RNG, and run ``postprocess`` on
    the FULL :class:`MultiBinaryBatchDecodeResult` (which carries per-row
    binary identity) -- returning whatever ``postprocess`` returns (an
    upload-ready torch-tensor pytree for the GPU path, or the raw result
    under the default identity for the CPU-only path).

    Parameters
    ----------
    collection_factory:
        The consumer's no-arg thunk returning a freshly-opened,
        PRODUCTION-configured :class:`IndexedMemmapCollection`. This seam
        opens ONE per refill thread (collections are not thread-safe) and
        closes it on shutdown. Byte-identity to production depends on the
        thunk building the collection EXACTLY as production does (same
        ``readers_by_spec`` / specs / vocab) -- the consumer owns that.
    params:
        The :class:`CrossDecodeParams` bundle threaded into every
        ``load_batch_cross_depth`` call (carries B = ``batch_size``,
        ``target_length`` / ``band`` selecting the length universe, L =
        ``context_len``, + the decode flags).
    postprocess:
        The FINAL produce stage -- a user transform run on the cross-binary
        decode result, ON the worker thread (off the train loop), returning
        the upload-ready batch. Defaults to identity (raw
        :class:`MultiBinaryBatchDecodeResult` for the CPU-only path). For the
        GPU path this is where numpy -> torch-tensor pytree happens.
    seed:
        Base seed for the per-thread RNG streams. ``None`` => a fresh
        non-reproducible generator per thread.
    """
    return _CrossProduceState(
        collection_factory=collection_factory,
        params=params,
        postprocess=postprocess,
        seed=seed,
    )


class _CrossProduceState:
    """The callable + closeable cross-binary produce, THREAD-LOCAL collection.

    Implements the :class:`CloseableProduce` seam: ``__call__()`` runs one
    cross-binary produce pipeline; ``close()`` releases the CALLING thread's
    collection. The collection + RNG live in :class:`threading.local` so each
    refill thread builds + reuses its OWN production-configured collection and
    draws from its OWN RNG stream -- an :class:`IndexedMemmapCollection`'s
    session/handle caches + :class:`ExitStack` + the RNG are not safe to share
    across threads, and an independent RNG per thread keeps each thread's
    draws reproducible from ``seed``. ``close()`` therefore frees exactly what
    the calling worker opened (the pool calls it per worker on shutdown).
    """

    def __init__(
        self,
        *,
        collection_factory: Callable[[], IndexedMemmapCollection],
        params: CrossDecodeParams,
        postprocess: Callable[[MultiBinaryBatchDecodeResult], Any],
        seed: Optional[int],
    ) -> None:
        self._collection_factory = collection_factory
        self._params = params
        self._postprocess = postprocess
        self._seed = seed
        self._local = threading.local()
        # Hand each thread a distinct, reproducible RNG stream: a SeedSequence
        # spawned per thread (None seed -> entropy-seeded, non-reproducible).
        # ``spawn`` is the documented thread-safe way to fork independent,
        # non-overlapping child streams; guard the shared parent state.
        self._seed_seq = np.random.SeedSequence(seed)
        self._spawn_lock = threading.Lock()

    def _thread_state(self):
        local = self._local
        if not hasattr(local, "rng"):
            with self._spawn_lock:
                child = self._seed_seq.spawn(1)[0]
            local.rng = np.random.default_rng(child)
            # This thread's own production-configured collection, built once
            # on first draw and reused for every later draw on this thread.
            local.collection = self._collection_factory()
        return local

    def __call__(self) -> Any:
        local = self._thread_state()
        p = self._params
        result = local.collection.load_batch_cross_depth(
            target_length=p.target_length,
            batch_size=p.batch_size,
            rng=local.rng,
            band=p.band,
            context_len=p.context_len,
            num_variants_per_section=p.num_variants_per_section,
            variant_padding=p.variant_padding,
            inlined_equivalent_call_targets_only=(
                p.inlined_equivalent_call_targets_only
            ),
            include_fid_sidecar=p.include_fid_sidecar,
        )
        # FINAL produce stage: the user transform (numpy -> upload-ready
        # pytree for the GPU path; identity for CPU-only), run here on the
        # worker thread so it overlaps the train loop.
        return self._postprocess(result)

    def close(self) -> None:
        """Release the CALLING thread's open collection.

        The :class:`CloseableProduce` seam (called by the pool's refill loop
        in a ``finally`` when this worker stops). Closes THIS thread's
        collection -- the collection's own :meth:`IndexedMemmapCollection.close`
        releases every session + handle it opened, on its own
        :class:`ExitStack` -- and clears the thread-local so a later draw
        rebuilds fresh. The collection is thread-local, so a worker frees
        exactly what it opened, mirroring the explicit-close discipline of
        :meth:`_ProduceState.close`. A thread that never drew anything has no
        collection yet -> a clean no-op.
        """
        local = self._local
        collection = getattr(local, "collection", None)
        if collection is not None:
            try:
                collection.close()
            finally:
                local.collection = None
