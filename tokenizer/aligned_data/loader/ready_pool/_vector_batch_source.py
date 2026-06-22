"""The DECODE seam: a ``produce`` closure over ``vector_batch_tokens``.

Single concern: build the OPAQUE no-arg ``produce`` callable a
:class:`PoolConfig` registers -- one that, per call, runs the full produce
PIPELINE on the worker thread: draw a section-pointer batch from a
PLUGGABLE sampler, decode it through
:func:`...vector_batch._entry.vector_batch_tokens`, then run the user
``postprocess`` (the FINAL stage) to return an upload-ready batch (e.g. a
torch-tensor pytree). It opens each binary's handles + session ONCE and
reuses them across draws (amortizing the adjacency build + catalog parse
the handles pay at open). This is the ONLY module in the package that
imports the ``vector_batch`` decode primitive; neither :class:`ReadyPool`
nor :class:`GpuReadyPool` knows it exists.

WHY ``postprocess`` runs HERE (the worker thread, the final produce
stage): the decode result is CPU numpy; Layer 2 uploads torch-tensor
leaves. Running the user-supplied numpy->torch (or any) transform as the
last produce stage keeps that adapt OFF the train loop (overlapped with
compute) AND keeps the array-library concern off the pools -- what sits in
the ready buffer is already upload-ready. ``postprocess`` defaults to
identity (the CPU-only path hands back the raw
:class:`VectorBatchResult`).

PLUGGABLE SAMPLING: the sampling policy is NOT hardcoded. The caller
passes a ``sampler`` -- ``Callable[[np.random.Generator], Draw]`` where a
:class:`Draw` is ``(binary_name, [SectionPointerSpec, ...])`` -- so any
draw policy (a fixed batch_size B over a chosen arm, a length-banded draw,
an :class:`AlignedDataLoader`-backed mixed draw) plugs in without this
module knowing the policy. The seam owns ONLY: the RNG, the open-once
per-binary handle/session cache, and threading the draw into
``vector_batch_tokens`` with the bundled :class:`DecodeParams`.

THREAD-SAFETY: a single :class:`PoolConfig.produce` may be driven by
SEVERAL refill threads (``threads_per_config > 1``). The per-binary
handle/session cache + the RNG are therefore THREAD-LOCAL -- each refill
thread opens its own mmap views + draws from its own RNG stream, so no
mmap handle or generator is shared across threads (mmap views + the
session's file handles are not safe to share concurrently). Per-thread
RNG seeds are derived from the config seed via the generator's spawn
mechanism so distinct threads draw distinct, reproducible streams.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.vector_batch._entry import (
    vector_batch_tokens,
)
from tokenizer.aligned_data.loader.vector_batch._result import (
    VectorBatchResult,
)
from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    open_vector_batch_arm_set,
)

from ._produce import CloseableProduce


__all__ = ["DecodeParams", "Draw", "Sampler", "make_vector_batch_produce"]


#: One sampler draw: the binary to decode from + its section pointers.
Draw = Tuple[str, List[SectionPointerSpec]]

#: A pluggable sampling policy: given an RNG, return one :data:`Draw`. The
#: batch size B is ``len(section_pointers)`` -- the sampler owns it.
Sampler = Callable[[np.random.Generator], Draw]


@dataclass(frozen=True)
class DecodeParams:
    """The ``vector_batch_tokens`` knobs shared by every draw of a config.

    Bundles exactly the decode-side parameters (``context_len`` is the
    sequence length L; ``max_depth`` the splice depth; the rest mirror the
    primitive's same-named flags). The batch size B is NOT here -- it is a
    property of the sampler's draw (``len(section_pointers)``).
    """

    context_len: int
    num_variants_per_section: int = 1
    max_depth: Union[int, np.ndarray] = 0
    variant_padding: VariantPadding = VariantPadding.PAD_NULL
    include_fid_sidecar: bool = False
    unmatched_inline: bool = False
    unmatched_inline_depth: int = 3


def _identity(result: VectorBatchResult) -> VectorBatchResult:
    """Default ``postprocess``: hand back the raw decode result (CPU-only)."""
    return result


def make_vector_batch_produce(
    *,
    base_path: Union[str, Path],
    sampler: Sampler,
    decode_params: DecodeParams,
    postprocess: Callable[[VectorBatchResult], Any] = _identity,
    vocab_manager=None,
    seed: Optional[int] = None,
) -> "CloseableProduce":
    """Build the no-arg ``produce`` callable for a :class:`PoolConfig`.

    The returned object is callable (``produce()`` runs the pipeline) AND
    honours the :class:`CloseableProduce` seam: its ``close()`` releases the
    CALLING thread's per-binary session + handle cache. The pool's refill
    loop calls ``close()`` in a ``finally`` when each worker stops, so the
    open-once-per-thread handles are released deterministically (mirroring
    the repo's explicit-close lifecycle) instead of waiting on thread exit.

    Each call runs the full produce pipeline: draw
    ``(binary_name, section_pointers)`` from ``sampler``, ensure that
    binary's both-arms handles + session are open (opened once per binary
    per thread, then reused), decode via :func:`vector_batch_tokens`, and
    run ``postprocess`` on the result -- returning whatever ``postprocess``
    returns (an upload-ready torch-tensor pytree for the GPU path, or the
    raw :class:`VectorBatchResult` under the default identity for the
    CPU-only path).

    Parameters
    ----------
    base_path:
        The memmap directory (the same key the index build +
        :class:`BinaryDataset` use).
    sampler:
        The PLUGGABLE draw policy (see :data:`Sampler`). Owns the batch
        size + arm + which binary each draw targets; this seam never
        interprets the policy.
    decode_params:
        The :class:`DecodeParams` bundle threaded into every
        ``vector_batch_tokens`` call.
    postprocess:
        The FINAL produce stage -- a user transform run on the decode
        result, ON the worker thread (off the train loop), returning the
        upload-ready batch. Defaults to identity (raw decode result for the
        CPU-only path). For the GPU path this is where numpy ->
        torch-tensor pytree happens, so Layer 2 only pins + uploads tensor
        leaves and the array-library concern never reaches the pools.
    vocab_manager:
        Optional unified :class:`VocabularyManager` so variant-axis tokens
        resolve against the corpus-wide ID space (threaded into each
        :class:`BinaryDataset`). ``None`` defers to the per-binary vocab.
    seed:
        Base seed for the per-thread RNG streams. ``None`` => a fresh
        non-reproducible generator per thread.
    """
    base_path = Path(base_path)
    return _ProduceState(
        base_path=base_path,
        sampler=sampler,
        decode_params=decode_params,
        postprocess=postprocess,
        vocab_manager=vocab_manager,
        seed=seed,
    )


class _ProduceState:
    """The callable + closeable produce, with THREAD-LOCAL handle caches.

    Implements the :class:`CloseableProduce` seam: ``__call__()`` runs one
    produce pipeline; ``close()`` releases the CALLING thread's cache. The
    cache + RNG live in :class:`threading.local` so each refill thread opens
    + reuses its OWN per-binary mmap handles + session and draws from its OWN
    RNG stream -- mmap views and the session's file handles are not safe to
    share across threads, and an independent RNG per thread keeps each
    thread's draws reproducible from ``seed``. ``close()`` therefore frees
    exactly what the calling worker opened (the pool calls it per worker on
    shutdown).
    """

    def __init__(
        self,
        *,
        base_path: Path,
        sampler: Sampler,
        decode_params: DecodeParams,
        postprocess: Callable[[VectorBatchResult], Any],
        vocab_manager,
        seed: Optional[int],
    ) -> None:
        self._base_path = base_path
        self._sampler = sampler
        self._params = decode_params
        self._postprocess = postprocess
        self._vocab_manager = vocab_manager
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
            # binary_name -> (BinarySession ctx, opened session, arm-set handles)
            local.cache = {}
        return local

    def __call__(self) -> Any:
        local = self._thread_state()
        binary_name, section_pointers = self._sampler(local.rng)
        session, handles = self._ensure_open(local, binary_name)
        p = self._params
        result = vector_batch_tokens(
            session,
            section_pointers,
            handles=handles,
            num_variants_per_section=p.num_variants_per_section,
            context_len=p.context_len,
            max_depth=p.max_depth,
            variant_padding=p.variant_padding,
            rng=local.rng,
            include_fid_sidecar=p.include_fid_sidecar,
            unmatched_inline=p.unmatched_inline,
            unmatched_inline_depth=p.unmatched_inline_depth,
        )
        # FINAL produce stage: the user transform (numpy -> upload-ready
        # pytree for the GPU path; identity for CPU-only), run here on the
        # worker thread so it overlaps the train loop.
        return self._postprocess(result)

    def _ensure_open(self, local, binary_name: str):
        """Open ``binary_name``'s session + both-arms handles once; reuse.

        Opened lazily on first draw of a binary on THIS thread and cached
        for every later draw of the same binary -- amortizing the adjacency
        build + columnar catalog parse the handles pay at open across the
        many decode calls the pool drives.
        """
        cache: Dict[str, Tuple] = local.cache
        if binary_name not in cache:
            dataset = BinaryDataset(
                self._base_path,
                binary_name,
                vocab_manager=self._vocab_manager,
            )
            session_cm = dataset.open_session()
            session = session_cm.__enter__()
            handles = open_vector_batch_arm_set(self._base_path, binary_name)
            cache[binary_name] = (session_cm, session, handles)
        _session_cm, session, handles = cache[binary_name]
        return session, handles

    def close(self) -> None:
        """Release the CALLING thread's open-once session + handle cache.

        The :class:`CloseableProduce` seam (called by the pool's refill loop
        in a ``finally`` when this worker stops). Closes every cached
        binary's arm-set handles + exits its session context for THIS thread
        only -- the cache is thread-local, so a worker frees exactly what it
        opened, mirroring the explicit-close discipline of
        :meth:`VectorBatchArmSet.close` / ``BinarySession``'s context exit.
        A thread that never drew anything has no cache yet -> a clean no-op.
        Each binary is released independently so one failure does not strand
        the rest; the cache is cleared so a later draw re-opens fresh.
        """
        local = self._local
        cache: Dict[str, Tuple] = getattr(local, "cache", None) or {}
        for session_cm, _session, handles in cache.values():
            try:
                handles.close()
            finally:
                session_cm.__exit__(None, None, None)
        cache.clear()
