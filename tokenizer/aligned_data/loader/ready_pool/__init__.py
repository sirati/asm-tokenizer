"""In-process, MULTITHREADED keep-N-ready batch pool over the c3 decode.

Two composable layers, separate concerns -- a threaded CPU ready-pool and
an opt-in CUDA H2D overlap that composes over it -- wrapping the existing
synchronous decode primitive
:func:`...vector_batch._entry.vector_batch_tokens`.

LAYER 1 -- CPU ready-pool (torch-free):
    :class:`PoolConfig`  -- one registered ``(produce, keep-N-ready)`` config.
    :class:`ReadyPool`   -- background THREADS keep each config's FIFO buffer
                            filled to ``ready_depth``; :meth:`ReadyPool.get`
                            pops the oldest (FIFO per config). NO
                            multiprocessing -- the decode releases the GIL in
                            its Rust kernels, so threads overlap real work.
    :class:`ReadyPoolWorkerDied` -- raised by ``get`` when a refill thread
                            died with the batch still owed (never a hang).

LAYER 2 -- GPU upload overlap (opt-in torch; composes over Layer 1):
    :class:`GpuReadyPool` -- pipelined double-buffered H2D on a
                            consumer-supplied copy stream; ``prime`` then
                            ``get`` name-the-next / collect-the-previous so
                            mixed (B, L) config sequences interleave. All
                            CUDA behind the injected ``CudaBackend`` seam.

DECODE SEAM (the only module importing ``vector_batch``):
    :func:`make_vector_batch_produce` -- build the produce closure: draw via
                            a PLUGGABLE sampler -> ``vector_batch_tokens`` ->
                            user ``postprocess`` (final, upload-ready stage),
                            opening per-binary handles + session once.
    :class:`DecodeParams` -- the decode knobs bundle; :data:`Sampler` /
                            :data:`Draw` -- the pluggable sampling seam.

CPU-only usage::

    cfg = PoolConfig(
        key="train",
        produce=make_vector_batch_produce(
            base_path=BASE, sampler=my_sampler,
            decode_params=DecodeParams(context_len=L, num_variants_per_section=1),
        ),
        ready_depth=4,
    )
    with ReadyPool([cfg]) as pool:
        batch = pool.get("train")            # VectorBatchResult (CPU numpy)

GPU-overlap usage (the user ``postprocess`` adapts numpy -> torch pytree)::

    cfg = PoolConfig(
        key="train",
        produce=make_vector_batch_produce(
            base_path=BASE, sampler=my_sampler,
            decode_params=DecodeParams(context_len=L),
            postprocess=to_torch_pytree,    # final produce stage, off train loop
        ),
        ready_depth=4,
    )
    with ReadyPool([cfg]) as pool, GpuReadyPool(pool, device="cuda:0") as gpool:
        S = torch.cuda.Stream(device="cuda:0")
        gpool.prime("train", stream=S)       # launch the first upload
        for _ in range(num_steps):
            batch = gpool.get("train", stream=S)  # GPU-resident; next upload launched
            train_step(batch)
"""

from __future__ import annotations

from ._config import PoolConfig
from ._gpu_pool import GpuReadyPool
from ._pool import ReadyPool, ReadyPoolWorkerDied
from ._vector_batch_source import (
    DecodeParams,
    Draw,
    Sampler,
    make_vector_batch_produce,
)


__all__ = [
    "DecodeParams",
    "Draw",
    "GpuReadyPool",
    "PoolConfig",
    "ReadyPool",
    "ReadyPoolWorkerDied",
    "Sampler",
    "make_vector_batch_produce",
]
