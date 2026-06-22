"""Construction-time DataLoader facade: ONE postprocess, fanned out + wired.

Single concern -- ERGONOMIC ASSEMBLY. This module composes the existing
pieces (:func:`make_vector_batch_produce` -> :class:`ReadyPool` -> optional
:class:`GpuReadyPool`) behind ONE constructor so a consumer states its
configs + a SINGLE construction-time ``postprocess`` once, instead of
hand-building a produce closure per config and assembling the pools itself.
It REIMPLEMENTS none of the pool / gpu / decode logic -- every method
delegates to the wrapped pool(s); the only thing this module owns is the
wiring.

THE REQUIREMENT (why this exists): "during construction a postprocessing
function is passed; it runs on the dataloader results and returns what is
uploaded to the GPU." That one ``postprocess`` is fanned out by passing the
SAME callable into every config's :func:`make_vector_batch_produce`, so it
runs as the final produce stage ON THE WORKER THREAD (off the train loop),
overlapped with compute -- NOT at upload time.

PER-CONFIG OVERRIDE (a nicety that falls out cleanly): a
:class:`DataLoaderConfig` may carry its own ``postprocess``; when present it
is used for that config, else the construction-level one. This is plain
default-resolution (``config.postprocess or self._postprocess``) -- data
selection, not branching on any internal -- so it adds no special-casing.

CPU vs GPU surface (one facade, the device decides):

  * ``device=None`` -> CPU path: :meth:`get(key)` returns the postprocessed
    CPU batch straight off the ready pool.
  * ``device=<cuda>`` -> GPU pipelined path: :meth:`prime(key, stream=...)`
    then :meth:`get(key, stream=...)` drive the double-buffered H2D overlap
    via the wrapped :class:`GpuReadyPool`.

B (batch size) stays SAMPLER-DRIVEN (``B = len(section_pointers)``) exactly
as today -- the facade only carries config metadata and threads the shared
``base_path`` / ``vocab_manager`` / ``seed`` into each produce; it never
reaches into the sampler's draw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Optional, Sequence, Union

from ._config import PoolConfig
from ._gpu_pool import GpuReadyPool
from ._pool import ReadyPool
from ._vector_batch_source import (
    DecodeParams,
    Sampler,
    _identity,
    make_vector_batch_produce,
)


__all__ = ["DataLoaderConfig", "VectorBatchDataLoader"]


@dataclass(frozen=True)
class DataLoaderConfig:
    """One registered config for :class:`VectorBatchDataLoader`.

    Bundles the per-config knobs the facade needs to build that config's
    produce + register it: ``key`` (retrieval label, any hashable),
    ``sampler`` (the pluggable draw policy -- owns B + arm + binary),
    ``decode_params`` (the :class:`DecodeParams` bundle carrying context_len
    L + decode flags), and ``ready_depth`` (keep-N-ready / backpressure
    bound). ``postprocess`` is the OPTIONAL per-config override: ``None``
    defers to the dataloader's construction-level postprocess.
    """

    key: Hashable
    sampler: Sampler
    decode_params: DecodeParams
    ready_depth: int = 4
    postprocess: Optional[Callable[[Any], Any]] = None


class VectorBatchDataLoader:
    """Construction-time-postprocess facade over the ready pool stack.

    Constructed with the shared decode context (``base_path`` /
    ``vocab_manager`` / ``seed``), a sequence of :class:`DataLoaderConfig`,
    ONE ``postprocess`` applied to every config (unless a config overrides
    it), and -- optionally -- a CUDA ``device`` (+ cuda knobs) to enable the
    pipelined GPU path. It builds each config's produce via
    :func:`make_vector_batch_produce`, constructs the :class:`ReadyPool`,
    and (when a device is given) wraps a :class:`GpuReadyPool`.

    Public surface (context manager):

      * always: :meth:`get(key)` -- CPU path, the postprocessed batch.
      * when ``device`` is set: :meth:`prime(key, *, stream)` +
        :meth:`get(key, *, stream)` -- the GPU pipelined path (delegated to
        the wrapped :class:`GpuReadyPool`).

    The facade owns NO threads, NO CUDA, NO decode -- it delegates each call
    to the wrapped pool(s).
    """

    def __init__(
        self,
        *,
        base_path: Union[str, Path],
        configs: Sequence[DataLoaderConfig],
        postprocess: Callable[[Any], Any] = _identity,
        vocab_manager: Any = None,
        seed: Optional[int] = None,
        threads_per_config: int = 1,
        device: Any = None,
        cuda_backend: Any = None,
        to_device: Optional[Callable[[Any, Any], Any]] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
        c_fallback: bool = False,
    ) -> None:
        if not configs:
            raise ValueError("at least one DataLoaderConfig is required")
        self._postprocess = postprocess
        # Fan the SINGLE construction-time postprocess (or a config's clean
        # override) into each config's produce, so it runs on the worker
        # thread as the final produce stage. Everything else here is pure
        # composition of the existing seam.
        pool_configs = [
            PoolConfig(
                key=cfg.key,
                produce=make_vector_batch_produce(
                    base_path=base_path,
                    sampler=cfg.sampler,
                    decode_params=cfg.decode_params,
                    postprocess=cfg.postprocess or self._postprocess,
                    vocab_manager=vocab_manager,
                    seed=seed,
                ),
                ready_depth=cfg.ready_depth,
            )
            for cfg in configs
        ]
        self._pool = ReadyPool(
            pool_configs, threads_per_config=threads_per_config
        )
        # GpuReadyPool composes OVER the pool only when a device is given;
        # otherwise the facade is the CPU path and prime/get(stream) are not
        # available (calling the GPU surface without a device is a usage
        # error surfaced by :meth:`_require_gpu`).
        self._gpu: Optional[GpuReadyPool] = None
        if device is not None:
            self._gpu = GpuReadyPool(
                self._pool,
                device=device,
                cuda_backend=cuda_backend,
                to_device=to_device,
                is_leaf=is_leaf,
                c_fallback=c_fallback,
            )

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "VectorBatchDataLoader":
        self._pool.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        # The wrapped ReadyPool owns thread lifecycle (and, via the produce
        # close() seam, per-worker decode-handle release on shutdown); the
        # GpuReadyPool holds no threads of its own.
        self._pool.close()

    # -- consumer API ------------------------------------------------------
    def get(self, key: Hashable, *, stream: Any = None) -> Any:
        """Return one postprocessed batch for ``key``.

        CPU path (no device): pops the ready batch off the pool. GPU path (a
        device was given): requires a ``stream`` and delegates to the
        pipelined :meth:`GpuReadyPool.get` (waits the in-flight upload,
        launches ``key``'s next). The construction-time ``postprocess``
        already ran on the worker thread, so what comes back is upload-ready.
        """
        if self._gpu is None:
            return self._pool.get(key)
        if stream is None:
            raise TypeError(
                "this VectorBatchDataLoader was built with a device (GPU "
                "path); get() requires a stream= (call prime(key, stream) "
                "once first)"
            )
        return self._gpu.get(key, stream=stream)

    def prime(self, key: Hashable, *, stream: Any) -> None:
        """Launch the first async upload for the GPU pipelined path.

        Only valid when the facade was built with a ``device``; delegates to
        :meth:`GpuReadyPool.prime`. Call once before the first GPU
        :meth:`get`.
        """
        self._require_gpu("prime").prime(key, stream=stream)

    def _require_gpu(self, op: str) -> GpuReadyPool:
        if self._gpu is None:
            raise RuntimeError(
                f"{op}() is the GPU pipelined path; build the "
                "VectorBatchDataLoader with device=<cuda> to use it"
            )
        return self._gpu
