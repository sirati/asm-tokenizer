"""Construction-time DataLoader facade: ONE postprocess, fanned out + wired.

Single concern -- ERGONOMIC ASSEMBLY, DECODE-AGNOSTIC. This module composes
the existing pieces (a decode ``produce`` seam -> :class:`ReadyPool` ->
optional :class:`GpuReadyPool`) behind ONE constructor so a consumer states
its configs + a SINGLE construction-time ``postprocess`` once, instead of
hand-building a produce closure per config and assembling the pools itself.
It REIMPLEMENTS none of the pool / gpu / decode logic -- every method
delegates to the wrapped pool(s); the only thing this module owns is the
wiring.

DECODE-AGNOSTIC (why the facade no longer hardcodes a seam): the training
distribution is EITHER single-binary (``vector_batch_tokens``) OR
cross-binary x cross-depth (``load_batch_cross_depth``), and the facade must
drive BOTH as first-class peers with NO branching on seam type. Each
:class:`DataLoaderConfig` therefore carries a ``make_produce`` THUNK --
``Callable[[postprocess], CloseableProduce]`` -- already bound to that
config's decode seam + parameters; the facade only resolves the postprocess
and calls the thunk. The two classmethod constructors
(:meth:`DataLoaderConfig.single_binary` / :meth:`DataLoaderConfig.cross_binary`)
build the thunk over :func:`make_vector_batch_produce` /
:func:`make_cross_binary_produce` respectively, so the facade never imports
either seam's internals beyond that wiring and never special-cases a seam.

THE REQUIREMENT (why this exists): "during construction a postprocessing
function is passed; it runs on the dataloader results and returns what is
uploaded to the GPU." That one ``postprocess`` is fanned out by passing the
SAME callable into every config's ``make_produce`` thunk, so it runs as the
final produce stage ON THE WORKER THREAD (off the train loop), overlapped
with compute -- NOT at upload time.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Optional, Sequence, Union

from ._config import PoolConfig
from ._cross_binary_source import (
    CrossDecodeParams,
    make_cross_binary_produce,
)
from ._gpu_pool import GpuReadyPool
from ._pool import ReadyPool
from ._produce import CloseableProduce
from ._vector_batch_source import (
    DecodeParams,
    Sampler,
    _identity,
    make_vector_batch_produce,
)


__all__ = ["DataLoaderConfig", "VectorBatchDataLoader"]


#: A config's decode-seam thunk: given the resolved postprocess (the FINAL
#: produce stage), return the bound :class:`CloseableProduce`. The seam +
#: its parameters are already captured inside; the facade only resolves the
#: postprocess and calls it -- which is what keeps the facade decode-agnostic.
MakeProduce = Callable[[Callable[[Any], Any]], CloseableProduce]


@dataclass(frozen=True)
class DataLoaderConfig:
    """One registered, DECODE-AGNOSTIC config for :class:`VectorBatchDataLoader`.

    Bundles the per-config knobs the facade needs to build that config's
    produce + register it WITHOUT the facade knowing which decode seam backs
    it: ``key`` (retrieval label, any hashable), ``make_produce`` (the THUNK
    -- given the resolved postprocess, returns the bound
    :class:`CloseableProduce` over whichever decode seam this config uses),
    and ``ready_depth`` (keep-N-ready / backpressure bound). ``postprocess``
    is the OPTIONAL per-config override: ``None`` defers to the dataloader's
    construction-level postprocess.

    Build instances via the classmethod constructors -- they capture the
    seam + its parameters inside ``make_produce`` so the facade stays
    seam-agnostic:

      * :meth:`single_binary` -- the per-binary ``vector_batch_tokens`` seam
        (B = ``len(section_pointers)``, sampler-driven).
      * :meth:`cross_binary` -- the cross-binary x cross-depth
        ``load_batch_cross_depth`` seam (B = ``params.batch_size``, a draw
        parameter), returning the per-row-identity
        :class:`MultiBinaryBatchDecodeResult`.
    """

    key: Hashable
    make_produce: MakeProduce
    ready_depth: int = 4
    postprocess: Optional[Callable[[Any], Any]] = None

    @classmethod
    def single_binary(
        cls,
        *,
        key: Hashable,
        base_path: Union[str, Path],
        sampler: Sampler,
        decode_params: DecodeParams,
        ready_depth: int = 4,
        postprocess: Optional[Callable[[Any], Any]] = None,
        vocab_manager: Any = None,
        seed: Optional[int] = None,
    ) -> "DataLoaderConfig":
        """A config for the single-binary ``vector_batch_tokens`` seam.

        Captures ``base_path`` / ``sampler`` / ``decode_params`` /
        ``vocab_manager`` / ``seed`` inside a ``make_produce`` thunk over
        :func:`make_vector_batch_produce` so the facade only threads the
        resolved postprocess through. B is the sampler's draw
        (``len(section_pointers)``).
        """
        return cls(
            key=key,
            make_produce=lambda pp: make_vector_batch_produce(
                base_path=base_path,
                sampler=sampler,
                decode_params=decode_params,
                postprocess=pp,
                vocab_manager=vocab_manager,
                seed=seed,
            ),
            ready_depth=ready_depth,
            postprocess=postprocess,
        )

    @classmethod
    def cross_binary(
        cls,
        *,
        key: Hashable,
        collection_factory: Callable[[], Any],
        params: CrossDecodeParams,
        ready_depth: int = 4,
        postprocess: Optional[Callable[[Any], Any]] = None,
        seed: Optional[int] = None,
    ) -> "DataLoaderConfig":
        """A config for the cross-binary x cross-depth ``load_batch_cross_depth`` seam.

        Captures ``collection_factory`` / ``params`` / ``seed`` inside a
        ``make_produce`` thunk over :func:`make_cross_binary_produce` so the
        facade only threads the resolved postprocess through. B is a draw
        parameter (``params.batch_size``); each batch carries per-row binary
        identity (``binary_id_per_row`` / ``binary_names``).
        """
        return cls(
            key=key,
            make_produce=lambda pp: make_cross_binary_produce(
                collection_factory=collection_factory,
                params=params,
                postprocess=pp,
                seed=seed,
            ),
            ready_depth=ready_depth,
            postprocess=postprocess,
        )

    @classmethod
    def from_produce(
        cls,
        *,
        key: Hashable,
        produce: CloseableProduce,
        ready_depth: int = 4,
    ) -> "DataLoaderConfig":
        """Register an ALREADY-BUILT produce (a consumer-owned closure).

        For the case where the consumer composes its OWN
        :class:`CloseableProduce` -- e.g. a redraw-retry wrapper around
        ``make_cross_binary_produce(..., postprocess=identity)`` that
        validates each draw against an asm-domain predicate and redraws
        (advancing the produce's rng) before translating. That
        validity/retry concern stays the consumer's; this only plugs the
        finished produce into the threaded keep-N + GPU-overlap engine. The
        ``make_produce`` thunk therefore IGNORES the facade's postprocess
        (the produce already bakes in whatever post-processing it wants) --
        ``lambda _pp: produce``.
        """
        return cls(
            key=key,
            make_produce=lambda _pp: produce,
            ready_depth=ready_depth,
            postprocess=None,
        )


class VectorBatchDataLoader:
    """Construction-time-postprocess facade over the ready pool stack.

    Constructed with a sequence of :class:`DataLoaderConfig` (each carrying
    its own decode-seam thunk + parameters), ONE ``postprocess`` applied to
    every config (unless a config overrides it), and -- optionally -- a CUDA
    ``device`` (+ cuda knobs) to enable the pipelined GPU path. For each
    config it builds that config's produce via ``cfg.make_produce(...)``,
    constructs the :class:`ReadyPool`, and (when a device is given) wraps a
    :class:`GpuReadyPool`. It is DECODE-AGNOSTIC: the thunk carries the seam,
    so the facade never branches on single- vs cross-binary.

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
        configs: Sequence[DataLoaderConfig],
        postprocess: Callable[[Any], Any] = _identity,
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
        # override) into each config's decode-seam thunk, so it runs on the
        # worker thread as the final produce stage. Everything else here is
        # pure composition of the existing seams -- the thunk carries which
        # decode seam each config uses, so this loop never branches on it.
        pool_configs = [
            PoolConfig(
                key=cfg.key,
                produce=cfg.make_produce(cfg.postprocess or self._postprocess),
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
