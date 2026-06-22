"""Layer 2: double-buffered, PIPELINED CUDA H2D overlap over the pool.

Single concern: hand a consumer a GPU-resident batch whose async H2D
upload was launched on a SEPARATE, consumer-supplied copy stream while the
consumer was still computing the PREVIOUS batch -- so the upload overlaps
compute instead of serializing in front of it. Composes OVER
:class:`ReadyPool` (Layer 1 owns the threaded CPU decode buffering) and
delegates EVERY ``torch.cuda`` primitive to an injected
:class:`CudaBackend` (the same seam :mod:`...gpu_prefetcher` uses), so the
overlap order is unit-testable WITHOUT a GPU via a fake backend, and a
future Rust-owned-stream backend is a pure drop-in.

This module knows NOTHING about decode or sampling -- it only pulls opaque
batches from the ready pool and moves their tensor leaves
(:func:`map_tensor_leaves`, recurses by container TYPE only). The pool's
batches are expected to ALREADY be upload-ready torch-tensor pytrees (the
worker thread ran the user ``postprocess`` as the final produce stage); so
this module never learns a field name and the numpy->torch adapt happened
off the train loop.

PIPELINED DOUBLE-BUFFER CONTRACT (Option P -- name-the-next, collect-the-
previous, so MIXED config sequences interleave without stalling):

    gpool.prime(k0, stream=S)        # launch the first upload (a k0 batch)
    b0 = gpool.get(k0, stream=S)     # wait+return the batch named by prime;
                                     # pop a ready k0 batch, launch its H2D on
                                     # S, record its event, stash as in-flight
    compute(b0)
    b1 = gpool.get(k1, stream=S)     # wait+return the batch named last call
                                     # (a k0 batch); launch a k1 batch's H2D ...

  Each :meth:`get` (a) waits the in-flight upload's copy event on the
  consumer's CURRENT (compute) stream and ``record_stream``s every moved
  leaf (the cross-stream-reuse fix) then returns that batch, and (b) pops a
  ready postprocessed batch of the NEWLY-named ``key``, launches its async
  H2D on ``stream``, records the event, and stashes it as the new in-flight
  upload -- so it overlaps the consumer's compute on the returned batch.
  The CONSUMER drives the config sequence; the pool keeps each config's
  buffer full, so naming a config is an O(1) pop + async launch (naming a
  config whose buffer is momentarily empty blocks on its decode).

  Option C (``c_fallback=True``): no copy stream is used; :meth:`get`
  returns the PINNED HOST batch and the consumer owns the ``.to(cuda)``.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Optional

from tokenizer.aligned_data.loader.gpu_prefetcher._cuda_backend import (
    CudaBackend,
    TorchCudaBackend,
)
from tokenizer.aligned_data.loader.gpu_prefetcher._pytree import (
    map_tensor_leaves,
)
from tokenizer.aligned_data.loader.gpu_prefetcher._to_device import (
    default_leaf_pred,
    default_to_device,
    pin_host,
)

from ._pool import ReadyPool

try:
    import torch
except ImportError:  # pragma: no cover - exercised only where torch absent
    torch = None  # type: ignore[assignment]


__all__ = ["GpuReadyPool"]


class GpuReadyPool:
    """Pipelined double-buffered H2D overlap layered over a :class:`ReadyPool`.

    See the module docstring for the contract. Public surface:
    :meth:`prime`, :meth:`get` (Option P / Option C), and the
    context-manager protocol (a no-op pass-through -- the wrapped pool owns
    thread lifecycle).
    """

    def __init__(
        self,
        pool: ReadyPool,
        *,
        device: Any,
        cuda_backend: Optional[CudaBackend] = None,
        to_device: Optional[Callable[[Any, Any], Any]] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
        c_fallback: bool = False,
    ) -> None:
        self._pool = pool
        self._device = torch.device(device) if torch is not None else device
        # The ONE seam owning every torch.cuda primitive (copy-stream move,
        # event record, consumer wait_event + record_stream). Production
        # default = real torch.cuda; tests inject a fake recording op order
        # + stream identities, so the overlap is validatable WITHOUT a GPU.
        self._cuda = cuda_backend or TorchCudaBackend()
        self._to_device = to_device or default_to_device
        self._is_leaf = is_leaf or default_leaf_pred
        self._c_fallback = c_fallback

        # The SINGLE in-flight upload across all configs: the (gpu_batch,
        # event) whose async copy the previous prime/get launched. The
        # pipeline names the NEXT batch and collects the PREVIOUS one, so a
        # mixed (curriculum) config sequence interleaves through this one
        # slot without per-key stalling. ``None`` = not primed.
        self._in_flight: Optional[Any] = None

    def __enter__(self) -> "GpuReadyPool":
        return self

    def __exit__(self, *_exc: Any) -> None:
        # Lifecycle of the threaded decode workers belongs to the wrapped
        # ReadyPool; this layer holds no threads of its own.
        return None

    # -- consumer API ------------------------------------------------------
    def prime(self, key: Hashable, *, stream: Any) -> None:
        """Launch the FIRST async upload (a ``key`` batch) on ``stream``.

        Call once before the first :meth:`get` so that get has a primed
        in-flight upload to wait+return. Option C: a no-op (no overlap).
        """
        if self._c_fallback:
            return
        self._in_flight = self._launch_upload(key, stream)

    def get(self, key: Hashable, *, stream: Any) -> Any:
        """Return the batch named on the PREVIOUS call; launch ``key``'s.

        Option P: waits the in-flight upload's copy event on the consumer's
        current (compute) stream + ``record_stream``s its leaves, then
        returns that batch. Before returning, pops a ready postprocessed
        batch of the newly-named ``key``, launches its async H2D on
        ``stream``, records the event, and stashes it as the new in-flight
        upload -- so it overlaps the consumer's compute on the returned
        batch. Call :meth:`prime` once before the first :meth:`get`.

        Option C (``c_fallback=True``): ``stream`` is ignored; returns the
        PINNED HOST batch of ``key`` and the consumer does the ``.to(cuda)``.
        """
        if self._c_fallback:
            cpu_batch = self._pool.get(key)
            return map_tensor_leaves(
                cpu_batch, pin_host, is_leaf=self._is_leaf
            )

        if self._in_flight is None:
            raise RuntimeError(
                "GpuReadyPool.get requires a primed in-flight upload; call "
                "prime(key, stream=...) once before the first get()"
            )
        # Collect the PREVIOUS batch: wait its copy event on the compute
        # stream + record_stream every moved leaf (cross-stream-reuse fix).
        gpu_batch, event = self._in_flight
        self._cuda.wait(gpu_batch, event, self._device, self._is_leaf)

        # Name the NEXT batch: pop + launch its upload so it overlaps the
        # consumer's compute on the batch we are about to return.
        self._in_flight = self._launch_upload(key, stream)
        return gpu_batch

    def _launch_upload(self, key: Hashable, stream: Any) -> Any:
        """Pop one upload-ready batch of ``key``; launch its async H2D.

        Returns ``(gpu_batch, event)`` with the copy enqueued on ``stream``
        and the event recorded after it -- the :class:`CudaBackend` owns
        those torch.cuda primitives. The bounded :meth:`ReadyPool.get` is
        the only place this layer blocks (and it surfaces worker death as a
        raise rather than a hang).
        """
        cpu_batch = self._pool.get(key)
        return self._cuda.upload(
            cpu_batch,
            self._device,
            stream,
            lambda t: self._to_device(t, self._device),
            self._is_leaf,
        )
