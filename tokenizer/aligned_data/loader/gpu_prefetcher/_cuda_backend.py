"""The CUDA overlap primitives, behind ONE injectable seam.

Single concern: own EVERY ``torch.cuda`` primitive the H2D-overlap path
needs -- create the copy stream, run the async move inside that stream's
context, record an event after the copy is enqueued, and (consumer side)
make the compute stream wait on that event AND register the uploaded
tensor's cross-stream use with the caching allocator. The prefetcher
orchestration depends only on this :class:`CudaBackend` surface; it never
touches ``torch.cuda.Stream`` / ``torch.cuda.stream`` / ``torch.cuda.Event``
/ ``torch.cuda.current_stream`` directly.

WHY a seam: torch is absent from the dev shell (and the one GPU is busy),
so the overlap ORCHESTRATION + ORDER must be validatable WITHOUT a GPU.
A fake backend (the tests inject one) records the ordered op sequence and
the stream identities; the production default :class:`TorchCudaBackend`
calls the real torch.cuda primitives, so production behavior is unchanged.

WHY ``record_stream`` (the correctness fix this seam carries): the GPU
output tensor is ALLOCATED on the copy stream but CONSUMED on the compute
stream. ``current_stream().wait_event(event)`` orders read-after-write
(compute can't read before the copy lands) but does NOT tell the caching
allocator the block is live on the compute stream -- the allocator tracks
liveness on the ALLOCATION (copy) stream only. Without ``record_stream``,
the NEXT upload on the copy stream can reuse the block while the compute
stream still reads the previous batch -> nondeterministic corruption. So
the consumer-side ``wait`` records each uploaded leaf onto the very stream
it waits on (the genuine consumer/compute stream, captured in the consumer
thread at ``get`` time). See PyTorch's CUDA caching-allocator guide and
``torch.Tensor.record_stream``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from ._pytree import map_tensor_leaves

try:
    import torch
except ImportError:  # pragma: no cover - exercised only where torch absent
    torch = None  # type: ignore[assignment]


class CudaBackend(Protocol):
    """The full CUDA-overlap primitive surface the prefetcher depends on.

    Three methods span the boundary: :meth:`make_copy_stream` (start, main
    thread), :meth:`upload` (upload thread), :meth:`wait` (get, consumer
    thread). Nothing else about torch.cuda crosses into the orchestrator.
    """

    def make_copy_stream(self, device: Any) -> Optional[Any]:
        """Create a SEPARATE copy stream on ``device`` (``None`` if no CUDA)."""

    def upload(
        self,
        cpu_batch: Any,
        device: Any,
        copy_stream: Any,
        move_one: Callable[[Any], Any],
        is_leaf: Callable[[Any], bool],
    ) -> tuple[Any, Any]:
        """Move every tensor leaf to ``device`` ON ``copy_stream``; record event.

        Returns ``(gpu_batch, event)`` where ``event`` was recorded on
        ``copy_stream`` AFTER the async copies were enqueued.
        """

    def wait(
        self,
        gpu_batch: Any,
        event: Any,
        device: Any,
        is_leaf: Callable[[Any], bool],
    ) -> None:
        """Consumer side: make the CURRENT stream wait on ``event`` and
        register every uploaded leaf's use on that current stream."""


class TorchCudaBackend:
    """Production default: the real torch.cuda primitives (behavior unchanged)."""

    def make_copy_stream(self, device: Any) -> Optional[Any]:
        if torch is None or device.type != "cuda" or not torch.cuda.is_available():
            return None
        return torch.cuda.Stream(device=device)

    def upload(
        self,
        cpu_batch: Any,
        device: Any,
        copy_stream: Any,
        move_one: Callable[[Any], Any],
        is_leaf: Callable[[Any], bool],
    ) -> tuple[Any, Any]:
        with torch.cuda.stream(copy_stream):
            gpu_batch = map_tensor_leaves(cpu_batch, move_one, is_leaf=is_leaf)
            event = torch.cuda.Event()
            event.record(copy_stream)
        return (gpu_batch, event)

    def wait(
        self,
        gpu_batch: Any,
        event: Any,
        device: Any,
        is_leaf: Callable[[Any], bool],
    ) -> None:
        consumer_stream = torch.cuda.current_stream(device=device)
        consumer_stream.wait_event(event)
        # Register the cross-stream use so the allocator won't reuse the
        # block (allocated on the copy stream) while compute still reads it.
        map_tensor_leaves(
            gpu_batch,
            lambda t: _record_stream(t, consumer_stream),
            is_leaf=is_leaf,
        )


def _record_stream(t: "torch.Tensor", stream: Any) -> "torch.Tensor":
    """Tag ``t`` as in-use on ``stream`` for the caching allocator; pass through."""
    t.record_stream(stream)
    return t
