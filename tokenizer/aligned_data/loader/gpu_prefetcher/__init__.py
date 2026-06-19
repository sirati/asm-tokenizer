"""Generic double-buffered GPU batch prefetcher.

Single concern: overlap the NEXT batch's CPU decode (in a SEPARATE
process, off the train loop's core) and its H2D upload (on a SEPARATE
CUDA copy stream, overlapping compute) with the CURRENT batch's GPU
compute, so :meth:`GpuBatchPrefetcher.get` hands back a batch that is
ALREADY GPU-resident.

Boundary contract (the design-first sentence):

  *Given two PICKLABLE callables -- ``make_source() -> source`` (opens the
  consumer's data, e.g. a memmap, ONCE per worker AFTER spawn) and
  ``produce(source, request) -> batch`` (the consumer's decode +
  post-process, returning an OPAQUE pytree of CPU torch tensors plus
  arbitrary non-tensor leaves) -- and a stream of opaque ``request``
  objects, serve those batches back in FIFO order, each GPU-resident,
  WITHOUT the prefetcher ever knowing a single field name of the batch.*

The "no consumer field names" rule is enforced structurally: the only
thing this module does to a batch is :func:`_map_tensor_leaves`, a
generic pytree walk that recurses by Python container TYPE alone
(dataclass / dict / list / tuple / namedtuple) and applies a move
function to each ``torch.Tensor`` leaf. Non-tensor leaves pass through
untouched. No attribute is ever read by name; no batch shape is assumed.

WHY a separate PROCESS for decode, a separate THREAD for the upload, and
all CUDA in the MAIN process:

  * CUDA contexts are per-process. A child process cannot share the
    parent's CUDA context, so the worker is PURE CPU -- it returns a
    CPU/pinned batch over an IPC queue. All CUDA (copy stream, pin,
    async H2D, event) lives in the MAIN process.
  * The MAIN-process upload thread mirrors torch's ``pin_memory`` thread:
    it does only LIGHT work (pin memcpy + launch async copy on the copy
    stream + record an event), never the heavy decode -- so it never
    blocks the compute-launching main thread for long.
  * Double buffering: ``decode_ahead`` requests are kept in flight in the
    worker(s); ``gpu_ahead`` uploads are kept in flight on the copy
    stream. ``get()`` pops the oldest ready (gpu_batch, event), waits the
    event on the CURRENT (compute) stream so the kernels serialize
    correctly behind the copy, and returns the gpu_batch.

torch is OPT-IN: imported at module top. Only torch-using consumers
import this module. The orchestration + pytree traversal are testable
WITHOUT torch via dependency injection (``to_device`` + the leaf
predicate), which is why ``_map_tensor_leaves`` takes ``is_leaf``.

Usage (single-GPU, spawn, decode_ahead=2, gpu_ahead=1, Option P)::

    # --- module level (PICKLABLE; no closures over the train loop) ---
    def make_source():
        # Opens the memmap / dataset ONCE inside the worker, post-spawn.
        return MyDataset(MMAP_PATH)

    def produce(src, req):
        sl, bs = req
        return src.build_batch(sl, bs)   # returns the consumer's batch

    # --- train loop (MAIN process owns all CUDA) ---
    with GpuBatchPrefetcher(
        make_source=make_source,
        produce=produce,
        device="cuda:0",
        decode_workers=1,
        decode_ahead=2,
        gpu_ahead=1,
        start_method="spawn",
    ) as p:
        plan = iter(sl_bs_plan)          # the flat (sl, bs) request stream
        # prime: keep decode_ahead requests in flight
        p.submit(next(plan)); p.submit(next(plan))
        for _ in range(num_microbatches):
            batch = p.get()              # GPU-resident; event already waited
            train_microbatch(batch)      # consumer owns grad-accum / .step()
            try:
                p.submit(next(plan))     # refill the pipeline
            except StopIteration:
                pass

The consumer passes ONLY ``make_source`` + ``produce`` + the ``(sl, bs)``
requests. It never tells the prefetcher anything about the batch's
fields; ``get()`` already did ``current_stream().wait_event(event)``, so
``train_microbatch`` receives a ready GPU batch. The prefetcher serves
the flat request stream in order -- the grad-accum / ``.step()`` boundary
stays entirely on the consumer side.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
from collections.abc import Iterable, Iterator
from typing import Any, Callable, Optional

from ._cuda_backend import CudaBackend, TorchCudaBackend
from ._pytree import map_tensor_leaves
from ._spawn_payload import UnpicklablePayloadError, ensure_picklable
from ._to_device import default_leaf_pred, default_to_device, pin_host
from ._worker import DecodeWorkerError, decode_worker
from ._worker_monitor import PrefetchWorkerDied, WorkerMonitor

# torch is OPT-IN. The package is meant to be imported by torch-using
# consumers; the GPU path (copy stream / event) is a hard no-op without
# it. The import is SOFT so the orchestration stays importable -- and
# unit-testable via dependency injection (an injected ``to_device`` +
# ``is_leaf``) -- in a torch-less environment.
try:
    import torch
except ImportError:  # pragma: no cover - exercised only where torch absent
    torch = None  # type: ignore[assignment]

__all__ = [
    "GpuBatchPrefetcher",
    "PrefetchWorkerDied",
    "UnpicklablePayloadError",
    "default_to_device",
]


# --------------------------------------------------------------------------
# Prefetcher
# --------------------------------------------------------------------------
class GpuBatchPrefetcher:
    """Double-buffered, separate-process + separate-CUDA-stream prefetcher.

    See the module docstring for the full design + usage. Public surface:
    :meth:`submit`, :meth:`get`, :meth:`start`, :meth:`close`, and the
    context-manager protocol.
    """

    def __init__(
        self,
        *,
        make_source: Callable[[], Any],
        produce: Callable[[Any, Any], Any],
        device: Any,
        decode_workers: int = 1,
        decode_ahead: int = 2,
        gpu_ahead: int = 1,
        start_method: str = "spawn",
        c_fallback: bool = False,
        to_device: Optional[Callable[[Any, Any], Any]] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
        cuda_backend: Optional[CudaBackend] = None,
    ) -> None:
        if decode_workers < 1:
            raise ValueError("decode_workers must be >= 1")
        if decode_ahead < 1:
            raise ValueError("decode_ahead must be >= 1")
        if gpu_ahead < 1:
            raise ValueError("gpu_ahead must be >= 1")
        if gpu_ahead > decode_ahead:
            # The decode slot covers the FULL submit->get lifecycle, so at
            # most decode_ahead batches are ever in flight; a larger
            # gpu_ahead (ready-queue capacity) could never be reached and
            # only obscures the real bound.
            raise ValueError("gpu_ahead must be <= decode_ahead")

        # NOTE: make_source is NOT called here. The memmap opens in the
        # worker, post-spawn -- the parent never holds an open mmap.
        self._make_source = make_source
        self._produce = produce
        self._device = torch.device(device) if torch is not None else device
        self._decode_workers = decode_workers
        self._decode_ahead = decode_ahead
        self._gpu_ahead = gpu_ahead
        self._start_method = start_method
        self._c_fallback = c_fallback
        self._to_device = to_device or default_to_device
        self._is_leaf = is_leaf or default_leaf_pred
        # The ONE seam owning every torch.cuda primitive (copy stream,
        # stream-context async move, event record, consumer wait_event +
        # record_stream). Production default = real torch.cuda; tests inject
        # a fake that records the op order + stream identities -- so the
        # overlap is validatable WITHOUT a GPU.
        self._cuda = cuda_backend or TorchCudaBackend()

        self._ctx = mp.get_context(start_method)
        self._request_q: "mp.Queue" = self._ctx.Queue()
        self._result_q: "mp.Queue" = self._ctx.Queue()
        self._workers: list = []
        # Liveness oversight for the worker pool: lets the upload thread
        # fail fast (instead of spinning forever) when every worker has
        # exited while a submitted request is still owed.
        self._monitor: Optional[WorkerMonitor] = None

        # FIFO bookkeeping: submit assigns monotonically increasing seqs;
        # the upload thread reorders worker results into seq order before
        # uploading, so get() is strictly FIFO regardless of worker race.
        self._submit_seq = 0
        self._next_upload_seq = 0
        self._pending: dict[int, Any] = {}  # seq -> reordered CPU batch

        # Backpressure. ``_decode_slots`` bounds the batches in flight over
        # the FULL submit->get lifecycle: acquired in ``submit``, released
        # in ``get`` (NOT in the upload thread -- releasing there would let
        # the upload thread, blocked on a full ready queue, also starve slot
        # releases, deadlocking a producer that submits > gpu_ahead before
        # consuming). ``_ready_q`` (maxsize gpu_ahead) bounds uploads
        # launched ahead of consumption; ``get`` drains it, so the upload
        # thread's only wait is for the consumer to catch up. No cycle.
        self._decode_slots = threading.Semaphore(decode_ahead)
        self._ready_q: "queue.Queue" = queue.Queue(maxsize=gpu_ahead)

        self._copy_stream = None
        self._upload_thread: Optional[threading.Thread] = None
        self._closing = threading.Event()
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "GpuBatchPrefetcher":
        if self._started:
            return self
        for _ in range(self._decode_workers):
            w = self._ctx.Process(
                target=decode_worker,
                args=(self._make_source, self._produce, self._request_q, self._result_q),
                daemon=True,
            )
            w.start()
            self._workers.append(w)
        self._monitor = WorkerMonitor(self._workers)

        self._copy_stream = self._cuda.make_copy_stream(self._device)

        self._upload_thread = threading.Thread(target=self._upload_loop, daemon=True)
        self._upload_thread.start()
        self._started = True
        return self

    def close(self) -> None:
        if not self._started:
            return
        self._closing.set()
        for _ in self._workers:
            try:
                self._request_q.put(None)
            except Exception:
                pass
        for w in self._workers:
            w.join(timeout=5.0)
            if w.is_alive():
                w.terminate()
        if self._upload_thread is not None:
            self._upload_thread.join(timeout=5.0)
        self._started = False

    def __enter__(self) -> "GpuBatchPrefetcher":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- public producer/consumer API -------------------------------------
    def submit(self, request: Any) -> None:
        """Enqueue ``request`` for decode. Non-blocking up to backpressure.

        Blocks only if ``decode_ahead`` requests are already in flight,
        which is the intended bound on outstanding decode work.
        """
        if not self._started:
            raise RuntimeError("prefetcher not started")
        if self._start_method == "spawn":
            # spawn ships the request by pickle through a background feeder
            # thread that drops an unpicklable payload SILENTLY -> get()
            # would park forever. Reject it eagerly, here, before it can
            # consume a decode slot or enter the queue.
            ensure_picklable(request)
        self._decode_slots.acquire()
        seq = self._submit_seq
        self._submit_seq += 1
        self._request_q.put((seq, request))

    def get(self) -> Any:
        """Pop the oldest ready batch in FIFO order.

        Option P (default): the batch is GPU-resident and the copy-stream
        event has been waited on the CURRENT (compute) stream INSIDE this
        method, so the consumer receives a ready GPU batch. Option C
        (``c_fallback=True``): returns the pinned HOST batch and the
        consumer calls ``.to(cuda)`` itself.
        """
        if not self._started:
            raise RuntimeError("prefetcher not started")
        item = self._ready_q.get()
        # Consuming a batch frees one slot for the producer to submit into.
        self._decode_slots.release()
        # Both a worker's produce() error (DecodeWorkerError) and a dead-pool
        # verdict (PrefetchWorkerDied) ride the ready queue as exceptions.
        if isinstance(item, BaseException):
            raise item
        gpu_batch, event = item
        if event is not None:
            # Consumer thread: make the compute stream wait on the copy
            # event AND register the uploaded leaves' cross-stream use.
            self._cuda.wait(gpu_batch, event, self._device, self._is_leaf)
        return gpu_batch

    def stream(self, requests: "Iterable[Any]") -> "Iterator[Any]":
        """Yield decoded batches for ``requests``, back-pressure-safe.

        The low-level :meth:`submit`/:meth:`get` pair lets a caller
        over-submit past ``decode_ahead`` -- and :meth:`submit` BLOCKS once
        ``decode_ahead`` requests are in flight, so a caller that submits the
        whole request stream before consuming wedges on its own back-pressure
        (an easy footgun). This helper removes it: it primes ``decode_ahead``
        requests, then yields each batch in FIFO order as it becomes ready,
        submitting the NEXT request only as a slot frees. At most
        ``decode_ahead`` requests are ever in flight, so neither
        :meth:`submit` nor the caller can deadlock. Usage::

            for batch in prefetcher.stream(requests):
                train_step(batch)

        ``requests`` is consumed lazily, so the request stream never has to be
        materialised; the refill happens BEFORE the yield so the worker
        decodes the next datum while the caller processes this one (the
        overlap that is the whole point).
        """
        it = iter(requests)
        inflight = 0
        exhausted = False
        while inflight < self._decode_ahead and not exhausted:
            try:
                self.submit(next(it))
            except StopIteration:
                exhausted = True
            else:
                inflight += 1
        while inflight > 0:
            batch = self.get()
            inflight -= 1
            if not exhausted:
                try:
                    self.submit(next(it))
                except StopIteration:
                    exhausted = True
                else:
                    inflight += 1
            yield batch

    # -- main-process upload thread ---------------------------------------
    def _upload_loop(self) -> None:
        """Pull CPU batches, reorder to FIFO, upload on the copy stream.

        Mirrors torch's pin_memory thread: light work only (the generic
        tensor move + an event record). Worker results may arrive out of
        order across multiple workers, so they are buffered by seq and
        uploaded strictly in submit order.
        """
        while not self._closing.is_set():
            try:
                seq, payload = self._result_q.get(timeout=0.2)
            except queue.Empty:
                # Drained the result queue and it came up empty: if a worker
                # has crashed (see WorkerMonitor -- a clean exit only happens
                # at close(), which ends this loop via _closing), its owed seq
                # can never arrive, so fail fast instead of spinning forever.
                if self._monitor is not None and self._monitor.crashed():
                    self._surface_dead_workers()
                continue

            kind, value = payload
            if kind == "err":
                # ``value`` is a WorkerExcInfo carrying the worker-side
                # formatted traceback; surface it in the re-raised error
                # so the internal frame isn't lost across the spawn.
                self._pending[seq] = DecodeWorkerError(value.as_message())
            else:
                self._pending[seq] = value

            self._drain_contiguous()

    def _drain_contiguous(self) -> None:
        """Upload buffered results as far as contiguous seqs allow (FIFO).

        The slot is released on get(), not here, so this thread blocking on a
        full ready queue never starves the producer's slots.
        """
        while self._next_upload_seq in self._pending:
            cpu_batch = self._pending.pop(self._next_upload_seq)
            self._next_upload_seq += 1
            if isinstance(cpu_batch, DecodeWorkerError):
                self._ready_q.put(cpu_batch)
                continue
            self._ready_q.put(self._upload_one(cpu_batch))

    def _surface_dead_workers(self) -> None:
        """Resolve EVERY owed seq through ``_ready_q`` when all workers died.

        Walks the owed seqs in FIFO order: a seq the (now-dead) workers DID
        produce is uploaded normally; a seq they never delivered can no
        longer arrive, so a :class:`PrefetchWorkerDied` is enqueued in its
        place. Either way ``get()`` advances instead of parking. No-op when
        nothing is owed (an idle, possibly torn-down pool). The bounded put
        blocks until the consumer drains; ``_closing`` breaks it on teardown.
        """
        cause = self._monitor.death_cause()
        # Resolve every submitted-but-undelivered seq, in order, up to the
        # high-water mark of submissions. ``_submit_seq`` is read afresh each
        # iteration so requests submitted AFTER the death are resolved too.
        while not self._closing.is_set() and self._next_upload_seq < self._submit_seq:
            # Out-of-order results that DID arrive before the death belong
            # ahead of any gap -- upload them first.
            self._drain_contiguous()
            if self._next_upload_seq >= self._submit_seq:
                break
            # The seq now at the head was never produced (no live worker can
            # ever deliver it); fill its FIFO slot with the dead-pool error.
            self._next_upload_seq += 1
            self._ready_q.put(
                PrefetchWorkerDied(
                    "decode worker(s) exited before delivering a submitted "
                    f"batch: {cause}"
                )
            )

    def _upload_one(self, cpu_batch: Any) -> Any:
        """Move every tensor leaf to device on the copy stream; record event.

        Option C short-circuits: pin the host batch (still generic) and
        return ``(host_batch, None)`` so ``get`` skips the event-wait and
        the consumer owns the ``.to(cuda)``.
        """
        if self._c_fallback:
            host = map_tensor_leaves(cpu_batch, pin_host, is_leaf=self._is_leaf)
            return (host, None)

        if self._copy_stream is None:
            # No CUDA (CPU device / no GPU / torch-less test path): just
            # apply the move fn (a no-op / plain .to) -- no overlap.
            moved = map_tensor_leaves(
                cpu_batch,
                lambda t: self._to_device(t, self._device),
                is_leaf=self._is_leaf,
            )
            return (moved, None)

        return self._cuda.upload(
            cpu_batch,
            self._device,
            self._copy_stream,
            lambda t: self._to_device(t, self._device),
            self._is_leaf,
        )
