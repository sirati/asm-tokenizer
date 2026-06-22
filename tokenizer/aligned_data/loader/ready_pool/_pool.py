"""Layer 1: the in-process, MULTITHREADED keep-N-ready batch pool.

Single concern: keep, for each registered :class:`PoolConfig`, up to
``ready_depth`` decoded batches buffered by background THREADS that
continuously call the config's opaque ``produce()`` -- so a consumer's
:meth:`ReadyPool.get` hands back an already-decoded batch (FIFO per
config) instead of paying the decode synchronously on the train thread.

Boundary contract (the design-first sentence):

  *Given a set of registered configs, each carrying a no-arg
  ``produce() -> batch`` closure (the decode + its bound sampler) and a
  target ready-depth, serve those batches back per config in FIFO order,
  each already decoded, WITHOUT this module knowing a single thing about
  how a batch is decoded, sampled, or shaped.*

WHY THREADS, NOT PROCESSES: the heavy decode (``vector_batch_tokens``)
releases the GIL inside its Rust/numpy kernels, so worker THREADS overlap
real CPU work without the pickling / spawn / IPC-queue tax a process pool
pays. This module therefore uses ONLY ``threading`` + ``queue`` -- no
``multiprocessing`` anywhere -- and the opaque ``produce`` may be any
in-process callable / closure (no picklability requirement).

BACKPRESSURE + keep-N-ready, in ONE primitive: each config owns a
``queue.Queue(maxsize=ready_depth)``. A worker loops ``produce() ->
ready_q.put(batch)``; the bounded ``put`` BLOCKS once ``ready_depth``
batches are buffered, so a worker never decodes beyond the target depth
(the queue depth IS the keep-N invariant). :meth:`get` pops the oldest
(FIFO), freeing a slot the worker immediately refills.

WORKER-DEATH SURFACING (mirrors ``GpuBatchPrefetcher`` / the loaded-box
deadlock fix): a worker thread that dies mid-``produce`` (an exception
escapes the loop) can no longer refill its queue, so a consumer blocked on
:meth:`get` would park forever. The death is surfaced two ways: the
escaped exception is enqueued on the very queue the consumer drains (so
the NEXT :meth:`get` re-raises it in FIFO position), AND a
:class:`WorkerMonitor` over the threads lets :meth:`get` poll-and-check
crashed-pool liveness so even a config whose every thread died silently
raises :class:`ReadyPoolWorkerDied` instead of hanging.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Dict, Hashable, List, Sequence

from tokenizer.aligned_data.loader.gpu_prefetcher._worker_monitor import (
    WorkerMonitor,
)

from ._config import PoolConfig


__all__ = ["ReadyPool", "ReadyPoolWorkerDied"]


class ReadyPoolWorkerDied(RuntimeError):
    """A config's refill thread(s) died while a batch was still owed.

    Raised in :meth:`ReadyPool.get` (instead of blocking forever) when the
    threads feeding a config's ready buffer have all exited without the
    requested batch ever arriving -- e.g. a ``produce`` that raised an
    exception the loop could not recover from.
    """


def _describe_thread(t: threading.Thread) -> str:
    """Cause formatter for a dead refill thread (no pid / exitcode)."""
    return f"{t.name}(alive={t.is_alive()})"


# How often :meth:`get` wakes to re-check pool liveness while blocked on an
# empty ready queue. Small enough that a silent all-threads-dead config
# surfaces promptly; large enough not to busy-spin.
_LIVENESS_POLL_SECS = 0.1


class ReadyPool:
    """In-process, threaded, multi-config keep-N-ready batch pool.

    See the module docstring for the design. Public surface:
    :meth:`start`, :meth:`get`, :meth:`close`, and the context-manager
    protocol. Each registered :class:`PoolConfig` gets its own FIFO ready
    buffer kept filled to ``ready_depth`` by ``threads_per_config`` daemon
    worker threads.
    """

    def __init__(
        self,
        configs: Sequence[PoolConfig],
        *,
        threads_per_config: int = 1,
    ) -> None:
        if threads_per_config < 1:
            raise ValueError("threads_per_config must be >= 1")
        if not configs:
            raise ValueError("at least one PoolConfig is required")
        keys = [c.key for c in configs]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate config keys: {keys}")

        self._configs: Dict[Hashable, PoolConfig] = {c.key: c for c in configs}
        self._threads_per_config = threads_per_config

        # Per-config FIFO ready buffer; maxsize=ready_depth IS the keep-N
        # invariant + the backpressure bound (a worker's bounded put blocks
        # once the buffer holds ready_depth batches, so it never decodes
        # past the target depth).
        self._ready: Dict[Hashable, "queue.Queue"] = {
            c.key: queue.Queue(maxsize=c.ready_depth) for c in configs
        }
        self._threads: Dict[Hashable, List[threading.Thread]] = {}
        # One liveness monitor per config: get(key) asks ONLY that config's
        # monitor whether its feeding threads have died, so a healthy config
        # never trips on an unrelated config's crash.
        self._monitors: Dict[Hashable, WorkerMonitor] = {}

        self._closing = threading.Event()
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "ReadyPool":
        if self._started:
            return self
        for key, cfg in self._configs.items():
            workers: List[threading.Thread] = []
            for i in range(self._threads_per_config):
                t = threading.Thread(
                    target=self._refill_loop,
                    args=(cfg,),
                    name=f"ready-pool[{key!r}]#{i}",
                    daemon=True,
                )
                t.start()
                workers.append(t)
            self._threads[key] = workers
            self._monitors[key] = WorkerMonitor(workers, describe=_describe_thread)
        self._started = True
        return self

    def close(self) -> None:
        if not self._started:
            return
        self._closing.set()
        # Unblock any worker parked on a full ready queue so it can observe
        # the closing flag and exit. A best-effort non-blocking drain frees
        # a slot per config; the workers are daemons, so a residual park is
        # reaped at interpreter exit regardless.
        for q in self._ready.values():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        for workers in self._threads.values():
            for t in workers:
                t.join(timeout=5.0)
        self._started = False

    def __enter__(self) -> "ReadyPool":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- consumer API ------------------------------------------------------
    def get(self, key: Hashable) -> Any:
        """Pop the oldest ready batch for ``key`` (FIFO); refill a slot.

        Blocks until a batch is ready. A worker exception that escaped the
        refill loop rides this queue and is re-raised here in FIFO
        position; if every thread feeding ``key`` has died silently,
        :class:`ReadyPoolWorkerDied` is raised (never an infinite park).
        """
        if not self._started:
            raise RuntimeError("ReadyPool not started")
        if key not in self._ready:
            raise KeyError(f"no registered config with key {key!r}")
        ready_q = self._ready[key]
        monitor = self._monitors[key]
        while True:
            try:
                item = ready_q.get(timeout=_LIVENESS_POLL_SECS)
            except queue.Empty:
                # Nothing buffered: if every feeding thread has exited (a
                # clean exit only happens at close(), gated by _closing),
                # the owed batch can never arrive -> fail fast.
                if not self._closing.is_set() and monitor.crashed():
                    raise ReadyPoolWorkerDied(
                        "ready-pool refill thread(s) exited before delivering "
                        f"a batch for config {key!r}: {monitor.death_cause()}"
                    )
                continue
            if isinstance(item, BaseException):
                raise item
            return item

    # -- background refill thread -----------------------------------------
    def _refill_loop(self, cfg: PoolConfig) -> None:
        """Continuously decode + buffer up to ``ready_depth`` for ``cfg``.

        The bounded ``put`` is the backpressure: it blocks once the ready
        buffer is full, so the loop decodes EXACTLY to the target depth and
        no further. A ``produce`` exception is enqueued (so the consumer
        re-raises it in FIFO position) and the loop exits -- the monitor
        then surfaces the dead thread to any later :meth:`get`.
        """
        ready_q = self._ready[cfg.key]
        while not self._closing.is_set():
            try:
                batch = cfg.produce()
            except BaseException as exc:  # noqa: BLE001 - surface to consumer
                self._offer(ready_q, exc)
                return
            self._offer(ready_q, batch)

    def _offer(self, ready_q: "queue.Queue", item: Any) -> None:
        """Put ``item`` on the bounded ready queue, honouring close.

        The bounded put with a short timeout is the keep-N-ready
        backpressure AND a clean shutdown seam: while the buffer is full
        the worker parks here, periodically re-checking ``_closing`` so it
        exits promptly on :meth:`close` instead of blocking forever on a
        consumer that has stopped draining.
        """
        while not self._closing.is_set():
            try:
                ready_q.put(item, timeout=_LIVENESS_POLL_SECS)
                return
            except queue.Full:
                continue
