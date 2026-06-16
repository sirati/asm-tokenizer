"""Liveness oversight for the decode worker pool.

Single concern: answer ONE question for the upload thread -- "has a worker
crashed?" -- without that thread knowing anything about
``multiprocessing.Process``. A decode worker only EVER exits cleanly on
the poison ``None`` request during ``close()``; an exit at any other time
is a crash (a post-``spawn`` re-import failure, a segfault, an OOM kill).
A crashed worker may have been mid-``produce`` on a seq that no surviving
peer will ever re-attempt -- the shared request queue assigns each request
to exactly one worker -- so that seq is permanently owed and would park
``get()`` forever. Treating ANY unexpected death as fatal (torch's
DataLoader does the same) closes that hole without per-seq worker
bookkeeping.

We never respawn workers, so "a worker has crashed" is monotonic: once a
non-closing process exits it stays exited. Combined with a freshly drained
result queue, that is the race-free trigger to fail the pipeline fast.

Boundary contract: the upload thread owns the FIFO/seq accounting (it
alone knows how many results are still owed); this module owns only the
process liveness verdict + a human-readable cause for the error. The two
compose in the upload loop: *drain the result queue, then ask whether a
worker has crashed*; if so, every still-owed seq is surfaced as a raise
through the ready queue instead of spinning forever.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Sequence


class PrefetchWorkerDied(RuntimeError):
    """A decode worker crashed while a submitted request was still owed.

    Raised in :meth:`GpuBatchPrefetcher.get` (in FIFO position) instead of
    blocking forever when the producer chain is compromised -- e.g. a worker
    that crashed on its post-``spawn`` re-import, segfaulted, or was
    OOM-killed without ever shipping a result.
    """


class WorkerMonitor:
    """Tracks a fixed decode-worker pool and flags unexpected deaths.

    The pool is fixed at construction (no respawn), so :meth:`crashed`
    transitions False -> True at most once and never back. That monotonic
    edge is the safe trigger: combined with a freshly drained result queue,
    it proves the owed result will never arrive.
    """

    def __init__(self, workers: Sequence["mp.Process"]) -> None:
        self._workers = list(workers)

    def crashed(self) -> bool:
        """True once ANY worker has exited (a clean exit only happens at
        ``close()``, which the caller gates out via its closing flag).

        A worker that is slow-but-alive keeps this False, so a healthy (if
        sluggish) pipeline never trips the fail-fast path.
        """
        return any(not w.is_alive() for w in self._workers)

    def death_cause(self) -> str:
        """Human-readable summary of the dead workers and their exit codes."""
        parts = [
            f"{w.name}(pid={w.pid}, exitcode={w.exitcode})"
            for w in self._workers
            if not w.is_alive()
        ]
        return ", ".join(parts) if parts else "no worker has exited"
