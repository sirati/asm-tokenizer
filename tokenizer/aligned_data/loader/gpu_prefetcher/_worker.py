"""Pure-CPU decode worker process body (no torch, no CUDA).

Single concern: in a SEPARATE process, open the consumer's source ONCE
(``make_source`` -- the memmap opens here, never in the parent, so a fork
never inherits an open mmap) and then serve decode requests forever,
shipping each produced CPU batch back over an IPC queue in
``(seq, payload)`` form so the main process can reorder to FIFO. CUDA is
per-process, so this side stays purely CPU: it returns a CPU/pinned
batch; all GPU work lives in the main process.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable


class DecodeWorkerError(RuntimeError):
    """A worker's ``produce`` raised; re-raised in the main process."""


def decode_worker(
    make_source: Callable[[], Any],
    produce: Callable[[Any, Any], Any],
    request_q: "mp.Queue",
    result_q: "mp.Queue",
) -> None:
    """Worker body: open the source ONCE, then decode requests forever.

    Each result is a ``(seq, payload)`` pair where ``seq`` is the FIFO
    sequence number the request carried and ``payload`` is either
    ``("ok", batch)`` or ``("err", repr)``. A poison ``None`` request ends
    the loop. A ``produce`` exception is shipped as an ``"err"`` payload
    (never swallowed, never a hang) so the main process can re-raise it
    in FIFO position.
    """
    source = make_source()
    while True:
        item = request_q.get()
        if item is None:
            return
        seq, request = item
        try:
            batch = produce(source, request)
            result_q.put((seq, ("ok", batch)))
        except BaseException as exc:  # surface, don't hang the consumer
            result_q.put((seq, ("err", repr(exc))))
