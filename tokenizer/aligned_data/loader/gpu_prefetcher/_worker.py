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
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class WorkerExcInfo:
    """Cross-process snapshot of a worker exception.

    The worker runs in a SEPARATE process, so the live exception object
    (and its traceback) cannot cross the IPC queue intact. This captures
    everything the main process needs to re-raise an informative error:
    the original exception type name, its ``repr``, and the formatted
    traceback string from the worker side.
    """

    type_name: str
    exc_repr: str
    formatted_tb: str

    def as_message(self) -> str:
        """One-shot message for the re-raised main-process error."""
        return (
            f"{self.type_name}: {self.exc_repr}\n"
            f"--- worker traceback ---\n{self.formatted_tb}"
        )


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
    ``("ok", batch)`` or ``("err", WorkerExcInfo)``. A poison ``None``
    request ends the loop. A ``produce`` exception is shipped as an
    ``"err"`` payload carrying the formatted worker-side traceback (never
    swallowed, never a hang) so the main process can re-raise it in FIFO
    position WITH the internal traceback that died with the worker frame.
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
            exc_info = WorkerExcInfo(
                type_name=type(exc).__name__,
                exc_repr=repr(exc),
                formatted_tb=traceback.format_exc(),
            )
            result_q.put((seq, ("err", exc_info)))
