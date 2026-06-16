"""Spawn-boundary payload contract.

Single concern: a value handed to a ``spawn`` worker must survive the
pickle IPC crossing. ``multiprocessing.Queue`` pickles in a BACKGROUND
feeder thread, so a payload whose pickling raises is dropped SILENTLY --
the worker stays alive but idle and the consumer's ``get()`` parks
forever. This module turns that silent drop into an explicit, eager error
raised in the SUBMITTING thread, at the exact call that supplied the bad
payload.

Boundary contract: the caller submits a request; this guard validates it
crosses the spawn boundary BEFORE it enters the queue. Requests are tiny
opaque tokens (e.g. an ``(sl, bs)`` tuple), so the pickle probe is
negligible next to the decode it gates.
"""

from __future__ import annotations

import pickle
from typing import Any


class UnpicklablePayloadError(TypeError):
    """A submitted request cannot cross the ``spawn`` pickle boundary.

    Raised eagerly in :meth:`GpuBatchPrefetcher.submit` instead of letting
    the queue feeder thread drop it silently (which would park ``get()``).
    """


def ensure_picklable(obj: Any) -> None:
    """Raise :class:`UnpicklablePayloadError` if ``obj`` will not pickle.

    The probe mirrors exactly what ``multiprocessing.Queue``'s feeder does
    (``pickle.dumps``), so anything that passes here crosses the spawn
    boundary cleanly; anything that fails surfaces here instead of silently
    stalling the worker.
    """
    try:
        pickle.dumps(obj)
    except Exception as exc:  # pickling raises a zoo of types; normalise here
        raise UnpicklablePayloadError(
            f"request is not picklable across the spawn boundary: {exc!r}"
        ) from exc
