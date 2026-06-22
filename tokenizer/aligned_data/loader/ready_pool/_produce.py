"""The typed ``produce`` contract the pool fills from -- and its OPTIONAL close.

Single concern: name, as a typed :class:`typing.Protocol`, the shape of the
opaque thing a :class:`PoolConfig` registers and the pool drives. Two shapes,
one OPTIONAL over the other:

  :data:`Produce`        -- the minimum: a no-arg callable returning one batch.
  :class:`CloseableProduce` -- the SAME callable that ALSO exposes a no-arg
                            ``close()`` releasing whatever per-worker resources
                            that worker's ``produce()`` calls allocated.

WHY a protocol, not a base class: the pool must stay ignorant of decode /
CUDA / handle internals (see :mod:`._pool` boundary contract). It only needs
to know "a produce MAY be closeable". A :func:`runtime_checkable` Protocol
lets the refill loop ask that question structurally -- ``is_closeable_produce``
below -- with NO ``isinstance`` ladder over concrete decode types and NO
import of the decode module. Mirrors the repo's explicit-close lifecycle
discipline (see ``DisassemblyProvider.close`` / ``VectorBatchArmSet.close``):
a worker that opened per-thread resources releases them in a ``finally`` when
it stops, instead of leaning on thread/process exit.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


__all__ = ["Produce", "CloseableProduce", "is_closeable_produce"]


@runtime_checkable
class Produce(Protocol):
    """The minimum a :class:`PoolConfig` registers: a no-arg batch source."""

    def __call__(self) -> Any: ...


@runtime_checkable
class CloseableProduce(Produce, Protocol):
    """A :data:`Produce` that ALSO releases its per-worker resources.

    The refill thread that drove this produce calls :meth:`close` exactly
    once, in a ``finally``, when it stops -- so a produce holding
    THREAD-LOCAL handles/sessions (each worker thread opened its own) frees
    the calling thread's resources deterministically rather than waiting on
    thread/process teardown.
    """

    def close(self) -> None: ...


def is_closeable_produce(produce: Produce) -> bool:
    """Does ``produce`` honour the OPTIONAL :class:`CloseableProduce` seam?

    A structural (duck-typed) check via the runtime-checkable protocol -- so
    the pool decides "call close() on shutdown?" WITHOUT knowing any concrete
    decode type and WITHOUT an ``isinstance`` ladder over them.
    """
    return isinstance(produce, CloseableProduce)
