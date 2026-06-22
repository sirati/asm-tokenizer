"""The one registration shape the ready pool keeps buffered.

Single concern: bundle, for ONE registered config, the OPAQUE no-arg
``produce`` callable (draws a section-pointer batch + decodes it -- the
sampler is bound inside the closure) and the target ready-depth (how many
decoded batches to keep buffered), keyed for retrieval. This module knows
NOTHING about decode internals or CUDA -- it is the clean seam the pool
fills from. The decode wiring lives in :mod:`._vector_batch_source`; the
pool (:mod:`._pool`) only ever calls ``produce()`` and honours
``ready_depth``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable


__all__ = ["PoolConfig"]


@dataclass(frozen=True)
class PoolConfig:
    """One registered (produce, keep-N-ready) config, keyed.

    ``key`` identifies the config at :meth:`ReadyPool.get` time; it is any
    hashable (a string label, a ``(batch_size, seq_len)`` tuple, an enum
    member -- the pool never interprets it). ``produce`` is the OPAQUE
    no-arg decode seam: each call draws one section-pointer batch from the
    config's bound sampler and returns one decoded batch (e.g. a
    :class:`VectorBatchResult`). ``ready_depth`` is how many decoded
    batches the pool keeps buffered for this config -- the keep-N-ready
    depth AND the backpressure bound (a worker never decodes past it).
    """

    key: Hashable
    produce: Callable[[], Any]
    ready_depth: int

    def __post_init__(self) -> None:
        if self.ready_depth < 1:
            raise ValueError(
                f"ready_depth must be >= 1, got {self.ready_depth} for "
                f"config {self.key!r}"
            )
