"""Reusable bounded asynchronous preparation pool.

The Rust implementation owns worker scheduling, bounded queues, cancellation,
failure propagation, and the GIL boundary. The Python facade maps arbitrary
caller keys to opaque Rust registration IDs; its producer callable is only a
compatibility adapter for existing no-argument data sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable

from ._native import RustPreloader

Producer = Callable[[], Any]


@dataclass(frozen=True)
class PoolConfig:
    """One caller-owned producer and its bounded ready depth."""

    key: Hashable
    producer: Producer
    ready_depth: int


class AsyncReadyPool:
    """Keyed facade over :class:`RustPreloader`.

    ``configs`` contains :class:`PoolConfig` values. Producers are called on
    Rust worker threads; the Rust adapter acquires the GIL only for the
    callback and releases it while waiting or joining.
    """

    def __init__(self, configs: Iterable[PoolConfig], workers: int = 1) -> None:
        config_list = list(configs)
        if not config_list:
            raise ValueError("at least one pool config is required")
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self._ids: dict[Hashable, int] = {}
        by_id: dict[int, Producer] = {}

        def dispatch(registration: int) -> Any:
            return by_id[registration]()

        self._impl = RustPreloader(dispatch, workers)
        for config in config_list:
            if config.key in self._ids:
                raise ValueError(f"duplicate pool key: {config.key!r}")
            registration = self._impl.register(config.ready_depth)
            self._ids[config.key] = registration
            by_id[registration] = config.producer

    def start(self) -> "AsyncReadyPool":
        self._impl.start()
        return self

    def get(self, key: Hashable) -> Any:
        try:
            registration = self._ids[key]
        except KeyError as exc:
            raise KeyError(f"unknown pool key: {key!r}") from exc
        return self._impl.get(registration)

    def ready_counts(self) -> dict[Hashable, int]:
        counts = dict(self._impl.ready_counts())
        return {key: counts[registration] for key, registration in self._ids.items()}

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "AsyncReadyPool":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = ["AsyncReadyPool", "PoolConfig", "Producer", "RustPreloader"]
