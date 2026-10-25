"""Reusable bounded asynchronous preparation pool.

The Rust implementation owns worker scheduling, bounded queues, cancellation,
failure propagation, and the GIL boundary. The Python facade maps arbitrary
caller keys to opaque Rust registration IDs. A key's producer is either a
Python callable (a compatibility adapter for existing no-argument data
sources) or a :class:`RustLoader`: a loader written in Rust against the
crate's loader interface, whose items are dicts of read-only numpy arrays.

Where a Rust loader runs is the backend, the same API either way:

- ``"local"``: the loader binary as a child process of this one, reading the
  data roots given here (``programs`` and ``roots``).
- ``"bridge"``: a loader-bridge server, such as a container's host, through
  its socket (``socket``, else ``$LOADER_BRIDGE_SOCKET``, else
  ``/run/loader-bridge/sock``). The server decides which loaders exist and
  which data roots they read.

``backend=`` picks it; left out, ``$ASYNC_READY_POOL_BACKEND`` does, else
``"local"``.

A loader's key is kept ready (``ready_depth`` items made ahead, each with the
loader's default request) or, with ``on_request=True``, filled on request:
:meth:`AsyncReadyPool.request` asks for each item with its own parameters
and :meth:`AsyncReadyPool.get` returns them in that order. A loader that can
be resumed puts ``resume_state`` in every item, the state after it; a
:class:`RustLoader` with ``resume=`` those bytes goes on with the next item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping

from ._native import DEFAULT_SOCKET, RustPreloader, SessionEnded

Producer = Callable[[], Any]


@dataclass(frozen=True)
class RustLoader:
    """A Rust loader as a key's producer: the loader's name, its typed
    parameters, and its inputs as ``"<root>"`` or ``"<root>/<path>"``."""

    loader: str
    params: Mapping[str, Any] = field(default_factory=dict)
    files: tuple[str, ...] = ()
    resume: bytes | None = None
    on_request: bool = False


@dataclass(frozen=True)
class PoolConfig:
    """One caller-owned producer and its bounded ready depth."""

    key: Hashable
    producer: Producer | RustLoader
    ready_depth: int


class AsyncReadyPool:
    """Keyed facade over :class:`RustPreloader`.

    ``configs`` contains :class:`PoolConfig` values. Python producers are
    called on Rust worker threads; the Rust adapter acquires the GIL only for
    the callback and releases it while waiting or joining. Each Rust loader
    named in ``configs`` gets one session with ``workers`` producer threads,
    shared by its keys.

    For ``backend="local"``: ``programs`` maps a loader name to its binary,
    ``roots`` maps data root names to directories, ``max_shared_bytes`` caps
    each session's shared memory.
    """

    def __init__(
        self,
        configs: Iterable[PoolConfig],
        workers: int = 1,
        *,
        backend: str | None = None,
        socket: str | None = None,
        programs: Mapping[str, str] | None = None,
        roots: Mapping[str, str] | None = None,
        max_shared_bytes: int | None = None,
    ) -> None:
        config_list = list(configs)
        if not config_list:
            raise ValueError("at least one pool config is required")
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self._keys: dict[Hashable, tuple[RustPreloader, int]] = {}
        self._impls: list[RustPreloader] = []
        by_id: dict[int, Producer] = {}
        callbacks: RustPreloader | None = None
        sessions: dict[str, RustPreloader] = {}

        def dispatch(registration: int) -> Any:
            return by_id[registration]()

        try:
            for config in config_list:
                if config.key in self._keys:
                    raise ValueError(f"duplicate pool key: {config.key!r}")
                producer = config.producer
                if isinstance(producer, RustLoader):
                    impl = sessions.get(producer.loader)
                    if impl is None:
                        local = {}
                        if (backend or _env_backend()) == "local":
                            program = (programs or {}).get(producer.loader)
                            if program is None:
                                raise ValueError(
                                    f"backend 'local' needs programs[{producer.loader!r}]"
                                )
                            local = {
                                "program": program,
                                "roots": list((roots or {}).items()),
                                "max_shared_bytes": max_shared_bytes,
                            }
                        impl = RustPreloader(
                            None,
                            workers,
                            loader=producer.loader,
                            backend=backend,
                            socket=socket,
                            **local,
                        )
                        sessions[producer.loader] = impl
                        self._impls.append(impl)
                    registration = impl.register(
                        config.ready_depth,
                        dict(producer.params),
                        list(producer.files),
                        resume=producer.resume,
                        on_request=producer.on_request,
                    )
                else:
                    if callbacks is None:
                        callbacks = RustPreloader(dispatch, workers)
                        self._impls.append(callbacks)
                    impl = callbacks
                    registration = impl.register(config.ready_depth)
                    by_id[registration] = producer
                self._keys[config.key] = (impl, registration)
        except BaseException:
            self.close()
            raise

    def start(self) -> "AsyncReadyPool":
        for impl in self._impls:
            impl.start()
        return self

    def get(self, key: Hashable) -> Any:
        try:
            impl, registration = self._keys[key]
        except KeyError as exc:
            raise KeyError(f"unknown pool key: {key!r}") from exc
        return impl.get(registration)

    def request(self, key: Hashable, params: Mapping[str, Any] | None = None) -> None:
        """Asks a key filled on request for its next item."""
        try:
            impl, registration = self._keys[key]
        except KeyError as exc:
            raise KeyError(f"unknown pool key: {key!r}") from exc
        impl.request(registration, dict(params) if params is not None else None)

    def ready_counts(self) -> dict[Hashable, int]:
        counts = {id(impl): dict(impl.ready_counts()) for impl in self._impls}
        return {
            key: counts[id(impl)].get(registration, 0)
            for key, (impl, registration) in self._keys.items()
        }

    def close(self) -> None:
        for impl in getattr(self, "_impls", []):
            impl.close()

    def __enter__(self) -> "AsyncReadyPool":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _env_backend() -> str:
    import os

    return os.environ.get("ASYNC_READY_POOL_BACKEND") or "local"


__all__ = [
    "AsyncReadyPool",
    "DEFAULT_SOCKET",
    "PoolConfig",
    "Producer",
    "RustLoader",
    "RustPreloader",
    "SessionEnded",
]
