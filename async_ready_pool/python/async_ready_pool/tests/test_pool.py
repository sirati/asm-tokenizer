from __future__ import annotations

import time

import pytest

from async_ready_pool import AsyncReadyPool, PoolConfig


def test_fills_and_returns_each_registration() -> None:
    values = iter(range(20))
    with AsyncReadyPool(
        [PoolConfig("numbers", lambda: next(values), 2)], workers=2
    ) as pool:
        deadline = time.monotonic() + 2
        while pool.ready_counts()["numbers"] < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert pool.ready_counts()["numbers"] == 2
        assert {pool.get("numbers") for _ in range(4)} == {0, 1, 2, 3}


def test_failure_is_scoped_to_one_registration() -> None:
    def broken() -> object:
        raise RuntimeError("source broken")

    with AsyncReadyPool(
        [
            PoolConfig("broken", broken, 1),
            PoolConfig("healthy", lambda: "ok", 1),
        ],
        workers=2,
    ) as pool:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                pool.get("broken")
            except RuntimeError as error:
                assert "source broken" in str(error)
                break
        else:
            pytest.fail("producer failure was not surfaced")
        assert pool.get("healthy") == "ok"


def test_loader_backends_are_checked(tmp_path) -> None:
    from async_ready_pool import RustLoader, RustPreloader, SessionEnded

    with pytest.raises(ValueError, match="needs program"):
        RustPreloader(None, 1, loader="tokens", backend="local")
    with pytest.raises(ValueError, match="unknown backend"):
        RustPreloader(None, 1, loader="tokens", backend="elsewhere")
    with pytest.raises(ValueError, match='for backend="local"'):
        RustPreloader(None, 1, loader="tokens", backend="bridge", program="/bin/true")
    with pytest.raises(SessionEnded):
        RustPreloader(None, 1, loader="tokens", backend="bridge", socket=str(tmp_path / "none"))
    with pytest.raises(ValueError, match="programs"):
        AsyncReadyPool([PoolConfig("k", RustLoader("tokens"), 1)], backend="local")
    with pytest.raises(ValueError, match="either a producer or a loader"):
        RustPreloader(lambda _id: 1, 1, loader="tokens")
    with pytest.raises(ValueError, match="for loaders"):
        RustPreloader(lambda _id: 1, 1).register(1, {"x": 1})
