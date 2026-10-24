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
