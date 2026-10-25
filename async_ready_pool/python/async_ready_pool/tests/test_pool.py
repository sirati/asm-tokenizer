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


def test_finite_plan_does_not_overproduce_or_poison_tail():
    draws = iter(range(5))
    with AsyncReadyPool([PoolConfig("plan", lambda: next(draws), 2, count=5)]) as pool:
        assert [pool.get("plan") for _ in range(5)] == list(range(5))
        with pytest.raises(StopIteration):
            pool.get("plan")


def test_empty_plan_is_exhausted():
    def unexpected():
        raise AssertionError("empty plan called producer")
    with AsyncReadyPool([PoolConfig("empty", unexpected, 2, count=0)]) as pool:
        with pytest.raises(StopIteration):
            pool.get("empty")


def test_close_wakes_waiting_consumer():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered = threading.Event()
    release = threading.Event()
    def produce():
        entered.set()
        assert release.wait(3)
        return 1
    pool = AsyncReadyPool([PoolConfig("plan", produce, 1)]).start()
    assert entered.wait(3)
    with ThreadPoolExecutor(2) as executor:
        get_result = executor.submit(pool.get, "plan")
        close_result = executor.submit(pool.close)
        release.set()
        close_result.result(3)
        # A result that completed before close may still be delivered.
        try:
            assert get_result.result(3) == 1
        except RuntimeError as error:
            assert "clos" in str(error)


def test_abandoned_pool_does_not_deadlock_under_gil():
    import subprocess
    import sys
    code = '''
import time, gc
from async_ready_pool import AsyncReadyPool, PoolConfig
pool = AsyncReadyPool([PoolConfig("a", lambda: time.sleep(.02), 1)]).start()
del pool
gc.collect()
print("collected")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert "collected" in result.stdout
