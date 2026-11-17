"""Cancellation is latched before startup and interrupts an actual empty-queue wait."""
import queue
import threading

import pytest

from tokenizer.aligned_data.loader.ready_pool import PoolConfig, ReadyPool


def test_cancel_before_start_is_latched():
    def produce():
        raise AssertionError("a cancelled pool must not start a producer")
    pool = ReadyPool([PoolConfig(key="batch", produce=produce, ready_depth=1)])
    pool.cancel()
    pool.cancel()
    with pytest.raises(RuntimeError, match="closed"):
        pool.start()
    pool.close()


def test_close_before_start_leaves_the_pool_startable():
    pool = ReadyPool([PoolConfig(key="batch", produce=lambda: "batch", ready_depth=1)])
    pool.close()
    pool.start()
    try:
        assert pool.get("batch") == "batch"
    finally:
        pool.close()


def test_cancel_interrupts_a_blocked_queue_wait(monkeypatch):
    import tokenizer.aligned_data.loader.ready_pool._pool as module
    waiting = threading.Event()
    producing = threading.Event()
    release_worker = threading.Event()
    queues, failures = [], []
    class ObservedQueue(queue.Queue):
        def __init__(self, maxsize=0):
            super().__init__(maxsize)
            queues.append(self)
            original_wait = self.not_empty.wait
            def wait(timeout=None):
                waiting.set()
                return original_wait(timeout)
            monkeypatch.setattr(self.not_empty, "wait", wait)
    monkeypatch.setattr(module.queue, "Queue", ObservedQueue)
    def produce():
        producing.set()
        assert release_worker.wait(5)
        return "unused batch"
    pool = ReadyPool([PoolConfig(key="batch", produce=produce, ready_depth=1)]).start()
    def consume():
        try:
            pool.get("batch")
        except BaseException as failure:
            failures.append(failure)
    consumer = threading.Thread(target=consume, daemon=True)
    try:
        assert producing.wait(2)
        consumer.start()
        assert waiting.wait(2)
        # Acquiring the queue mutex proves Condition.wait has released it to block.
        with queues[0].not_empty:
            pool.cancel()
        consumer.join(2)
        assert not consumer.is_alive()
        assert len(failures) == 1 and "closed" in str(failures[0])
        with pytest.raises(RuntimeError, match="closed"):
            pool.start()
    finally:
        release_worker.set()
        pool.close()
        if consumer.ident:
            consumer.join(5)


def test_cancel_after_queue_delivery_refuses_the_batch(monkeypatch):
    import tokenizer.aligned_data.loader.ready_pool._pool as module
    class CancellingQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            result = super().get(block, timeout)
            pool.cancel()
            return result
    monkeypatch.setattr(module.queue, "Queue", CancellingQueue)
    pool = ReadyPool([PoolConfig(key="batch", produce=lambda: "batch", ready_depth=1)]).start()
    try:
        with pytest.raises(RuntimeError, match="closed"):
            pool.get("batch")
    finally:
        pool.close()
