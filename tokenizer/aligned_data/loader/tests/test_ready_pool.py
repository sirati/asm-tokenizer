"""Tests for the threaded ready pool + its pipelined CUDA H2D overlap.

Layer 1 (:class:`ReadyPool`) is exercised with FAKE no-arg ``produce``
callables: the keep-N-ready invariant per config, FIFO order per config,
the backpressure bound, clean shutdown, and worker-death surfaced as a
RAISE (never a hang). Layer 2 (:class:`GpuReadyPool`) is exercised through
an INJECTED FAKE CudaBackend (no GPU) that records the ordered op sequence
+ stream identities -- asserting the pipelined overlap order: the previous
batch's copy event waited on the compute stream BEFORE return, the
newly-named batch's copy launched on the supplied stream, record_stream
issued. Mirrors the gpu_prefetcher cuda-backend test style.

A NO-MULTIPROCESSING guard asserts the package never imports
``multiprocessing`` (the team forbids it; threads suffice because the
decode releases the GIL in its Rust kernels).

A hang in the threaded orchestration must FAIL FAST with the parked thread
stacks (the same SIGALRM + faulthandler ceiling the gpu_prefetcher suite
uses), never sit for minutes.
"""

from __future__ import annotations

import faulthandler
import pathlib
import signal
import sys
import threading
import time
from typing import Any

import pytest

from tokenizer.aligned_data.loader.gpu_prefetcher._pytree import (
    map_tensor_leaves,
)
from tokenizer.aligned_data.loader.ready_pool import (
    GpuReadyPool,
    PoolConfig,
    ReadyPool,
    ReadyPoolWorkerDied,
)


_HARD_TIMEOUT_SECS = 30


@pytest.fixture(autouse=True)
def _hard_timeout():
    if not hasattr(signal, "SIGALRM"):  # pragma: no cover - non-POSIX
        yield
        return

    def _on_alarm(signum, frame):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        raise TimeoutError(
            f"test exceeded {_HARD_TIMEOUT_SECS}s hard timeout -- likely a "
            "ready-pool deadlock (see dumped thread stacks above)"
        )

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, _HARD_TIMEOUT_SECS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


# ==========================================================================
# Z. NO MULTIPROCESSING anywhere in the package (a hard team constraint)
# ==========================================================================
def test_package_never_imports_multiprocessing():
    # AST-level check: catch a real ``import multiprocessing`` /
    # ``from multiprocessing import ...`` statement, NOT the word appearing
    # in a docstring or comment (the package documents that it forbids it).
    import ast

    pkg_dir = pathlib.Path("tokenizer/aligned_data/loader/ready_pool")
    offenders = []
    for py in pkg_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    a.name == "multiprocessing"
                    or a.name.startswith("multiprocessing.")
                    for a in node.names
                ):
                    offenders.append(str(py))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "multiprocessing" or mod.startswith(
                    "multiprocessing."
                ):
                    offenders.append(str(py))
    assert offenders == [], (
        "ready_pool must never import multiprocessing (threads only); "
        f"offending files: {offenders}"
    )


# ==========================================================================
# A. LAYER 1 -- ReadyPool with FAKE produce
# ==========================================================================
class _CountingProduce:
    """A fake ``produce``: returns a monotonically increasing tagged batch.

    Thread-safe counter so the keep-N-ready + FIFO invariants are checked
    against a deterministic produce sequence.
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._n = 0
        self._lock = threading.Lock()

    def __call__(self) -> dict:
        with self._lock:
            i = self._n
            self._n += 1
        return {"tag": self.tag, "i": i}


def test_fifo_order_per_config_single_thread():
    cfg = PoolConfig(key="a", produce=_CountingProduce("a"), ready_depth=3)
    with ReadyPool([cfg], threads_per_config=1) as pool:
        got = [pool.get("a")["i"] for _ in range(8)]
    # ONE worker thread + FIFO queue => strictly increasing produce order.
    assert got == list(range(8))


def test_multiple_configs_each_isolated_fifo():
    cfgs = [
        PoolConfig(key="a", produce=_CountingProduce("a"), ready_depth=2),
        PoolConfig(key="b", produce=_CountingProduce("b"), ready_depth=4),
    ]
    with ReadyPool(cfgs, threads_per_config=1) as pool:
        a = [pool.get("a") for _ in range(5)]
        b = [pool.get("b") for _ in range(5)]
    assert [x["tag"] for x in a] == ["a"] * 5
    assert [x["i"] for x in a] == list(range(5))
    assert [x["tag"] for x in b] == ["b"] * 5
    assert [x["i"] for x in b] == list(range(5))


def test_keep_n_ready_invariant_and_backpressure_bound():
    # A produce that counts how many batches it has decoded. With one
    # worker and ready_depth=N, the worker fills to N then BLOCKS on the
    # bounded queue (never decodes past N+1 -- the one in its hand). So
    # before any get(), decoded count is bounded by ready_depth (+1 for the
    # batch the worker is mid-put on). This is the keep-N-ready + the
    # backpressure bound in one.
    READY_DEPTH = 3
    produce = _CountingProduce("a")
    cfg = PoolConfig(key="a", produce=produce, ready_depth=READY_DEPTH)
    with ReadyPool([cfg], threads_per_config=1) as pool:
        # Let the worker fill the buffer + park on the bounded put.
        deadline = time.monotonic() + 5.0
        while produce._n < READY_DEPTH and time.monotonic() < deadline:
            time.sleep(0.01)
        # Give the parked worker a beat to prove it does NOT keep decoding.
        time.sleep(0.2)
        # Bound: filled buffer (READY_DEPTH) + at most ONE in the worker's
        # hand blocked on the full put. NEVER unbounded.
        assert produce._n <= READY_DEPTH + 1, (
            f"decoded {produce._n} but ready_depth={READY_DEPTH}: "
            "backpressure bound violated (worker decoded past keep-N)"
        )
        # Draining frees slots; the worker refills (keep-N stays satisfied).
        first = [pool.get("a")["i"] for _ in range(READY_DEPTH + 2)]
    assert first == list(range(READY_DEPTH + 2))


def test_clean_shutdown_joins_threads():
    cfg = PoolConfig(key="a", produce=_CountingProduce("a"), ready_depth=2)
    pool = ReadyPool([cfg], threads_per_config=2).start()
    pool.get("a")
    pool.close()
    # All refill threads joined (none left alive after close()).
    for workers in pool._threads.values():
        for t in workers:
            assert not t.is_alive(), f"thread {t.name} survived close()"


class _ProduceRaisesAfter:
    """Produce N good batches, then RAISE -- simulates a refill thread death."""

    def __init__(self, good: int) -> None:
        self._good = good
        self._n = 0
        self._lock = threading.Lock()

    def __call__(self) -> dict:
        with self._lock:
            i = self._n
            self._n += 1
        if i >= self._good:
            raise ValueError(f"produce failed at batch {i}")
        return {"i": i}


def test_worker_exception_surfaces_as_raise_not_hang():
    # The escaped produce exception rides the ready queue and re-raises in
    # FIFO position; the good batches before it are still served.
    cfg = PoolConfig(
        key="a", produce=_ProduceRaisesAfter(good=2), ready_depth=4
    )
    with ReadyPool([cfg], threads_per_config=1) as pool:
        assert pool.get("a")["i"] == 0
        assert pool.get("a")["i"] == 1
        with pytest.raises(ValueError, match="produce failed at batch 2"):
            pool.get("a")


def _produce_dies_immediately() -> dict:
    # Raises on the FIRST call -- the single feeding thread dies before ANY
    # batch is buffered, so get() would park forever pre-fix. It must RAISE
    # promptly (the surfaced exception OR the dead-pool verdict).
    raise RuntimeError("instant produce death")


def test_silent_all_threads_dead_raises_not_hang():
    # Even if a death somehow left NO exception on the queue, a config whose
    # every feeding thread has exited must surface ReadyPoolWorkerDied via
    # the monitor rather than hang. We force the racy case by draining the
    # surfaced exception first, then asserting the next get() still raises
    # (the monitor verdict) instead of parking.
    cfg = PoolConfig(
        key="a", produce=_produce_dies_immediately, ready_depth=2
    )
    with ReadyPool([cfg], threads_per_config=1) as pool:
        t0 = time.monotonic()
        # First get(): the surfaced exception (FIFO) -- a RuntimeError.
        with pytest.raises(RuntimeError):
            pool.get("a")
        # Second get(): nothing left on the queue + the thread is dead, so
        # the monitor verdict raises ReadyPoolWorkerDied -- NOT a park.
        with pytest.raises(ReadyPoolWorkerDied, match="exited before"):
            pool.get("a")
        assert time.monotonic() - t0 < 10.0, "must raise promptly, not park"


def test_get_unknown_key_raises():
    cfg = PoolConfig(key="a", produce=_CountingProduce("a"), ready_depth=2)
    with ReadyPool([cfg]) as pool:
        with pytest.raises(KeyError):
            pool.get("nope")


def test_duplicate_keys_rejected_and_bad_depth_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ReadyPool(
            [
                PoolConfig("a", _CountingProduce("a"), 1),
                PoolConfig("a", _CountingProduce("a"), 1),
            ]
        )
    with pytest.raises(ValueError, match="ready_depth"):
        PoolConfig("a", _CountingProduce("a"), 0)


# ==========================================================================
# B. LAYER 2 -- GpuReadyPool overlap order via an INJECTED FAKE CudaBackend
# ==========================================================================
class FakeTensor:
    """Stand-in for a torch tensor leaf with a fake current device."""

    def __init__(self, name: str, device: str = "cpu") -> None:
        self.name = name
        self.device = device

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FakeTensor({self.name!r}@{self.device})"


def _is_fake_tensor(obj: Any) -> bool:
    return isinstance(obj, FakeTensor)


class _FakeStream:
    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_FakeStream({self.label!r})"


class _FakeEvent:
    def __init__(self, on_stream: _FakeStream) -> None:
        self.on_stream = on_stream


class _FakeCudaBackend:
    """A torch-free CudaBackend logging the ordered overlap ops + streams.

    Mirrors the gpu_prefetcher fake backend; the compute stream is the
    consumer's "current" stream the wait + record_stream run on. The
    upload runs on whatever ``stream`` the pool passes (the consumer's copy
    stream), proving Layer 2 launches each H2D on the supplied stream.
    """

    def __init__(self) -> None:
        self.log: list[tuple] = []
        self.lock = threading.Lock()
        self.compute_stream = _FakeStream("compute")
        self._cur_stream = self.compute_stream

    def _emit(self, *event: Any) -> None:
        with self.lock:
            self.log.append(event)

    def move_leaf(self, t: FakeTensor) -> FakeTensor:
        self._emit("pin", t.name)
        self._emit("copy", t.name, self._cur_stream.label, True)
        return FakeTensor(t.name, device="cuda")

    # -- CudaBackend surface ----------------------------------------------
    def make_copy_stream(self, device: Any) -> _FakeStream:  # pragma: no cover
        return _FakeStream("copy")  # Layer 2 uses the consumer-supplied one

    def upload(self, cpu_batch, device, copy_stream, move_one, is_leaf):
        event = _FakeEvent(on_stream=copy_stream)
        prev, self._cur_stream = self._cur_stream, copy_stream
        try:
            gpu_batch = map_tensor_leaves(cpu_batch, move_one, is_leaf=is_leaf)
        finally:
            self._cur_stream = prev
        self._emit("event_record", copy_stream.label)
        return (gpu_batch, event)

    def wait(self, gpu_batch, event, device, is_leaf):
        self._emit("wait_event", self.compute_stream.label, id(event))
        map_tensor_leaves(
            gpu_batch,
            lambda t: self._emit(
                "record_stream", t.name, self.compute_stream.label
            )
            or t,
            is_leaf=is_leaf,
        )


def _make_pool(tag: str = "a", depth: int = 4) -> ReadyPool:
    counter = {"n": 0}
    lock = threading.Lock()

    def produce():
        with lock:
            i = counter["n"]
            counter["n"] += 1
        # An upload-ready "pytree": two fake-tensor leaves + a scalar.
        return {"x": FakeTensor(f"{tag}x{i}"), "i": i}

    cfg = PoolConfig(key=tag, produce=produce, ready_depth=depth)
    return ReadyPool([cfg]).start()


def test_overlap_order_pipelined_with_fake_cuda():
    backend = _FakeCudaBackend()
    copy_stream = _FakeStream("copy")
    pool = _make_pool("a")
    try:
        gpool = GpuReadyPool(
            pool,
            device="cuda:fake",
            cuda_backend=backend,
            to_device=lambda t, _d: backend.move_leaf(t),
            is_leaf=_is_fake_tensor,
        )
        gpool.prime("a", stream=copy_stream)
        b0 = gpool.get("a", stream=copy_stream)
        b1 = gpool.get("a", stream=copy_stream)
    finally:
        pool.close()

    # Both returned batches are GPU-resident (every leaf moved).
    assert b0["x"].device == "cuda"
    assert b1["x"].device == "cuda"

    log = backend.log
    ops = [e[0] for e in log]

    # prime launched the first upload (copy on the SUPPLIED copy stream,
    # pinned first, non_blocking, event after).
    first_copy = next(k for k, e in enumerate(log) if e[0] == "copy")
    first_pin = next(k for k, e in enumerate(log) if e[0] == "pin")
    assert first_pin < first_copy, "leaf must be pinned before its copy"
    assert log[first_copy][2] == copy_stream.label, "copy on the supplied stream"
    assert log[first_copy][3] is True, "copy must be non_blocking"

    # The FIRST get(): the consumer waits on the primed event on the COMPUTE
    # stream BEFORE returning, and record_stream's the leaf on compute.
    i_wait = ops.index("wait_event")
    assert log[i_wait][1] == backend.compute_stream.label
    i_first_event = ops.index("event_record")
    assert i_wait > i_first_event, "wait must follow the copy event record"
    rec = [e for e in log if e[0] == "record_stream"]
    assert rec, "every uploaded leaf must be record_stream'd"
    assert all(e[2] == backend.compute_stream.label for e in rec)

    # Pipelined: get() ALSO launched the NEXT upload on the supplied stream
    # (a second copy on copy_stream appears after the first wait).
    copies = [k for k, e in enumerate(log) if e[0] == "copy"]
    assert len(copies) >= 2, "get() must launch the next batch's upload"
    assert all(log[k][2] == copy_stream.label for k in copies), (
        "every async H2D must run on the consumer-supplied copy stream"
    )


def test_get_before_prime_raises():
    backend = _FakeCudaBackend()
    pool = _make_pool("a")
    try:
        gpool = GpuReadyPool(
            pool,
            device="cuda:fake",
            cuda_backend=backend,
            to_device=lambda t, _d: backend.move_leaf(t),
            is_leaf=_is_fake_tensor,
        )
        with pytest.raises(RuntimeError, match="primed"):
            gpool.get("a", stream=_FakeStream("copy"))
    finally:
        pool.close()


def _pin_marker(t: FakeTensor) -> FakeTensor:
    """Stand-in for pin_host on a FakeTensor: tag it pinned, pass through."""
    return FakeTensor(t.name + "#pinned", device=t.device)


def test_option_c_returns_pinned_host_no_stream_ops():
    # Option C: no copy stream, no event -- get() returns the PINNED HOST
    # batch (the consumer owns the .to(cuda)). The fake backend's upload/
    # wait are never called. We inject a fake pin via the is_leaf seam so
    # the package's pin step is exercised without a real torch tensor.
    import tokenizer.aligned_data.loader.ready_pool._gpu_pool as gp

    backend = _FakeCudaBackend()
    pool = _make_pool("a")
    orig_pin = gp.pin_host
    gp.pin_host = _pin_marker  # the Option-C host move, faked
    try:
        gpool = GpuReadyPool(
            pool,
            device="cuda:fake",
            cuda_backend=backend,
            is_leaf=_is_fake_tensor,
            c_fallback=True,
        )
        gpool.prime("a", stream=_FakeStream("copy"))  # no-op under Option C
        batch = gpool.get("a", stream=_FakeStream("copy"))
    finally:
        gp.pin_host = orig_pin
        pool.close()
    assert "i" in batch  # the scalar passed through
    # The host leaf was PINNED (Option-C move), NOT moved to cuda.
    assert batch["x"].name.endswith("#pinned")
    assert batch["x"].device == "cpu"
    # No upload/wait/event ops were recorded (Option C bypasses the stream).
    assert backend.log == []


def test_mixed_config_sequence_interleaves():
    # The pipeline names the NEXT key and collects the PREVIOUS one, so a
    # consumer can interleave two configs (curriculum shape switching)
    # through the single in-flight slot without stalling.
    backend = _FakeCudaBackend()
    copy_stream = _FakeStream("copy")
    counter = {"a": 0, "b": 0}
    lock = threading.Lock()

    def make_produce(tag):
        def produce():
            with lock:
                i = counter[tag]
                counter[tag] += 1
            return {"x": FakeTensor(f"{tag}{i}"), "tag": tag, "i": i}
        return produce

    cfgs = [
        PoolConfig("a", make_produce("a"), 3),
        PoolConfig("b", make_produce("b"), 3),
    ]
    pool = ReadyPool(cfgs).start()
    try:
        gpool = GpuReadyPool(
            pool,
            device="cuda:fake",
            cuda_backend=backend,
            to_device=lambda t, _d: backend.move_leaf(t),
            is_leaf=_is_fake_tensor,
        )
        gpool.prime("a", stream=copy_stream)
        # Name b after priming a => first get returns an 'a' batch.
        b_first = gpool.get("b", stream=copy_stream)
        # Name a => this get returns the 'b' batch named last call.
        b_second = gpool.get("a", stream=copy_stream)
    finally:
        pool.close()
    assert b_first["tag"] == "a", "first get returns the primed (a) batch"
    assert b_second["tag"] == "b", "next get returns the previously-named (b)"


# ==========================================================================
# C. DECODE SEAM end-to-end -- real vector_batch_tokens through the pool
# ==========================================================================
# Validates make_vector_batch_produce wires the pluggable sampler ->
# vector_batch_tokens -> postprocess, opens per-binary handles+session once
# and reuses them across many draws, on a REAL small corpus fixture.
def test_vector_batch_produce_end_to_end(tmp_path):
    import numpy as np

    from tokenizer.aligned_data.loader.batch_decode._types import (
        SectionPointerSpec,
    )
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind
    from tokenizer.aligned_data.loader.ready_pool import (
        DecodeParams,
        make_vector_batch_produce,
    )
    from tokenizer.aligned_data.loader.vector_batch._result import (
        VectorBatchResult,
    )
    from tokenizer.aligned_data.loader.vector_batch.tests._byte_identity_harness import (  # noqa: E501
        _BINARY_NAME,
        _nonempty_matched_idxs,
        _prepare,
    )
    from tokenizer.aligned_data.sorted_index.tests.fixtures import (
        build_combined_fixture,
        make_test_vocab_manager,
    )

    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    assert idxs, "fixture must have sample-able matched sections"

    L = 64
    B = min(3, len(idxs))

    def sampler(rng) -> tuple:
        # A fixed-B matched-arm draw -- the pluggable sampler owns B + arm.
        chosen = rng.choice(len(idxs), size=B, replace=False)
        pointers = [
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(idxs[i]))
            for i in chosen
        ]
        return (_BINARY_NAME, pointers)

    # postprocess marks that the FINAL produce stage ran (off the worker).
    def postprocess(result: VectorBatchResult) -> dict:
        return {"result": result, "B": result.tokens.shape[0]}

    produce = make_vector_batch_produce(
        base_path=base,
        sampler=sampler,
        decode_params=DecodeParams(
            context_len=L,
            num_variants_per_section=2,
            max_depth=3,
            include_fid_sidecar=True,
        ),
        postprocess=postprocess,
        vocab_manager=make_test_vocab_manager(),
        seed=123,
    )

    cfg = PoolConfig(key="train", produce=produce, ready_depth=3)
    with ReadyPool([cfg], threads_per_config=1) as pool:
        batches = [pool.get("train") for _ in range(5)]

    for b in batches:
        res = b["result"]
        assert isinstance(res, VectorBatchResult)
        # context_len = L was honoured; the row count is the sampler's B.
        assert res.tokens.shape[1] == L
        assert res.tokens.shape[0] == b["B"]
        assert res.tokens.dtype == np.uint16
        # include_fid_sidecar threaded through.
        assert res.fid_sidecar is not None
    # 5 draws succeeded against ONE reused per-binary session+handles
    # (open-once-reuse) -- a fresh open per draw would still pass but this
    # exercises the reuse path the pool depends on for throughput.


# ==========================================================================
# D. CLEANUP SEAM -- the OPTIONAL CloseableProduce protocol on shutdown
# ==========================================================================
# The pool's refill loop calls produce.close() in a finally when each worker
# stops, IF the produce honours the CloseableProduce protocol -- duck-typed
# via the runtime-checkable Protocol, never an isinstance ladder over decode
# types. A non-closeable produce (the common case in the fake tests) must
# still work untouched.
from tokenizer.aligned_data.loader.ready_pool import (  # noqa: E402
    CloseableProduce,
    is_closeable_produce,
)


class _CloseableProduce:
    """A fake closeable produce: counts calls + records its close() call(s).

    ``close`` is invoked PER REFILL THREAD (the thread-local resource owner),
    so the counter is thread-safe and the test asserts close was called at
    least once -- exactly once per worker that ran this produce.
    """

    def __init__(self) -> None:
        self._n = 0
        self.close_calls = 0
        self._lock = threading.Lock()

    def __call__(self) -> dict:
        with self._lock:
            i = self._n
            self._n += 1
        return {"i": i}

    def close(self) -> None:
        with self._lock:
            self.close_calls += 1


def test_closeable_produce_satisfies_protocol():
    # The structural seam: a __call__ + close() object is a CloseableProduce;
    # a plain __call__-only fake (and a bare function) is NOT.
    assert is_closeable_produce(_CloseableProduce())
    assert isinstance(_CloseableProduce(), CloseableProduce)
    assert not is_closeable_produce(_CountingProduce("a"))
    assert not is_closeable_produce(_produce_dies_immediately)


def test_worker_closes_produce_on_clean_shutdown():
    produce = _CloseableProduce()
    cfg = PoolConfig(key="a", produce=produce, ready_depth=2)
    pool = ReadyPool([cfg], threads_per_config=1).start()
    pool.get("a")  # ensure the worker ran at least one produce.
    pool.close()
    # The single refill thread released its produce resources on shutdown.
    assert produce.close_calls == 1, (
        f"expected one close() on clean shutdown, got {produce.close_calls}"
    )


def test_worker_closes_produce_per_thread_on_shutdown():
    # N threads over one config => close() called once PER worker (each owns
    # its own thread-local resources; the pool closes each independently).
    produce = _CloseableProduce()
    cfg = PoolConfig(key="a", produce=produce, ready_depth=4)
    pool = ReadyPool([cfg], threads_per_config=3).start()
    for _ in range(6):
        pool.get("a")
    pool.close()
    assert produce.close_calls == 3, (
        f"expected one close() per worker (3), got {produce.close_calls}"
    )


def test_worker_closes_produce_even_after_produce_raises():
    # The finally fires on the fatal-exception exit too: a produce that
    # raises (worker dies) still gets close() so its resources are released.
    class _RaiseThenCloseable(_CloseableProduce):
        def __call__(self) -> dict:
            super().__call__()
            raise ValueError("boom")

    produce = _RaiseThenCloseable()
    cfg = PoolConfig(key="a", produce=produce, ready_depth=2)
    with ReadyPool([cfg], threads_per_config=1) as pool:
        with pytest.raises(ValueError, match="boom"):
            pool.get("a")
        # The dead worker still ran its close() in the finally.
        deadline = time.monotonic() + 5.0
        while produce.close_calls < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert produce.close_calls == 1


def test_non_closeable_produce_shuts_down_without_crash():
    # The seam is OPTIONAL: a plain __call__-only produce must shut down
    # cleanly (the pool simply skips the close() call). No crash, threads
    # joined.
    cfg = PoolConfig(key="a", produce=_CountingProduce("a"), ready_depth=2)
    pool = ReadyPool([cfg], threads_per_config=1).start()
    pool.get("a")
    pool.close()  # must not raise despite no close() on the produce.
    for workers in pool._threads.values():
        for t in workers:
            assert not t.is_alive()


def test_close_failure_does_not_break_shutdown():
    # A close() that raises must not crash teardown nor leave threads alive
    # (best-effort release; the worker is already stopping).
    class _BadClose(_CloseableProduce):
        def close(self) -> None:
            raise RuntimeError("close blew up")

    produce = _BadClose()
    cfg = PoolConfig(key="a", produce=produce, ready_depth=2)
    pool = ReadyPool([cfg], threads_per_config=1).start()
    pool.get("a")
    pool.close()  # swallows the close() failure.
    for workers in pool._threads.values():
        for t in workers:
            assert not t.is_alive()


# ==========================================================================
# E. FACADE -- VectorBatchDataLoader (construction-time postprocess + wiring)
# ==========================================================================
# A single construction-time postprocess is fanned out to EVERY config's
# produce (running on the worker thread); the facade composes ReadyPool [+
# GpuReadyPool] and delegates get()/prime+get(stream). Exercised with a fake
# produce injected through make_vector_batch_produce on a real fixture for
# the CPU path, and the fake CudaBackend pattern for the GPU path.
def test_facade_construction_postprocess_applied_to_every_config(tmp_path):
    import numpy as np

    from tokenizer.aligned_data.loader.batch_decode._types import (
        SectionPointerSpec,
    )
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind
    from tokenizer.aligned_data.loader.ready_pool import (
        DataLoaderConfig,
        DecodeParams,
        VectorBatchDataLoader,
    )
    from tokenizer.aligned_data.loader.vector_batch.tests._byte_identity_harness import (  # noqa: E501
        _BINARY_NAME,
        _nonempty_matched_idxs,
        _prepare,
    )
    from tokenizer.aligned_data.sorted_index.tests.fixtures import (
        build_combined_fixture,
        make_test_vocab_manager,
    )

    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    assert idxs

    L = 48

    def make_sampler(b):
        def sampler(rng) -> tuple:
            chosen = rng.choice(len(idxs), size=b, replace=False)
            pointers = [
                SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(idxs[i]))
                for i in chosen
            ]
            return (_BINARY_NAME, pointers)

        return sampler

    # ONE construction-time postprocess; the marker proves it ran for EVERY
    # config's batches (off the worker thread, as the final produce stage).
    def postprocess(result) -> dict:
        return {"marked": True, "tokens": result.tokens}

    vocab = make_test_vocab_manager()
    cfgs = [
        DataLoaderConfig.single_binary(
            key="small",
            base_path=base,
            sampler=make_sampler(min(2, len(idxs))),
            decode_params=DecodeParams(context_len=L),
            ready_depth=2,
            vocab_manager=vocab,
            seed=7,
        ),
        DataLoaderConfig.single_binary(
            key="big",
            base_path=base,
            sampler=make_sampler(min(3, len(idxs))),
            decode_params=DecodeParams(context_len=L),
            ready_depth=2,
            vocab_manager=vocab,
            seed=7,
        ),
    ]

    with VectorBatchDataLoader(
        configs=cfgs,
        postprocess=postprocess,
    ) as loader:
        for key in ("small", "big"):
            for _ in range(3):
                batch = loader.get(key)
                assert batch["marked"] is True
                assert batch["tokens"].shape[1] == L
                assert batch["tokens"].dtype == np.uint16


def test_facade_per_config_postprocess_override(tmp_path):
    from tokenizer.aligned_data.loader.batch_decode._types import (
        SectionPointerSpec,
    )
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind
    from tokenizer.aligned_data.loader.ready_pool import (
        DataLoaderConfig,
        DecodeParams,
        VectorBatchDataLoader,
    )
    from tokenizer.aligned_data.loader.vector_batch.tests._byte_identity_harness import (  # noqa: E501
        _BINARY_NAME,
        _nonempty_matched_idxs,
        _prepare,
    )
    from tokenizer.aligned_data.sorted_index.tests.fixtures import (
        build_combined_fixture,
        make_test_vocab_manager,
    )

    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    assert idxs

    def sampler(rng) -> tuple:
        i = int(rng.choice(len(idxs)))
        return (_BINARY_NAME, [SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(idxs[i]))])

    vocab = make_test_vocab_manager()
    cfgs = [
        DataLoaderConfig.single_binary(
            key="default",
            base_path=base,
            sampler=sampler,
            decode_params=DecodeParams(context_len=32),
            ready_depth=2,
            vocab_manager=vocab,
            seed=7,
        ),
        # This config overrides the construction-level postprocess cleanly.
        DataLoaderConfig.single_binary(
            key="override",
            base_path=base,
            sampler=sampler,
            decode_params=DecodeParams(context_len=32),
            ready_depth=2,
            postprocess=lambda result: {"via": "override"},
            vocab_manager=vocab,
            seed=7,
        ),
    ]

    with VectorBatchDataLoader(
        configs=cfgs,
        postprocess=lambda result: {"via": "construction"},
    ) as loader:
        assert loader.get("default")["via"] == "construction"
        assert loader.get("override")["via"] == "override"


def test_facade_cpu_get_without_device():
    # Without a device the facade is the CPU path: get(key) delegates to the
    # wrapped ReadyPool. A fake closeable produce stands in for the decode.
    from tokenizer.aligned_data.loader.ready_pool import VectorBatchDataLoader

    loader = VectorBatchDataLoader.__new__(VectorBatchDataLoader)
    # Wire a tiny pool by hand to assert delegation without a real decode.
    cfg = PoolConfig(key="a", produce=_CountingProduce("a"), ready_depth=2)
    loader._postprocess = None
    loader._pool = ReadyPool([cfg], threads_per_config=1)
    loader._gpu = None
    with loader:
        assert loader.get("a")["i"] == 0
        # CPU path rejects the GPU-only surface cleanly.
        with pytest.raises(RuntimeError, match="GPU pipelined path"):
            loader.prime("a", stream=_FakeStream("copy"))


def test_facade_gpu_prime_get_delegates_to_gpu_pool():
    # With a device the facade exposes the pipelined GPU path; prime/get(
    # stream) delegate to the wrapped GpuReadyPool. Reuses the fake CudaBackend
    # so the overlap is exercised WITHOUT a GPU.
    from tokenizer.aligned_data.loader.ready_pool import VectorBatchDataLoader

    backend = _FakeCudaBackend()
    copy_stream = _FakeStream("copy")

    counter = {"n": 0}
    lock = threading.Lock()

    def produce():
        with lock:
            i = counter["n"]
            counter["n"] += 1
        return {"x": FakeTensor(f"ax{i}"), "i": i}

    # Build the facade but inject the fake-backed pool/gpu through the same
    # public knobs the production path uses (cuda_backend + to_device +
    # is_leaf seams). We hand-wire the pool to use the fake produce (the
    # facade's only job is wiring; the decode is orthogonal here).
    loader = VectorBatchDataLoader.__new__(VectorBatchDataLoader)
    loader._postprocess = None
    cfg = PoolConfig(key="a", produce=produce, ready_depth=4)
    loader._pool = ReadyPool([cfg], threads_per_config=1)
    loader._gpu = GpuReadyPool(
        loader._pool,
        device="cuda:fake",
        cuda_backend=backend,
        to_device=lambda t, _d: backend.move_leaf(t),
        is_leaf=_is_fake_tensor,
    )
    with loader:
        loader.prime("a", stream=copy_stream)
        b0 = loader.get("a", stream=copy_stream)
        b1 = loader.get("a", stream=copy_stream)
    assert b0["x"].device == "cuda"
    assert b1["x"].device == "cuda"
    # The pipeline launched uploads on the supplied copy stream (delegation).
    copies = [e for e in backend.log if e[0] == "copy"]
    assert len(copies) >= 2
    assert all(e[2] == copy_stream.label for e in copies)


def test_facade_gpu_get_requires_stream():
    from tokenizer.aligned_data.loader.ready_pool import VectorBatchDataLoader

    backend = _FakeCudaBackend()
    loader = VectorBatchDataLoader.__new__(VectorBatchDataLoader)
    loader._postprocess = None
    cfg = PoolConfig(
        key="a", produce=lambda: {"x": FakeTensor("a0")}, ready_depth=2
    )
    loader._pool = ReadyPool([cfg], threads_per_config=1)
    loader._gpu = GpuReadyPool(
        loader._pool,
        device="cuda:fake",
        cuda_backend=backend,
        to_device=lambda t, _d: backend.move_leaf(t),
        is_leaf=_is_fake_tensor,
    )
    with loader:
        with pytest.raises(TypeError, match="requires a stream"):
            loader.get("a")  # GPU path without a stream is a usage error.


def test_facade_rejects_empty_configs():
    from tokenizer.aligned_data.loader.ready_pool import (
        VectorBatchDataLoader,
    )

    with pytest.raises(ValueError, match="at least one"):
        VectorBatchDataLoader(configs=[])


def test_facade_from_produce_registers_prebuilt_and_ignores_postprocess():
    # from_produce plugs a consumer-owned CloseableProduce (e.g. a redraw-
    # retry wrapper) into the facade's threaded keep-N engine. The
    # construction postprocess is IGNORED (the produce already bakes in
    # whatever it wants), and close() still fires per worker on shutdown.
    from tokenizer.aligned_data.loader.ready_pool import (
        DataLoaderConfig,
        VectorBatchDataLoader,
    )

    produce = _CloseableProduce()

    def _would_mutate(batch):
        return {**batch, "postprocessed": True}

    cfg = DataLoaderConfig.from_produce(
        key="retry", produce=produce, ready_depth=2
    )
    with VectorBatchDataLoader(configs=[cfg], postprocess=_would_mutate) as dl:
        batch = dl.get("retry")
    # The produce's OWN output (FIFO oldest), NOT run through the
    # construction postprocess -- the thunk dropped _pp on the floor.
    assert "postprocessed" not in batch
    assert batch == {"i": 0}
    # The pool released the consumer's produce on shutdown.
    assert produce.close_calls == 1


# ==========================================================================
# F. CROSS-BINARY decode seam -- make_cross_binary_produce over
#    load_batch_cross_depth, BYTE-IDENTICAL to the primitive.
# ==========================================================================
# The cross-binary x cross-depth training distribution: one batch draws
# across MULTIPLE binaries and downstream REQUIRES per-row binary identity.
# make_cross_binary_produce must WRAP (never reimplement) the existing
# load_batch_cross_depth primitive, so a seam draw is byte-identical to a
# direct primitive draw on the same collection with the same RNG. These tests
# build a REAL two-binary collection fixture (mirroring the sorted_index
# cross-depth tests) so the byte-identity gate runs hermetically.

import contextlib  # noqa: E402
import logging as _xb_logging  # noqa: E402


@contextlib.contextmanager
def caplog_silence():
    """Mute the cross-depth path's noisy per-draw logging during the gate."""
    prev_disable = _xb_logging.root.manager.disable
    _xb_logging.disable(_xb_logging.CRITICAL)
    try:
        yield
    finally:
        _xb_logging.disable(prev_disable)


def _build_cross_binary_collection_factory(tmp_path):
    """Build a TWO-binary, three-depth collection + a no-arg open thunk.

    Lays two distinct binaries (distinct func/section pools) into two memmap
    dirs with the real sorted index + geometry sidecars over depths {0,1,3}
    under a MAX reduction, mirroring the sorted_index cross-depth fixture.
    Returns ``(specs, collection_factory)`` -- the factory opens a FRESH
    production-shaped :class:`IndexedMemmapCollection` over BOTH dirs each
    call (so the seam can open one per thread, and the reference can open its
    own).
    """
    import numpy as np

    from tokenizer.aligned_data.realized_lengths import (
        generate_realized_geometry,
    )
    from tokenizer.aligned_data.sorted_index import (
        IndexSpec,
        IndexedMemmapCollection,
        LengthReduction,
        MissingIndexPolicy,
        ReductionKind,
    )
    from tokenizer.aligned_data.sorted_index._builder import (
        write_sorted_index_files,
    )
    from tokenizer.aligned_data.loader.tests._corpus import (
        MatchedFunctionSpec,
        build_corpus_with_registry,
    )
    from tokenizer.aligned_data.loader.tests._corpus.specs import VariantSpec
    from tokenizer.aligned_data.sorted_index.tests.fixtures import (
        _DeterministicVariantRegistry,
        make_test_vocab_manager,
    )
    from tokenizer.aligned_data.sorted_index.tests._length_helpers import (
        ensure_sidecar,
    )

    max_red = LengthReduction(ReductionKind.MAX)
    depths = (0, 1, 3)
    specs = [IndexSpec(reduction=max_red, depth=d) for d in depths]

    def _simple_variant(vkey, seed_base: int, n_tokens: int) -> VariantSpec:
        base = 272 + (seed_base + 1) * 100
        tokens = np.arange(base, base + n_tokens, dtype=np.uint16)
        return VariantSpec(
            vkey=vkey,
            tokens=tokens,
            block_rl=np.array([n_tokens], dtype=np.uint8),
            insn_rl=np.array([2, n_tokens - 2], dtype=np.uint8),
        )

    def _build_binary(memmap_dir, binary_name: str, salt: int) -> None:
        memmap_dir.mkdir(parents=True, exist_ok=True)
        matched = (
            MatchedFunctionSpec(
                func_name=f"f_{binary_name}_a",
                variants=(
                    _simple_variant((f"{binary_name}_a", 0), salt + 0, 7),
                    _simple_variant((f"{binary_name}_a", 1), salt + 1, 9),
                ),
                called=(),
            ),
            MatchedFunctionSpec(
                func_name=f"f_{binary_name}_b",
                variants=(
                    _simple_variant((f"{binary_name}_b", 0), salt + 2, 6),
                    _simple_variant((f"{binary_name}_b", 1), salt + 3, 8),
                ),
                called=(),
            ),
        )
        build_corpus_with_registry(
            memmap_dir,
            binary_name,
            matched=matched,
            unmatched=(),
            variants=_DeterministicVariantRegistry(),
        )
        ensure_sidecar(memmap_dir, binary_name)
        write_sorted_index_files(
            memmap_dir, binary_name, reductions=[max_red], depths=list(depths)
        )
        generate_realized_geometry(memmap_dir, binary_name)

    dir_a = tmp_path / "pkgA"
    dir_b = tmp_path / "pkgB"
    _build_binary(dir_a, "alpha", salt=0)
    _build_binary(dir_b, "beta", salt=10)

    def collection_factory():
        return IndexedMemmapCollection.discover(
            [dir_a, dir_b],
            specs=specs,
            on_missing=MissingIndexPolicy.SKIP_WITH_ERROR_LOG,
            vocab_manager=make_test_vocab_manager(),
        )

    return specs, collection_factory


def _assert_multibinary_equal(ref, got) -> None:
    """Byte-equality of two :class:`MultiBinaryBatchDecodeResult`."""
    import numpy as np

    assert np.array_equal(got.inner.tokens, ref.inner.tokens)
    assert np.array_equal(got.binary_id_per_row, ref.binary_id_per_row)
    assert got.binary_names == ref.binary_names
    assert np.array_equal(got.depth_per_row, ref.depth_per_row)
    # Inner sidecar arrays + their row offsets.
    assert np.array_equal(got.inner.identities, ref.inner.identities)
    assert np.array_equal(
        got.inner.identity_row_offsets, ref.inner.identity_row_offsets
    )
    assert np.array_equal(
        got.inner.numbers_significant, ref.inner.numbers_significant
    )
    assert np.array_equal(
        got.inner.numbers_sign_exponent, ref.inner.numbers_sign_exponent
    )
    assert np.array_equal(
        got.inner.number_row_offsets, ref.inner.number_row_offsets
    )
    # fid sidecar (present iff include_fid_sidecar).
    assert np.array_equal(got.inner.fid_sidecar, ref.inner.fid_sidecar)
    assert np.array_equal(
        got.inner.fid_row_offsets, ref.inner.fid_row_offsets
    )


def test_cross_binary_produce_byte_identical_to_primitive(tmp_path):
    # THE GATE: a seam draw reproduces load_batch_cross_depth EXACTLY. The
    # seam derives its per-thread rng as
    # default_rng(SeedSequence(S).spawn(1)[0]); construct the reference RNG
    # the SAME way so the cross-(binary x spec) urn draws are identical.
    import numpy as np

    from tokenizer.aligned_data.loader.ready_pool import (
        CrossDecodeParams,
        PoolConfig,
        ReadyPool,
        make_cross_binary_produce,
    )

    _, collection_factory = _build_cross_binary_collection_factory(tmp_path)

    SEED = 4242
    params = CrossDecodeParams(
        target_length=0,
        batch_size=16,
        context_len=64,
        num_variants_per_section=1,
        band=(1, 10_000_000),
        include_fid_sidecar=True,
    )

    # Reference: drive the primitive directly with the SAME per-thread rng
    # the seam will derive from SEED.
    ref_rng = np.random.default_rng(
        np.random.SeedSequence(SEED).spawn(1)[0]
    )
    with caplog_silence():
        with collection_factory() as ref_coll:
            ref = ref_coll.load_batch_cross_depth(
                target_length=params.target_length,
                batch_size=params.batch_size,
                rng=ref_rng,
                band=params.band,
                context_len=params.context_len,
                num_variants_per_section=params.num_variants_per_section,
                variant_padding=params.variant_padding,
                inlined_equivalent_call_targets_only=(
                    params.inlined_equivalent_call_targets_only
                ),
                include_fid_sidecar=params.include_fid_sidecar,
            )

    # Seam: a 1-thread ReadyPool over make_cross_binary_produce, get() once.
    produce = make_cross_binary_produce(
        collection_factory=collection_factory,
        params=params,
        seed=SEED,
    )
    cfg = PoolConfig(key="xb", produce=produce, ready_depth=2)
    with caplog_silence():
        with ReadyPool([cfg], threads_per_config=1) as pool:
            got = pool.get("xb")

    _assert_multibinary_equal(ref, got)
    # Sanity: a genuine cross-binary batch carries both binaries' identities
    # (alphabetical, unqualified when the dir names don't collide).
    assert set(got.binary_names) == {"alpha", "beta"}
    assert got.binary_id_per_row.max() < len(got.binary_names)


def test_cross_binary_seam_thread_safety(tmp_path):
    # threads_per_config=2 over the cross seam runs many gets without error;
    # each result carries binary_id_per_row aligned to binary_names (ids in
    # range, names consistent). Each thread owns its OWN collection + RNG.
    from tokenizer.aligned_data.loader.ready_pool import (
        CrossDecodeParams,
        PoolConfig,
        ReadyPool,
        make_cross_binary_produce,
    )

    _, collection_factory = _build_cross_binary_collection_factory(tmp_path)

    params = CrossDecodeParams(
        target_length=0,
        batch_size=12,
        context_len=48,
        num_variants_per_section=1,
        band=(1, 10_000_000),
    )
    produce = make_cross_binary_produce(
        collection_factory=collection_factory,
        params=params,
        seed=7,
    )
    cfg = PoolConfig(key="xb", produce=produce, ready_depth=4)
    with caplog_silence():
        with ReadyPool([cfg], threads_per_config=2) as pool:
            results = [pool.get("xb") for _ in range(20)]
    for res in results:
        n = len(res.binary_names)
        assert n >= 1
        assert res.binary_id_per_row.min() >= 0
        assert res.binary_id_per_row.max() < n
        assert res.binary_id_per_row.shape[0] == res.inner.tokens.shape[0]


def test_cross_binary_callable_batch_size_resolved_per_draw(tmp_path):
    # batch_size may be a no-arg callable resolved on EACH draw (e.g. an OOM
    # auto-recovery loop shrinking the live B for this bucket); consecutive
    # produce() calls honour the changing size -- so a dynamic-B consumer can
    # drive the clean .cross_binary facade instead of bypassing it.
    from tokenizer.aligned_data.loader.ready_pool import (
        CrossDecodeParams,
        make_cross_binary_produce,
    )

    _, collection_factory = _build_cross_binary_collection_factory(tmp_path)

    # Both sizes are below the fixture's available-section count, so the urn
    # draws EXACTLY B; differing values prove the callable is resolved per draw.
    sizes = iter([8, 4])
    params = CrossDecodeParams(
        target_length=0,
        batch_size=lambda: next(sizes),
        context_len=48,
        band=(1, 10_000_000),
    )
    produce = make_cross_binary_produce(
        collection_factory=collection_factory, params=params, seed=99
    )
    with caplog_silence():
        first = produce()
        second = produce()
        produce.close()
    # Each draw resolved the callable -> B rows in that batch.
    assert first.inner.tokens.shape[0] == 8
    assert second.inner.tokens.shape[0] == 4
