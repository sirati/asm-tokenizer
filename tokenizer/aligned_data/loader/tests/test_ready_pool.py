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
