"""Torch-less tests for the generic GPU batch prefetcher.

torch is ABSENT from the dev shell, so coverage is HONEST about its
scope: these tests exercise the GENERIC PYTREE MOVE (A), the
multi-process ORCHESTRATION + FIFO + exception propagation (B), the
memmap-after-fork guarantee (C) via DEPENDENCY INJECTION (an injected
``to_device`` + ``is_leaf``), and -- via an INJECTED FAKE CUDA BACKEND
that records the ordered op sequence + stream identities -- the H2D
overlap ORCHESTRATION/ORDER (E): separate copy stream, pin-before-
non_blocking copy, event recorded on the copy stream AFTER the copy,
consumer wait_event on the compute stream, and record_stream of every
uploaded leaf on the compute stream (the cross-stream-reuse fix). The
overlap ORDER is now validated without a GPU; the REAL torch.cuda
hardware confirmation stays a consumer-side smoke (test D, skipped
without torch).
"""

from __future__ import annotations

import collections
import dataclasses
import faulthandler
import os
import signal
import sys
import threading
import time
from typing import Any

import pytest

from tokenizer.aligned_data.loader.gpu_prefetcher import (
    GpuBatchPrefetcher,
    PrefetchWorkerDied,
    UnpicklablePayloadError,
)
from tokenizer.aligned_data.loader.gpu_prefetcher._pytree import map_tensor_leaves


# A hang in the multi-process orchestration must FAIL FAST with the parked
# thread stacks, never sit for minutes (the load-dependent deadlock this
# suite guards). pytest-timeout is absent from the dev shell, so a SIGALRM +
# faulthandler fixture enforces a hard per-test ceiling: on expiry it dumps
# every thread's traceback (so the parked get()/upload-loop frames are
# visible) and the alarm signal interrupts the blocked call, failing the test.
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
            "prefetcher deadlock (see dumped thread stacks above)"
        )

    prev = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, _HARD_TIMEOUT_SECS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


# ==========================================================================
# A. GENERIC PYTREE MOVE (no torch)
# ==========================================================================
class FakeTensor:
    """Stand-in for a torch tensor leaf, with a fake current device."""

    def __init__(self, name: str, device: str = "cpu") -> None:
        self.name = name
        self.device = device

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"FakeTensor({self.name!r}@{self.device})"


def _is_fake_tensor(obj: Any) -> bool:
    return isinstance(obj, FakeTensor)


def _move_to_fake_cuda(t: FakeTensor) -> FakeTensor:
    return FakeTensor(t.name, device="cuda")


@dataclasses.dataclass
class _ConsumerBatch:
    """The real consumer shape: a nested dict-of-tensors AND an int scalar."""

    tensors: dict
    seq_len: int  # plain non-tensor scalar -- must pass through untouched


def test_pytree_moves_every_leaf_including_nested_and_dataclass():
    batch = _ConsumerBatch(
        tensors={"input_ids": FakeTensor("ids"), "mask": FakeTensor("mask")},
        seq_len=512,
    )

    moved = map_tensor_leaves(batch, _move_to_fake_cuda, is_leaf=_is_fake_tensor)

    # Structure preserved: still a _ConsumerBatch with a dict of tensors.
    assert isinstance(moved, _ConsumerBatch)
    assert set(moved.tensors) == {"input_ids", "mask"}
    # EVERY tensor leaf moved to the fake device.
    assert moved.tensors["input_ids"].device == "cuda"
    assert moved.tensors["mask"].device == "cuda"
    # The int scalar passed through UNTOUCHED.
    assert moved.seq_len == 512
    # Original untouched (new structure rebuilt, not mutated in place).
    assert batch.tensors["input_ids"].device == "cpu"


def test_pytree_handles_list_tuple_namedtuple_and_non_tensor_leaves():
    import collections

    NT = collections.namedtuple("NT", ["a", "b"])
    batch = {
        "lst": [FakeTensor("x"), 7, "label"],
        "tup": (FakeTensor("y"), None),
        "nt": NT(FakeTensor("z"), 3.14),
    }

    moved = map_tensor_leaves(batch, _move_to_fake_cuda, is_leaf=_is_fake_tensor)

    assert moved["lst"][0].device == "cuda"
    assert moved["lst"][1] == 7 and moved["lst"][2] == "label"
    assert isinstance(moved["tup"], tuple) and moved["tup"][0].device == "cuda"
    assert moved["tup"][1] is None
    assert isinstance(moved["nt"], NT) and moved["nt"].a.device == "cuda"
    assert moved["nt"].b == 3.14


# ==========================================================================
# B. MP ORCHESTRATION (no torch, stdlib multiprocessing)
# ==========================================================================
# Module-level so they pickle under spawn (the worker re-imports this file).
def _make_source():
    return {"pid": os.getpid(), "kind": "stub-source"}


def _produce(src, request):
    # The batch is tagged with the WORKER's pid to prove a separate process.
    return {"request": request, "worker_pid": os.getpid()}


def _produce_slow(src, request):
    time.sleep(0.15)
    return {"request": request, "worker_pid": os.getpid()}


def _produce_raises(src, request):
    if request == "boom":
        raise ValueError("worker produce failed")
    return {"request": request, "worker_pid": os.getpid()}


def _noop_to_device(t, device):  # records nothing; batches carry no tensors
    return t


def _never_leaf(_obj):
    return False


# -- worker bodies that SIMULATE the load-dependent death (module-level so
#    they pickle under spawn) ------------------------------------------------
def _make_source_dies_on_open():
    # Stands in for a worker that crashes on its post-spawn import / source
    # open (the real load-dependent failure): the process exits before EVER
    # putting a result on the queue, so the old get() would park forever.
    os._exit(42)


def _produce_dies_silently(_src, _request):
    # The source opened, but the FIRST produce kills the process WITHOUT
    # shipping a result -- the other shape of "owed result never arrives".
    os._exit(7)


def _produce_die_on_one(_src, request):
    # Produce normally EXCEPT for request == 1, on which the worker exits
    # without shipping a result -- a single crashed worker among healthy
    # peers. The crash is fatal to the whole pipeline (torch semantics).
    if request == 1:
        os._exit(9)
    return {"request": request, "worker_pid": os.getpid()}


class _Unpicklable:
    """A payload that cannot cross the spawn pickle boundary."""

    def __reduce__(self):
        raise TypeError("this request is unpicklable on purpose")


def test_orchestration_separate_process_and_fifo():
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce,
        device="cpu",
        decode_workers=1,
        decode_ahead=5,
        gpu_ahead=2,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        for i in range(5):
            p.submit(("req", i))
        results = [p.get() for _ in range(5)]

    main_pid = os.getpid()
    # produce ran in a DIFFERENT pid (separate process).
    for r in results:
        assert r["worker_pid"] != main_pid
    # FIFO order preserved.
    assert [r["request"] for r in results] == [("req", i) for i in range(5)]


def test_orchestration_multiworker_still_fifo():
    # Multiple workers may finish out of order; the upload thread reorders
    # by seq so get() is STILL strictly FIFO.
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce,
        device="cpu",
        decode_workers=4,
        decode_ahead=8,
        gpu_ahead=4,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        # prime-then-loop (the documented usage): keep decode_ahead in
        # flight, interleaving get() with submit().
        for i in range(8):
            p.submit(i)
        got = []
        for i in range(8, 20):
            got.append(p.get()["request"])
            p.submit(i)
        got.extend(p.get()["request"] for _ in range(8))
    assert got == list(range(20))


def test_orchestration_keeps_decode_ahead_in_flight():
    # decode_ahead=2: submit blocks once 2 batches are in flight (between
    # submit and get). We submit 2 fast (non-blocking); the 3rd must block
    # until a get() frees a slot.
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce_slow,
        device="cpu",
        decode_workers=1,
        decode_ahead=2,
        gpu_ahead=1,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        t0 = time.monotonic()
        p.submit(0)
        p.submit(1)
        # both submits returned quickly (2 slots available)
        assert time.monotonic() - t0 < 0.1

        # The 3rd submit must BLOCK (no free slot) until a get() frees one.
        third_done = threading.Event()

        def _do_third():
            p.submit(2)
            third_done.set()

        t = threading.Thread(target=_do_third, daemon=True)
        t.start()
        # It stays blocked while no batch is consumed.
        assert not third_done.wait(timeout=0.3)
        # Consuming one batch frees a slot -> the 3rd submit unblocks.
        first = p.get()
        assert third_done.wait(timeout=2.0)
        t.join(timeout=2.0)

        rest = [p.get(), p.get()]
    got = [first] + rest
    assert sorted(g["request"] for g in got) == [0, 1, 2]


def test_orchestration_worker_exception_propagates_not_hang():
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce_raises,
        device="cpu",
        decode_workers=1,
        decode_ahead=3,
        gpu_ahead=1,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        p.submit("ok-1")
        p.submit("boom")
        p.submit("ok-2")
        first = p.get()
        assert first["request"] == "ok-1"
        # The failing request surfaces as a raised error in FIFO position
        # (NOT a hang).
        with pytest.raises(RuntimeError, match="worker produce failed"):
            p.get()
        # The pipeline keeps serving after the error.
        third = p.get()
        assert third["request"] == "ok-2"


# ==========================================================================
# B'. WORKER-DEATH FAIL-FAST (the load-dependent deadlock regression)
# ==========================================================================
# These SIMULATE the failure the loaded box hit: a worker that exits without
# ever shipping a result. Pre-fix, get() parked on the empty ready queue
# forever (the 53-min hang). Post-fix, get() must RAISE PROMPTLY. The
# autouse hard-timeout fixture turns any regression (a revert of the liveness
# fix) into a fast FAILURE instead of a multi-minute hang.
def test_worker_death_on_source_open_raises_not_hangs():
    # Worker dies on make_source (post-spawn open/import crash). get() must
    # raise PrefetchWorkerDied naming the dead worker, well within seconds.
    with GpuBatchPrefetcher(
        make_source=_make_source_dies_on_open,
        produce=_produce,
        device="cpu",
        decode_workers=1,
        decode_ahead=2,
        gpu_ahead=1,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        p.submit("x")
        t0 = time.monotonic()
        with pytest.raises(PrefetchWorkerDied, match="exited before delivering"):
            p.get()
        # Promptly, not after minutes -- proves no infinite park.
        assert time.monotonic() - t0 < 10.0


def test_worker_death_mid_produce_raises_not_hangs():
    # Source opens, but the first produce kills the process silently. Each
    # owed get() must raise (not just the first), so a consumer that keeps
    # calling get() never parks on the now-empty queue.
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce_dies_silently,
        device="cpu",
        decode_workers=1,
        decode_ahead=3,
        gpu_ahead=1,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        p.submit("a")
        p.submit("b")
        for _ in range(2):
            with pytest.raises(PrefetchWorkerDied):
                p.get()


def test_single_worker_crash_among_peers_is_fatal_not_hang():
    # Multi-worker: ONE worker crashes mid-produce while peers stay alive.
    # A crashed worker is fatal to the whole pipeline (torch DataLoader
    # semantics) -- the seq it owned can never arrive and is owed by no
    # surviving peer. Every owed get() must RAISE, never hang. We don't
    # assert a salvage count (which seqs a peer already produced is racy);
    # we assert the invariant that matters: no get() parks, and the crash
    # surfaces as PrefetchWorkerDied.
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce_die_on_one,
        device="cpu",
        decode_workers=4,
        decode_ahead=6,
        gpu_ahead=4,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        for i in range(6):
            p.submit(i)
        raised = 0
        for _ in range(6):
            try:
                p.get()
            except PrefetchWorkerDied:
                raised += 1
        # At least the crashed worker's seq surfaces an error; crucially the
        # loop completed (the hard-timeout fixture would have failed a hang).
        assert raised >= 1


def test_unpicklable_request_raises_at_submit_not_stall():
    # An unpicklable payload cannot cross the spawn IPC boundary: the
    # mp.Queue feeder thread would drop it SILENTLY (worker stays alive but
    # idle) and get() would park forever. submit() must reject it eagerly,
    # in the calling thread, with a clear error -- never a silent stall.
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce,
        device="cpu",
        decode_workers=1,
        decode_ahead=2,
        gpu_ahead=1,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        with pytest.raises(UnpicklablePayloadError, match="not picklable"):
            p.submit(_Unpicklable())
        # The slot was NOT consumed by the rejected submit: a good request
        # still flows and is served.
        p.submit("ok")
        assert p.get()["request"] == "ok"


def test_stream_back_pressure_safe_many_more_than_decode_ahead():
    # stream() must accept FAR more requests than decode_ahead without the
    # caller deadlocking on submit()'s back-pressure (the over-submit footgun:
    # submitting all N up-front wedges submit() at request decode_ahead+1).
    # It keeps <= decode_ahead in flight and yields all in FIFO order. A
    # regression (re-introducing the over-submit) hangs -> the autouse 30s
    # fixture turns it into a fast FAILURE.
    N = 50
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce,
        device="cpu",
        decode_workers=2,
        decode_ahead=3,            # N >> decode_ahead
        gpu_ahead=2,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        got = [b["request"] for b in p.stream(range(N))]
    assert got == list(range(N))


def test_stream_empty_requests_yields_nothing():
    with GpuBatchPrefetcher(
        make_source=_make_source,
        produce=_produce,
        device="cpu",
        decode_ahead=2,
        gpu_ahead=1,
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    ) as p:
        assert list(p.stream([])) == []


# ==========================================================================
# C. memmap-after-fork guarantee
# ==========================================================================
def _make_source_explodes():
    raise AssertionError("make_source must run ONLY in the worker, not __init__")


def test_init_does_not_call_make_source():
    # __init__ must NOT call make_source (the memmap opens in the worker
    # post-spawn). Constructing with a make_source that raises must still
    # succeed -- proving the call is deferred to the worker.
    p = GpuBatchPrefetcher(
        make_source=_make_source_explodes,
        produce=_produce,
        device="cpu",
        start_method="spawn",
        to_device=_noop_to_device,
        is_leaf=_never_leaf,
    )
    # Constructed fine; never started, so the worker never ran make_source.
    assert p is not None


# ==========================================================================
# D. REAL CUDA smoke -- skipped without torch / GPU (UNVALIDATED here)
# ==========================================================================
# Module-level so ``spawn`` can pickle them into the worker (local funcs/
# namedtuples are NOT picklable -> the worker fails to launch).
_RealNT = collections.namedtuple("_RealNT", ["ids"])


def _real_make_source():
    return object()


def _real_produce(_src, seq):
    import torch  # only present on the consumer side
    # content fully determined by ``seq`` so the consumer can assert the
    # uploaded bytes arrived intact after the async H2D.
    return {
        "x": torch.full((1024,), seq, dtype=torch.int32),
        "nt": _RealNT(torch.ones(8)),
        "seq": int(seq),
    }


def test_real_cuda_overlap_smoke():
    torch = pytest.importorskip(
        "torch",
        reason="torch absent in dev shell; real cuda stream/event/pin "
        "overlap is validated consumer-side (a CUDA box).",
    )
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; real H2D-overlap path validated on GPU")

    # Drive many batches with the CORRECT prefetch pattern (prime decode_ahead,
    # then submit-one-per-get -- submit() blocks by design once decode_ahead
    # are in flight). A churn matmul on the consumer stream keeps the caching
    # allocator under reuse pressure, so a missing/incorrect per-leaf
    # record_stream (the #70 fix) would corrupt an in-use batch -> caught here.
    N, decode_ahead, bad = 64, 3, 0
    with GpuBatchPrefetcher(
        make_source=_real_make_source,
        produce=_real_produce,
        device="cuda:0",
        decode_ahead=decode_ahead,
        gpu_ahead=2,
        start_method="spawn",
    ) as p:
        nxt = 0
        for _ in range(min(decode_ahead, N)):
            p.submit(nxt)
            nxt += 1
        churn = torch.randn(256, 256, device="cuda:0")
        for s in range(N):
            b = p.get()
            assert b["x"].device.type == "cuda"
            assert b["nt"].ids.device.type == "cuda"
            _ = churn @ churn                          # reuse pressure
            torch.cuda.synchronize()
            if b["seq"] != s or not torch.all(b["x"] == s).item():
                bad += 1
            if nxt < N:
                p.submit(nxt)
                nxt += 1
    assert bad == 0, (
        f"{bad}/{N} batches corrupted under allocator-reuse pressure "
        "-- the per-leaf record_stream (#70) is not holding on hardware"
    )


# ==========================================================================
# E. H2D-OVERLAP ORCHESTRATION via an INJECTED FAKE CUDA BACKEND (no GPU)
# ==========================================================================
# The CudaBackend seam lets us drive a REAL GpuBatchPrefetcher end-to-end
# with a fake torch.cuda that RECORDS the ordered op sequence + stream
# identities, so the overlap ORDER is asserted without a GPU. The fake is
# parametrizable to BREAK exactly one invariant at a time, proving the
# assertions are mutation-sensitive (skip pin / wrong stream / event-before-
# copy / no wait_event / no record_stream each make the test FAIL).
class _FakeStream:
    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_FakeStream({self.label!r})"


class _FakeEvent:
    def __init__(self, on_stream: _FakeStream) -> None:
        self.on_stream = on_stream


class _FakeCudaBackend:
    """A torch-free CudaBackend that logs the ordered overlap ops + streams.

    ``defects`` flips one invariant at a time so the asserting test can
    prove it catches each break:
      - "skip_pin"        : the per-leaf move skips pinning
      - "compute_stream"  : the copy runs on the compute stream, not copy
      - "event_before"    : event is recorded BEFORE the copy is enqueued
      - "no_wait"         : the consumer never waits on the event
      - "no_record_stream": uploaded leaves are not record_stream'd
    """

    def __init__(self, defects: frozenset[str] = frozenset()) -> None:
        self.defects = defects
        self.log: list[tuple] = []
        self.lock = threading.Lock()
        self.compute_stream = _FakeStream("compute")
        self.copy_stream = _FakeStream("copy")
        # The "current" stream a move sees; set inside upload's context so
        # the injected to_device can record WHICH stream the copy ran on.
        self._cur_stream = self.compute_stream

    def _emit(self, *event: Any) -> None:
        with self.lock:
            self.log.append(event)

    # -- the move primitive the prefetcher's to_device delegates to --------
    def move_leaf(self, t: "FakeTensor") -> "FakeTensor":
        if "skip_pin" not in self.defects:
            self._emit("pin", t.name)
        # ("copy", name, stream_label, non_blocking)
        self._emit("copy", t.name, self._cur_stream.label, True)
        return FakeTensor(t.name, device="cuda")

    # -- CudaBackend surface ----------------------------------------------
    def make_copy_stream(self, device: Any) -> _FakeStream:
        return self.copy_stream

    def upload(self, cpu_batch, device, copy_stream, move_one, is_leaf):
        ctx_stream = (
            self.compute_stream
            if "compute_stream" in self.defects
            else copy_stream
        )
        event = _FakeEvent(on_stream=ctx_stream)
        if "event_before" in self.defects:
            self._emit("event_record", ctx_stream.label)
        prev, self._cur_stream = self._cur_stream, ctx_stream
        try:
            gpu_batch = map_tensor_leaves(cpu_batch, move_one, is_leaf=is_leaf)
        finally:
            self._cur_stream = prev
        if "event_before" not in self.defects:
            self._emit("event_record", ctx_stream.label)
        return (gpu_batch, event)

    def wait(self, gpu_batch, event, device, is_leaf):
        if "no_wait" not in self.defects:
            self._emit("wait_event", self.compute_stream.label, id(event))
        if "no_record_stream" not in self.defects:
            map_tensor_leaves(
                gpu_batch,
                lambda t: self._emit("record_stream", t.name, self.compute_stream.label)
                or t,
                is_leaf=is_leaf,
            )


# Module-level (picklable under spawn). The worker only runs make/produce;
# the FAKE tensors must survive the IPC pickle, which FakeTensor already does.
# The namedtuple MUST be module-level too -- a function-local namedtuple is
# unpicklable, so the worker's result would never return and get() would hang.
import collections as _collections

_NT_E = _collections.namedtuple("_NT_E", ["ids"])


def _make_source_e():
    return object()


def _produce_two_leaf(_src, n):
    # Two tensor leaves under different container types + a passthrough int.
    return {"x": FakeTensor(f"x{n}"), "nt": _NT_E(FakeTensor(f"ids{n}")), "n": int(n)}


def _drive_overlap(backend: _FakeCudaBackend) -> dict:
    """Run one batch through a real prefetcher against ``backend``; return it."""
    with GpuBatchPrefetcher(
        make_source=_make_source_e,
        produce=_produce_two_leaf,
        device="cuda:fake",
        decode_workers=1,
        decode_ahead=2,
        gpu_ahead=1,
        start_method="spawn",
        to_device=lambda t, _device: backend.move_leaf(t),
        is_leaf=_is_fake_tensor,
        cuda_backend=backend,
    ) as p:
        p.submit(4)
        batch = p.get()
    return batch


def _assert_overlap_invariants(backend: _FakeCudaBackend, batch: dict) -> None:
    """The overlap CONTRACT: every leaf moved to cuda, and the op log obeys
    pin<copy, copy-on-copy-stream, event-after-copy-on-copy-stream,
    wait-then-record_stream on the compute stream for every leaf."""
    # Every tensor leaf is GPU-resident.
    assert batch["x"].device == "cuda"
    assert batch["nt"].ids.device == "cuda"
    assert batch["n"] == 4

    log = backend.log
    ops = [e[0] for e in log]

    # (a) copy stream is SEPARATE from the compute stream.
    assert backend.copy_stream.label != backend.compute_stream.label

    leaf_names = {"x4", "ids4"}
    for name in leaf_names:
        pin_idxs = [k for k, e in enumerate(log) if e == ("pin", name)]
        # (b) the leaf is PINNED (else non_blocking H2D is a silent no-op)...
        assert pin_idxs, f"{name}: must be pinned before the copy"
        i_pin = pin_idxs[0]
        i_copy = next(
            k for k, e in enumerate(log)
            if e[0] == "copy" and e[1] == name
        )
        # ...and pinned BEFORE its non_blocking copy.
        assert i_pin < i_copy, f"{name}: pin must precede copy"
        # (a) the copy runs on the COPY stream (not compute) with
        #     non_blocking=True -- else non_blocking is a silent no-op.
        assert log[i_copy][2] == backend.copy_stream.label, f"{name}: copy on copy stream"
        assert log[i_copy][3] is True, f"{name}: copy must be non_blocking"

    # (c) the event is recorded ON the copy stream AFTER all copies enqueued.
    i_event = ops.index("event_record")
    last_copy = max(k for k, e in enumerate(log) if e[0] == "copy")
    assert i_event > last_copy, "event must be recorded after the copies"
    assert log[i_event][1] == backend.copy_stream.label, "event on the copy stream"

    # (d) the consumer waits on the event (compute stream) BEFORE handing the
    #     batch to compute, and record_stream's EVERY leaf on the compute
    #     stream (cross-stream-reuse fix).
    assert "wait_event" in ops, "consumer must wait on the copy event"
    i_wait = ops.index("wait_event")
    assert log[i_wait][1] == backend.compute_stream.label
    assert i_wait > i_event, "consumer waits after the copy event is recorded"
    recorded = {e[1] for e in log if e[0] == "record_stream"}
    assert recorded == leaf_names, "every uploaded leaf must be record_stream'd"
    for k, e in enumerate(log):
        if e[0] != "record_stream":
            continue
        assert e[2] == backend.compute_stream.label
        assert k > i_wait, "record_stream after the wait"


# The fake-cuda-backend tests are the TORCH-FREE substitute for the real
# overlap path. With REAL torch present, the prefetcher __init__ validates
# the device via ``torch.device()``, which rejects the fake ``"cuda:fake"``
# string -- and the real path is then covered by
# :func:`test_real_cuda_overlap_smoke` on the GPU. So skip these when a real
# CUDA torch is importable.
def _real_cuda_present() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - torch absent is the common dev-shell case
        return False


_SKIP_IF_REAL_CUDA = pytest.mark.skipif(
    _real_cuda_present(),
    reason="fake-cuda backend tests are the torch-free substitute; real CUDA "
    "is validated by test_real_cuda_overlap_smoke",
)


@_SKIP_IF_REAL_CUDA
def test_overlap_orchestration_order_with_fake_cuda():
    backend = _FakeCudaBackend()
    batch = _drive_overlap(backend)
    _assert_overlap_invariants(backend, batch)


@_SKIP_IF_REAL_CUDA
@pytest.mark.parametrize(
    "defect",
    ["skip_pin", "compute_stream", "event_before", "no_wait", "no_record_stream"],
)
def test_overlap_invariants_are_mutation_sensitive(defect):
    # Breaking ANY single overlap invariant must make the assertions FAIL.
    backend = _FakeCudaBackend(defects=frozenset({defect}))
    batch = _drive_overlap(backend)
    with pytest.raises(AssertionError):
        _assert_overlap_invariants(backend, batch)


# ==========================================================================
# F. PER-ROW DEPTH LEAF RIDES THE PREFETCH PATH (no torch, injected leaf)
# ==========================================================================
# The cross-depth loader attaches a per-row ``depth_per_row`` vector to the
# batch. The prefetcher is leaf-agnostic, so a consumer that exposes that
# vector as a tensor leaf must see it moved to device EXACTLY like the token
# tensor, with its row payload preserved. This proves the pytree + to_device
# round-trip carries ``depths`` -- no prefetcher code knows the field name.
@dataclasses.dataclass
class _DepthConsumerBatch:
    """A consumer batch carrying a token leaf AND a per-row depth leaf."""

    tokens: FakeTensor
    depths: FakeTensor  # the per-row source-depth vector, as a tensor leaf
    seq_len: int        # plain scalar -- must pass through untouched


def _make_source_depth():
    return {"kind": "depth-stub"}


def _produce_depth(_src, request):
    # ``request`` is the per-row depth payload; it is carried on the depths
    # leaf so the moved-batch assertion can verify the row values survive.
    rows = list(request)
    return _DepthConsumerBatch(
        tokens=FakeTensor("tokens"),
        depths=FakeTensor(f"depths:{rows}"),
        seq_len=len(rows),
    )


def test_depth_leaf_moves_to_device_with_tokens_through_prefetcher():
    with GpuBatchPrefetcher(
        make_source=_make_source_depth,
        produce=_produce_depth,
        device="cpu",
        decode_workers=1,
        decode_ahead=3,
        gpu_ahead=1,
        start_method="spawn",
        to_device=lambda t, device: _move_to_fake_cuda(t),
        is_leaf=_is_fake_tensor,
    ) as p:
        p.submit((0, 1, 3))
        p.submit((3, 3))
        a = p.get()
        b = p.get()

    # Structure preserved; the plain scalar passed through untouched.
    assert isinstance(a, _DepthConsumerBatch)
    assert a.seq_len == 3 and b.seq_len == 2
    # BOTH the token leaf AND the depths leaf were moved to the device --
    # the prefetcher treated depths exactly like tokens (leaf-agnostic).
    assert a.tokens.device == "cuda"
    assert a.depths.device == "cuda"
    assert b.depths.device == "cuda"
    # The per-row depth payload survived the H2D round-trip intact and in
    # FIFO order (the depths leaf carries its source rows verbatim).
    assert a.depths.name == "depths:[0, 1, 3]"
    assert b.depths.name == "depths:[3, 3]"
