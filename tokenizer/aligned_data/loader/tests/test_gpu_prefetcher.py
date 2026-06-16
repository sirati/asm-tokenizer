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

import dataclasses
import os
import threading
import time
from typing import Any

import pytest

from tokenizer.aligned_data.loader.gpu_prefetcher import GpuBatchPrefetcher
from tokenizer.aligned_data.loader.gpu_prefetcher._pytree import map_tensor_leaves


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
def test_real_cuda_overlap_smoke():
    torch = pytest.importorskip(
        "torch",
        reason="torch absent in dev shell; real cuda stream/event/pin "
        "overlap is UNVALIDATED here and must be validated consumer-side.",
    )
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; real H2D-overlap path unvalidated here")

    import collections

    NT = collections.namedtuple("NT", ["ids"])

    def make_source():
        return object()

    def produce(_src, n):
        return {"x": torch.arange(n), "nt": NT(torch.ones(n)), "n": int(n)}

    with GpuBatchPrefetcher(
        make_source=make_source,
        produce=produce,
        device="cuda:0",
        decode_ahead=2,
        gpu_ahead=1,
        start_method="spawn",
    ) as p:
        p.submit(4)
        p.submit(8)
        b0 = p.get()
        assert b0["x"].device.type == "cuda"
        assert b0["nt"].ids.device.type == "cuda"
        assert b0["n"] == 4


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


def test_overlap_orchestration_order_with_fake_cuda():
    backend = _FakeCudaBackend()
    batch = _drive_overlap(backend)
    _assert_overlap_invariants(backend, batch)


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
