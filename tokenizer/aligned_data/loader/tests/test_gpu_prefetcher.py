"""Torch-less tests for the generic GPU batch prefetcher.

torch is ABSENT from the dev shell, so coverage is HONEST about its
scope: these tests exercise the GENERIC PYTREE MOVE (A), the
multi-process ORCHESTRATION + FIFO + exception propagation (B), and the
memmap-after-fork guarantee (C) via DEPENDENCY INJECTION (an injected
``to_device`` + ``is_leaf``). The REAL cuda overlap -- copy stream, event
record/wait, pinned async H2D -- is UNVALIDATED here and MUST be
validated by the consumer on their GPU (test D, skipped without torch).
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
