"""Cadence + content tests for ``tokenizer.progress.log_progress``.

Drives the iterator with a monkeypatched ``time.monotonic`` so the
report schedule (start, first after 10 s, then every 60 s, final) is
asserted deterministically — no sleeping.
"""

import logging

import pytest

from tokenizer import progress
from tokenizer.progress import log_progress


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, secs: float) -> None:
        self.now += secs


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(progress.time, "monotonic", fake)
    return fake


def _records(caplog):
    return [r.getMessage() for r in caplog.records]


def test_items_pass_through_unchanged(clock, caplog):
    logger = logging.getLogger("test_progress")
    with caplog.at_level(logging.INFO, logger="test_progress"):
        out = list(log_progress(iter([1, 2, 3]), desc="d", logger=logger, total=3))
    assert out == [1, 2, 3]


def test_report_schedule_start_10s_then_60s(clock, caplog):
    logger = logging.getLogger("test_progress")

    def items():
        # 1 item per second for 200 seconds.
        for i in range(200):
            clock.advance(1.0)
            yield i

    with caplog.at_level(logging.INFO, logger="test_progress"):
        for _ in log_progress(items(), desc="d", logger=logger, total=200, unit="functions"):
            pass

    msgs = _records(caplog)
    # start line + reports at ~10s, ~70s, ~130s, ~190s + final line
    assert msgs[0] == "d: 0/200 functions"
    assert msgs[1] == "d: 10/200 functions (+10, 1.0/s)"
    assert msgs[2] == "d: 70/200 functions (+60, 1.0/s)"
    assert msgs[3] == "d: 130/200 functions (+60, 1.0/s)"
    assert msgs[4] == "d: 190/200 functions (+60, 1.0/s)"
    assert msgs[5].startswith("d: finished 200/200 functions in 200s")
    assert len(msgs) == 6


def test_delta_and_rate_reflect_window(clock, caplog):
    logger = logging.getLogger("test_progress")

    def items():
        # 5 items in the first 10s (one per 2s), then 120 items in the
        # next 60s (two per second).
        for i in range(5):
            clock.advance(2.0)
            yield i
        for i in range(120):
            clock.advance(0.5)
            yield i

    with caplog.at_level(logging.INFO, logger="test_progress"):
        for _ in log_progress(items(), desc="d", logger=logger, total=125):
            pass

    msgs = _records(caplog)
    assert msgs[1] == "d: 5/125 functions (+5, 0.5/s)"
    assert msgs[2] == "d: 125/125 functions (+120, 2.0/s)"


def test_unknown_total_renders_question_mark(clock, caplog):
    logger = logging.getLogger("test_progress")
    with caplog.at_level(logging.INFO, logger="test_progress"):
        list(log_progress(iter([1]), desc="d", logger=logger))
    msgs = _records(caplog)
    assert msgs[0] == "d: 0/? functions"
    assert "finished 1/? functions" in msgs[-1]


def test_final_line_emitted_on_consumer_error(clock, caplog):
    logger = logging.getLogger("test_progress")

    def items():
        clock.advance(1.0)
        yield 1
        clock.advance(1.0)
        yield 2

    with caplog.at_level(logging.INFO, logger="test_progress"):
        gen = log_progress(items(), desc="d", logger=logger, total=2)
        next(gen)
        gen.close()  # consumer abandons mid-iteration

    msgs = _records(caplog)
    assert any(m.startswith("d: finished 1/2 functions") for m in msgs)
