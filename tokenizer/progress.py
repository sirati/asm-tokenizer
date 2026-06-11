"""Log-friendly progress reporting for long iterations.

Single concern: own the policy "report iteration progress to a logger
at a bounded cadence" as a drop-in replacement for a TTY progress bar
in worker contexts. Workers' stdout/stderr are captured into log
files, where a carriage-return-rendered bar (tqdm) lands as thousands
of spam lines; this wrapper emits a handful of plain log lines
instead:

* one report when iteration starts,
* one report ~10 s in (early confirmation the loop is moving),
* one report every 60 s thereafter,
* one final report when iteration ends (also on error, so the log
  records where the loop stopped).

Each periodic line carries ``current/total``, the delta since the
previous report, and the rate over that window in items/s.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

_FIRST_REPORT_AFTER_SECS = 10.0
_REPORT_EVERY_SECS = 60.0


def log_progress(
    iterable: Iterable[T],
    *,
    desc: str,
    logger: logging.Logger,
    total: Optional[int] = None,
    unit: str = "functions",
) -> Iterator[T]:
    """Yield items from ``iterable`` unchanged, logging progress at the
    module cadence. ``total`` is display-only (``?`` when unknown).
    """
    start = time.monotonic()
    total_str = str(total) if total is not None else "?"
    count = 0
    last_count = 0
    last_report = start
    next_due = start + _FIRST_REPORT_AFTER_SECS

    logger.info(f"{desc}: 0/{total_str} {unit}")
    try:
        for item in iterable:
            count += 1
            yield item
            now = time.monotonic()
            if now >= next_due:
                window = now - last_report
                delta = count - last_count
                rate = delta / window if window > 0 else 0.0
                logger.info(
                    f"{desc}: {count}/{total_str} {unit} (+{delta}, {rate:.1f}/s)"
                )
                last_count = count
                last_report = now
                next_due = now + _REPORT_EVERY_SECS
    finally:
        elapsed = time.monotonic() - start
        avg = count / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"{desc}: finished {count}/{total_str} {unit} "
            f"in {elapsed:.0f}s (avg {avg:.1f}/s)"
        )


@contextmanager
def log_stage(logger: logging.Logger, desc: str) -> Iterator[None]:
    """Log ``desc: starting`` on entry and ``desc: done in X.Xs`` on
    clean exit (``desc: failed after X.Xs`` when the block raises, so
    the log records where a run stopped).

    Companion to ``log_progress`` for non-iterating long stages —
    analysis passes, O(binary) scans, bulk I/O — which would otherwise
    leave the log silent for their whole duration.
    """
    start = time.monotonic()
    logger.info(f"{desc}: starting")
    try:
        yield
    except BaseException:
        logger.info(f"{desc}: failed after {time.monotonic() - start:.1f}s")
        raise
    logger.info(f"{desc}: done in {time.monotonic() - start:.1f}s")
