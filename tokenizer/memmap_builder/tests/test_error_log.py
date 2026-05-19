"""Unit tests for the ``<binary>.error.log`` TSV writer chokepoint."""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone

import pytest

from tokenizer.memmap_builder.error_log import (
    ALLOWED_REASONS,
    write_error_log_entry,
)


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_round_trip_all_allowed_reasons() -> None:
    """Writing one row per allowed reason produces a well-formed TSV.

    Each row has 4 tab-separated columns, ends with ``\\n``, starts
    with the matching reason, and carries a UTC ISO-8601 timestamp
    within 5 seconds of "now".
    """
    sio = io.StringIO()
    before = datetime.now(timezone.utc)
    reasons = sorted(ALLOWED_REASONS)
    for reason in reasons:
        write_error_log_entry(sio, reason, f"fn_for_{reason}", 12345)
    after = datetime.now(timezone.utc)

    raw = sio.getvalue()
    lines = raw.splitlines(keepends=True)
    assert len(lines) == len(reasons)

    for reason, line in zip(reasons, lines):
        assert line.endswith("\n"), line
        cols = line.rstrip("\n").split("\t")
        assert len(cols) == 4, cols
        assert cols[0] == reason
        assert cols[1] == f"fn_for_{reason}"
        assert cols[2] == "12345"
        ts_field = cols[3]
        assert _ISO_RE.match(ts_field), ts_field
        parsed = datetime.fromisoformat(ts_field.rstrip("Z")).replace(
            tzinfo=timezone.utc
        )
        # 5s either side of the bracketed wall clock interval.
        assert before - timedelta(seconds=5) <= parsed <= after + timedelta(seconds=5)


def test_missing_handle_is_noop() -> None:
    """``error_log is None`` must return ``None`` without raising."""
    result = write_error_log_entry(None, "insn_len_overflow", "fn", 16777216)
    assert result is None


def test_invalid_reason_rejected() -> None:
    """An unknown reason must raise ``ValueError`` mentioning it.

    Validation happens before the handle is touched, so passing a real
    sink still results in nothing being written when the call raises.
    """
    sio = io.StringIO()
    with pytest.raises(ValueError, match="bogus_reason"):
        write_error_log_entry(sio, "bogus_reason", "fn", 0)
    assert sio.getvalue() == ""


def test_allowed_reasons_set_is_exhaustive() -> None:
    """Guard against silent reason-set drift relative to the design.

    The plan enumerates exactly these four overflow reasons; if more
    are added the writer's surface MUST be revisited at the same time.
    """
    assert ALLOWED_REASONS == frozenset(
        {
            "insn_len_overflow",
            "block_len_overflow",
            "overlong_length_overflow",
            "offset_overflow",
        }
    )
