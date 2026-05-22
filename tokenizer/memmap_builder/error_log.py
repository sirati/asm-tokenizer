"""Single chokepoint for the per-binary ``<binary>.error.log`` TSV rows.

Centralising the row format here keeps the format detail from leaking
into the call sites that translate an encoder-overflow event into a
"skipped function" log entry; callers pass plain ``str`` / ``int`` and
this module owns the column layout, timestamp form, and the
no-op-on-missing-handle contract.
"""

from __future__ import annotations

from datetime import datetime, timezone

ALLOWED_REASONS: frozenset[str] = frozenset(
    {
        "insn_len_overflow",
        "block_word_count_overflow",
        "token_count_overflow",
        "offset_overflow",
        "bin_offset_overflow",
        "bin_length_overflow",
    }
)


def write_error_log_entry(error_log, reason: str, func_name: str, value: int) -> None:
    """Write one TSV row to ``<binary>.error.log``.

    The row format is::

        <reason>\\t<func_name>\\t<value>\\t<iso_timestamp>\\n

    where ``iso_timestamp`` is the current UTC time formatted as
    ``YYYY-MM-DDTHH:MM:SSZ`` (ISO 8601, second precision).

    Passing ``error_log is None`` is a no-op so callers (e.g. dry-run
    paths that never opened the file) can call unconditionally without
    a branch on their side.
    """
    if reason not in ALLOWED_REASONS:
        raise ValueError(f"unknown error_log reason: {reason!r}")
    if error_log is None:
        return
    iso_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    error_log.write(f"{reason}\t{func_name}\t{value}\t{iso_timestamp}\n")
