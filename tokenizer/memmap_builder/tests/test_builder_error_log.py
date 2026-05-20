"""Builder-level error.log plumbing test.

Drives the writer-chain directly (bypassing the full corpus walk so the
test runs in seconds rather than minutes): opens an in-memory data file
+ an in-memory error_log, trips the ``insn_len`` cap, and asserts:

  * cap overflow (``insn_len >= NORMAL_INSN_CAP``) logs one TSV row into
    ``error_log``, the partial data write is truncated, and the
    corresponding ``write_index_entry`` is skipped — the index file
    stays empty.

The overlong + sentinel paths are gone in the self-describing record
layout (record header carries its own geometry); the sole producer-side
cap-handling contract worth pinning here is the log-and-skip behavior.
"""

from __future__ import annotations

import io as stdio

import numpy as np

from tokenizer.aligned_data.io import (
    write_function_binary_data,
    write_index_entry,
)
from tokenizer.memmap_builder.error_log import ALLOWED_REASONS


# The cap the encoder enforces on the per-record u24 ``insn_len`` field.
_INSN_LEN_CAP = 1 << 24


def _zero_block_uint8(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.uint8)


def test_insn_len_overflow_logs_and_skips_index_entry() -> None:
    """An insn_runlength of >= NORMAL_INSN_CAP trips ``insn_len_overflow``.

    The writer truncates the partial data write to zero bytes and
    returns ``None``. The caller (test stand-in for pass-1) honours
    the ``None`` by not calling ``write_index_entry`` — the index file
    stays empty.
    """
    data_buf = stdio.BytesIO()
    error_log = stdio.StringIO()
    index_buf = stdio.BytesIO()

    insn = np.zeros(_INSN_LEN_CAP, dtype=np.uint8)
    block = _zero_block_uint8(4)
    tokens = np.zeros(2, dtype=np.uint16)

    write_result = write_function_binary_data(
        data_buf,
        tokens,
        block,
        insn,
        dedup_cache=None,
        func_name="overflow_fn",
        error_log=error_log,
    )

    assert write_result is None
    assert data_buf.getvalue() == b""

    # Caller side: None signals "skip". index file stays untouched.
    if write_result is not None:  # pragma: no cover - documenting the gate
        offset, _total = write_result
        write_index_entry(
            index_buf, offset, func_name="overflow_fn", error_log=error_log,
        )

    assert index_buf.getvalue() == b""

    log_lines = error_log.getvalue().splitlines()
    assert len(log_lines) == 1
    cols = log_lines[0].split("\t")
    assert cols[0] == "insn_len_overflow"
    assert cols[0] in ALLOWED_REASONS
    assert cols[1] == "overflow_fn"
    assert int(cols[2]) == _INSN_LEN_CAP
