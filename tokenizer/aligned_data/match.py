"""Low-level CSV-scanning primitives shared by the parsed-record iterator
and the validator.

Single concern: open a per-version tokenizer-output CSV, consume the
optional ``version=...`` prelude row, and yield raw row lists (vocab
rows filtered out). Higher-level layers (per-CSV parsing in
:mod:`tokenizer.aligned_data.parsed_record_iter`; lockstep merge across
N CSVs in :mod:`tokenizer.aligned_data.lockstep`) build on these
primitives.
"""

import csv
import os
import sys
from typing import List

# Real-corpus per-binary CSVs carry full-function ``tokens_base64`` cells
# that routinely exceed the default 131072-byte field limit (large
# functions in nmap / openssl / clamav). Raise to ``sys.maxsize`` so the
# csv.reader doesn't reject those rows. Module-load is the right hook —
# every csv.reader in this package is constructed lazily after this
# point.
csv.field_size_limit(sys.maxsize)


class PositionTrackingWrapper:
    """Wrapper that tracks bytes read without interfering with csv.reader buffering."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fd = os.open(file_path, os.O_RDONLY)
        self.file = open(self.fd, newline="", encoding="ascii")
        self.bytes_read = 0

    def get_position(self) -> int:
        """Get current file position using lseek."""
        return os.lseek(self.fd, 0, os.SEEK_CUR)

    def close(self):
        self.file.close()


def is_vocab_row(row: List[str]) -> bool:
    # Vocab row: field0 == 'vocabulary' and field1 does not start with a digit
    return row[0] == "vocabulary" and (not row[1] or not row[1][0].isdigit())


def is_version_prelude_row(row: List[str]) -> bool:
    # v2 prelude: a single-cell row whose only field starts with "version=".
    # Written by tokenizer/main_loop.py before the header. v1 files lack
    # this row entirely (their first row is the header itself).
    return len(row) == 1 and row[0].startswith("version=")


def open_csv_skip_vocab(path: str):
    """Open ``path`` and return ``(wrapper, raw_row_iterator, header)``.

    Consumes the optional v2 ``version=...`` prelude row. The returned
    iterator yields raw row lists (vocab rows filtered out). The
    ``wrapper`` exposes ``get_position()`` for progress reporting; the
    caller owns its lifecycle via ``wrapper.close()``.
    """
    wrapper = PositionTrackingWrapper(path)
    reader = csv.reader(wrapper.file)
    first = next(reader)
    if is_version_prelude_row(first):
        header = next(reader)
    else:
        header = first

    def row_iterator():
        for row in reader:
            if not is_vocab_row(row):
                yield row

    return wrapper, row_iterator(), header
