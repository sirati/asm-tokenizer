import csv
from typing import List, Tuple, Dict, Iterator, Any

def is_vocab_row(row: List[str]) -> bool:
    # Vocab row: field0 == 'vocabulary' and field1 does not start with a digit
    return row[0] == 'vocabulary' and (not row[1] or not row[1][0].isdigit())

def open_csv_skip_vocab(path: str):
    f = open(path, newline='', encoding='ascii')
    reader = csv.reader(f)
    # Skip header
    header = next(reader)
    # Advance to first non-vocab row
    while True:
        pos = f.tell()
        try:
            row = next(reader)
        except StopIteration:
            return None, None, None
        if not is_vocab_row(row):
            f.seek(pos)
            break
    return f, reader, header

def lockstep_function_match(csv_paths: List[str]) -> Iterator[Tuple[str, List[List[str]]]]:
    """
    Yields (function_name, [row_per_version]) for all matched functions across all csv_paths.
    Only yields if all files have the same function_name at the current position, and the function passes the filters.
    Assumes function names are sorted in all files.
    """
    files = []
    readers = []
    headers = []
    current_rows = []
    for path in csv_paths:
        f, reader, header = open_csv_skip_vocab(path)
        if f is None:
            raise RuntimeError(f"Could not open or find data in {path}")
        files.append(f)
        readers.append(reader)
        headers.append(header)
        # Read first row
        while True:
            row = next(reader)
            if not is_vocab_row(row):
                break
        current_rows.append(row)
    while True:
        # Get current function names
        fnames = [row[0] for row in current_rows]
        # If any file is exhausted, stop
        if any(f is None for f in current_rows):
            break
        # If all names match and not .L-prefixed and block_runlength < 4096
        if all(f == fnames[0] for f in fnames) and not fnames[0].startswith('.L'):
            block_lengths = [row[3] for row in current_rows]
            try:
                import numpy as np
                from tokenizer.compact_base64_utils import base64_to_ndarray_vec
                if all(base64_to_ndarray_vec(bl).sum() < 4096 for bl in block_lengths):
                    yield (fnames[0], [row for row in current_rows])
            except Exception:
                pass
            # Advance all
            for i, reader in enumerate(readers):
                try:
                    while True:
                        row = next(reader)
                        if not is_vocab_row(row):
                            current_rows[i] = row
                            break
                except StopIteration:
                    current_rows[i] = None
        else:
            # Advance the file(s) with the smallest function name
            min_name = min(f for f in fnames if f is not None)
            for i, fname in enumerate(fnames):
                if fname == min_name:
                    try:
                        while True:
                            row = next(readers[i])
                            if not is_vocab_row(row):
                                current_rows[i] = row
                                break
                    except StopIteration:
                        current_rows[i] = None
    for f in files:
        f.close()

