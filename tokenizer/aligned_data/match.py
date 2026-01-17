import csv
import os
from typing import Any, Callable, Dict, List, Optional


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


def open_csv_skip_vocab(path: str):
    wrapper = PositionTrackingWrapper(path)
    reader = csv.reader(wrapper.file)
    # Skip header
    header = next(reader)

    # Create an iterator that filters out all vocab rows
    def row_iterator():
        for row in reader:
            if not is_vocab_row(row):
                yield row

    return wrapper, row_iterator(), header


def create_normalized_header(headers: List[List[str]]) -> List[str]:
    """Create normalized header: union of all headers maintaining order."""
    normalized_header = []
    seen_fields = set()
    for header in headers:
        for field in header:
            if field not in seen_fields:
                normalized_header.append(field)
                seen_fields.add(field)
    return normalized_header


def create_column_mapping(header: List[str], normalized_header: List[str]) -> Dict[str, Any]:
    """Create mapping from normalized header field to source row index."""
    header_to_idx = {field: idx for idx, field in enumerate(header)}
    return {field: header_to_idx.get(field) for field in normalized_header if field in header_to_idx}


def normalize_row(row: List[str], column_mapping: Dict[str, int]) -> Dict[str, str]:
    """Convert a row to a dict using pre-computed column mapping."""
    if row is None:
        return None
    return {field: row[idx] for field, idx in column_mapping.items()}


def lockstep_function_match(csv_paths: List[str], progress_callback: Optional[Callable[[int], None]] = None):
    """
    Yields dicts with 'function_name' and 'rows' keys for functions that appear in at least 2 files.
    Each dict's 'rows' value is a list where each element is either a dict (the CSV row as dict)
    or None if that version doesn't have the function.
    Only yields if at least 2 files have the same function_name, and the function passes the filters.
    Assumes function names are sorted in all files.
    Row dicts are normalized to have all headers from all files (with None for missing fields).

    Args:
        csv_paths: List of paths to CSV files to process
        progress_callback: Optional callback function that receives total bytes processed
    """
    wrappers = []
    readers = []
    headers = []
    current_rows = []
    for path in csv_paths:
        wrapper, reader, header = open_csv_skip_vocab(path)
        if wrapper is None:
            raise RuntimeError(f"Could not open or find data in {path}")
        wrappers.append(wrapper)
        readers.append(reader)
        headers.append(header)
        # Read first row (vocab rows already skipped by open_csv_skip_vocab)
        try:
            row = next(reader)
            current_rows.append(row)
        except StopIteration:
            current_rows.append(None)

    normalized_header = create_normalized_header(headers)
    column_mappings = [create_column_mapping(header, normalized_header) for header in headers]

    iteration_count = 0
    while True:
        iteration_count += 1
        # Get current function names (None if row is exhausted)
        fnames = [row[0] if row is not None else None for row in current_rows]
        # If all files are exhausted, stop
        if all(f is None for f in fnames):
            break

        # Find the minimum function name among non-None entries
        min_name = min(f for f in fnames if f is not None)

        # Check how many files have this minimum name
        matching_indices = [i for i, fname in enumerate(fnames) if fname == min_name]
        count = len(matching_indices)

        # Build result list with normalized row dicts
        result_rows = [
            normalize_row(current_rows[i], column_mappings[i]) if i in matching_indices else None
            for i in range(len(csv_paths))
        ]
        yield {"function_name": min_name, "rows": result_rows, "count": count}

        # Advance the file(s) with the minimum function name
        for i in matching_indices:
            try:
                row = next(readers[i])
                current_rows[i] = row
            except StopIteration:
                current_rows[i] = None

        if progress_callback is not None and iteration_count % 100 == 0:
            total_bytes = sum(wrapper.get_position() for wrapper in wrappers)
            progress_callback(total_bytes)

    if progress_callback is not None:
        total_bytes = sum(wrapper.get_position() for wrapper in wrappers)
        progress_callback(total_bytes)

    for wrapper in wrappers:
        wrapper.close()
