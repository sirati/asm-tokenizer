import numpy as np
import csv


def read_function_section(file1_path, start, length):
    """
    Read a section from the CSV sections file (file 1) given start and length (in bytes), using numpy.memmap for efficient access.
    Returns: list of rows (as lists of strings).
    """
    # Use memmap to read the section as bytes
    mm = np.memmap(file1_path, dtype=np.uint8, mode="r", offset=start, shape=(length,))
    section_bytes = mm.tobytes()
    # Use csv.reader on the section
    lines = section_bytes.decode("ascii").splitlines()
    reader = csv.reader(lines)
    rows = list(reader)
    return rows
