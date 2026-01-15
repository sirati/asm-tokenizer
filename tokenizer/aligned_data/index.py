import numpy as np
import os


def load_index_memmap(index_path):
    """
    Load the index file (file 3) as a numpy memmap.
    Each entry: 4 bytes start, 3 bytes length, 1 byte avg_len (8 bytes per entry).
    Returns: memmap of shape (N, 8) as uint8.
    """
    filesize = os.path.getsize(index_path)
    assert filesize % 8 == 0, f"Index file size {filesize} is not a multiple of 8."
    n_entries = filesize // 8
    return np.memmap(index_path, dtype=np.uint8, mode="r", shape=(n_entries, 8))


def extract_avg_lengths(index_memmap):
    """
    Given the memmap of the index file, extract the average length column as uint8.
    """
    return index_memmap[:, 7]


def create_length_lookup_map(avg_lengths):
    """
    Create a lookup array of size 256, where each entry contains the starting index for that average length,
    or the same as the previous if not present. (Assumes avg_lengths is sorted.)
    Uses numpy.unique for efficiency.
    Returns: lookup (np.ndarray, shape=(256,), dtype=int)
    """
    unique_lengths, indices = np.unique(avg_lengths, return_index=True)
    lookup = np.zeros(256, dtype=int)
    last_idx = 0
    u_ptr = 0
    for length in range(256):
        if u_ptr < len(unique_lengths) and unique_lengths[u_ptr] == length:
            last_idx = indices[u_ptr]
            u_ptr += 1
        lookup[length] = last_idx
    return lookup


def select_random_function_by_length(index_memmap, lookup, target_length, rng=None):
    """
    Given a target average length, use the lookup array to randomly select an entry from the correct range.
    Returns: index of the selected function in the index file.
    """
    if rng is None:
        rng = np.random.default_rng()
    start = lookup[target_length]
    n = len(index_memmap)
    end = lookup[target_length + 1] if target_length + 1 < len(lookup) else n
    if end == start:
        raise ValueError(f"No function with average length {target_length} found.")
    return rng.integers(start, end)


def read_index_entry(index_memmap, idx):
    """
    Given the memmap and an index, return (start, length, avg_len) for the function.
    """
    entry = index_memmap[idx]
    start = int.from_bytes(entry[0:4], "little")
    length = int.from_bytes(entry[4:7], "little")
    avg_len = entry[7]
    return start, length, avg_len
