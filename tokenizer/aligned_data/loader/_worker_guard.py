"""Hard guard against opening a ``BinarySession`` inside a PyTorch
DataLoader worker process.

Single concern: detect ``torch.utils.data.get_worker_info() is not
None`` and raise a clear migration message. The session opens
per-binary file handles + memmaps; carrying those across a fork
boundary is the actual risk the egress-copy lifetime contract does
not protect against.

Torch is an optional dependency here -- the loader is usable without
torch installed, in which case the guard is a no-op (the check is
gated on the import succeeding).
"""

from __future__ import annotations


def assert_main_process() -> None:
    """Raise if invoked inside a PyTorch DataLoader worker process.

    Memmap-backed reads are fast enough that ``num_workers=0`` is the
    expected mode. The guard prevents accidental misuse rather than
    silently surviving with subtle fd / mmap issues across the fork.
    No-op when torch is not installed.
    """
    try:
        from torch.utils.data import get_worker_info
    except ImportError:
        return
    info = get_worker_info()
    if info is None:
        return
    raise RuntimeError(
        "BinarySession must run in the main process; PyTorch DataLoader "
        f"worker context detected (worker id={info.id} of {info.num_workers}). "
        "Use num_workers=0 or precompute batches in the main process."
    )
