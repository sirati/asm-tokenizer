"""Hard guard against opening ``BinarySession`` inside a PyTorch
DataLoader worker process.

The guard fires at ``__enter__`` (the moment per-binary file handles
and memmaps would otherwise be acquired in the child process); it is
a no-op when torch is not installed AND outside of a DataLoader
worker. Tests cover all three paths.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from tokenizer.aligned_data.loader._worker_guard import assert_main_process


def test_assert_main_process_passes_in_main_process() -> None:
    """Outside any DataLoader worker, the guard is a no-op."""
    # Should return None without raising. If torch is installed,
    # get_worker_info() returns None in the main process; if not, the
    # import inside the guard fails and the guard short-circuits.
    assert assert_main_process() is None


def test_assert_main_process_no_op_when_torch_absent(monkeypatch) -> None:
    """No torch installed -> the guard returns silently."""
    # Hide torch.utils.data even if torch is installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.utils", None)
    monkeypatch.setitem(sys.modules, "torch.utils.data", None)
    assert assert_main_process() is None


def test_assert_main_process_raises_in_worker(monkeypatch) -> None:
    """Fake a DataLoader worker context and confirm the guard raises."""
    fake_module = SimpleNamespace(
        get_worker_info=lambda: SimpleNamespace(id=2, num_workers=4),
    )
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_module)
    with pytest.raises(RuntimeError, match="num_workers=0"):
        assert_main_process()
