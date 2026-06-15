"""Default per-leaf move + default leaf predicate (torch opt-in).

Single concern: the DEFAULT policy for "move one tensor leaf to the
device" and "is this value a movable leaf". Both are injectable on the
prefetcher; these are the torch-backed defaults. The torch import is
SOFT so the package stays importable (and the orchestration testable)
without torch -- these defaults are simply only *called* on the GPU path.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised only where torch absent
    torch = None  # type: ignore[assignment]


def default_to_device(t: "torch.Tensor", device: "torch.device") -> "torch.Tensor":
    """Pin (if a non-pinned CPU tensor) then async H2D onto ``device``.

    Pinned host memory is the precondition for a truly overlapping
    ``non_blocking`` copy; an already-pinned tensor (the consumer may pin
    inside ``produce``) is uploaded directly. Non-CUDA targets fall back
    to a plain blocking ``.to`` -- correct there, just without overlap.
    """
    if device.type != "cuda":
        return t.to(device)
    if t.device.type == "cpu" and not t.is_pinned():
        t = t.pin_memory()
    return t.to(device, non_blocking=True)


def pin_host(t: "torch.Tensor") -> "torch.Tensor":
    """Option-C move: pin a non-pinned CPU tensor (no H2D); else pass through."""
    if t.device.type == "cpu" and not t.is_pinned():
        return t.pin_memory()
    return t


def default_leaf_pred(obj: Any) -> bool:
    """Default movable-leaf predicate: a torch tensor (when torch present).

    Torch-less callers MUST inject their own ``is_leaf``; without torch
    there is no tensor type, so nothing is a leaf and every batch passes
    through structurally unchanged.
    """
    return torch is not None and torch.is_tensor(obj)
