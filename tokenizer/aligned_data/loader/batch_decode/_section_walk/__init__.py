"""Stage 1 outer wiring.

Public surface:

* :func:`walk_sections` -- compose Phase-1 modules into a
  :class:`Stage1Batch`. Two dispatch shapes (see the walker module's
  docstring): synchronous (``collector=None``) and pending
  (``collector`` provided + caller flushes / finalises).
* :class:`PendingStage1Batch` -- the pending shape that the
  ``collector``-provided path returns.
* :func:`finalise_pending_stage1` -- post-flush finaliser that turns a
  pending batch into a :class:`Stage1Batch`.
"""

from ._pending import PendingStage1Batch, finalise_pending_stage1
from ._walker import walk_sections


__all__ = [
    "PendingStage1Batch",
    "finalise_pending_stage1",
    "walk_sections",
]
