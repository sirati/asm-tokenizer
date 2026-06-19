"""Stage 3c -- per-:class:`TokenType` number ``idx_2d`` construction.

See :mod:`._entry` for the full ALG-7 + ALG-8 byte-layout docstring.

Module layout (one concern per file):

* :mod:`._band_constants` -- vocab anchors + canonical NUMBER-block
  TokenType ordering + per-type row widths.
* :mod:`._flat_segments` -- GIL-bound front matter: walk the shared
  Step-1 call_target columns once + concatenate the flat per-segment
  NUMBER-band context the GIL-released emission kernel consumes.
* :mod:`._entry` -- :func:`build_number_idx_2d` orchestrator: build the
  flat segments, run ``dedup_hashmap.build_number_idx_2d_kernel`` (the
  GIL-released carrier-recovery + ALG-2/7/8 row emission state machine),
  reconstruct per-call_target slices from the per-carrier ROW counts.

The carrier identification (segmented expanded->raw recovery + byte
offsets) and the per-:class:`TokenType` row emission (ALG-7 fixed-width,
ALG-2 F128 1/2-chunk, ALG-8 VC2 multi-chunk packing) live in the Rust
kernel ``dedup_hashmap.build_number_idx_2d_kernel`` (see
``dedup_hashmap/src/number_idx_2d.rs``).
"""

from ._band_constants import _NUMBER_BLOCK_TOKEN_TYPES
from ._entry import build_number_idx_2d


__all__ = ["build_number_idx_2d", "_NUMBER_BLOCK_TOKEN_TYPES"]
