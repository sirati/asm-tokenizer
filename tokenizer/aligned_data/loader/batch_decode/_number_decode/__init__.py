"""Stage 3c -- per-:class:`TokenType` number ``idx_2d`` construction.

See :mod:`._entry` for the full ALG-7 + ALG-8 byte-layout docstring.

Module layout (one concern per file):

* :mod:`._band_constants` -- vocab anchors + canonical NUMBER-block
  TokenType ordering + per-type row widths.
* :mod:`._batched_carriers` -- cross-call_target NUMBER-band carrier
  identification + location (segmented expanded->raw map + byte
  offsets); the flat carrier table the per-type emitters batch over.
* :mod:`._emit_fixed_fp` -- batched per-carrier row emission for F16 /
  BF16 / F32 / F64 / F80.
* :mod:`._emit_f128` -- batched per-carrier row emission for FLOAT128
  (1- and 2-chunk variants + the mid-cut LSB-only case).
* :mod:`._emit_vc2` -- batched per-carrier row emission for
  VALUED_CONST_V2 (ALG-8 multi-chunk packing).
* :mod:`._entry` -- :func:`build_number_idx_2d` orchestrator: build the
  carrier table, dispatch per-type emit, reconstruct per-call_target
  slices.
"""

from ._band_constants import _NUMBER_BLOCK_TOKEN_TYPES
from ._entry import build_number_idx_2d


__all__ = ["build_number_idx_2d", "_NUMBER_BLOCK_TOKEN_TYPES"]
