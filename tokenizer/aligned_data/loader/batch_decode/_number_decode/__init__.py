"""Stage 3c -- per-:class:`TokenType` number ``idx_2d`` construction.

See :mod:`._entry` for the full ALG-7 + ALG-8 byte-layout docstring.

Module layout (one concern per file):

* :mod:`._band_constants` -- vocab anchors + canonical NUMBER-block
  TokenType ordering + per-type row widths.
* :mod:`._emit_fixed_fp` -- per-source row emission for F16 / BF16 /
  F32 / F64 / F80.
* :mod:`._emit_f128` -- per-source row emission for FLOAT128 (1- and
  2-chunk variants + the mid-cut LSB-only case).
* :mod:`._emit_vc2` -- per-source row emission for VALUED_CONST_V2
  (ALG-8 multi-chunk packing).
* :mod:`._per_call_target` -- per-call-target expanded-stream walk +
  per-source dispatch.
* :mod:`._entry` -- :func:`build_number_idx_2d` orchestrator over the
  4-level batch hierarchy.
"""

from ._band_constants import _NUMBER_BLOCK_TOKEN_TYPES
from ._entry import build_number_idx_2d


__all__ = ["build_number_idx_2d", "_NUMBER_BLOCK_TOKEN_TYPES"]
