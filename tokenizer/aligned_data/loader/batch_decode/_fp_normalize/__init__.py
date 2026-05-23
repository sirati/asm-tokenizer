"""Vectorized per-TokenType FP normalization to the ``f96`` sidecar shape.

Public entry: :func:`normalize_per_token_type` -- dispatches per
:class:`TokenType` to the right per-type kernel, gathering payload bytes
from a shared ``inline_bytes`` buffer via the caller-supplied ``idx_2d``
indexers.

Package layout (one concern per module):

* :mod:`._primitives` -- shared vectorized bit ops (leading-1 position via
  binary-search ladder; vectorized :func:`pack_sign_exp`; vectorized
  :func:`_emit_chunk`; vectorized :func:`_encode_infnan`).
* :mod:`._ieee_narrow` -- F16 / BF16 / F32 / F64 (IEEE-754 narrow widths
  whose effective mantissa fits in u64).
* :mod:`._f80` -- x87 extended-precision (10-byte payload; explicit leading
  bit + 15-bit exponent).
* :mod:`._f128` -- IEEE-754 binary128 (16-byte payload; fixed 2-chunks-per-
  finite-source layout per plan; 1-chunk NaN/Inf).
* :mod:`._vc2` -- VC2 (``valued_const_v2``) integer multi-chunk encoder
  (8-byte payload per chunk; per-source sign expanded via
  ``chunk_exponent_sidecar``).
* :mod:`._dispatch` -- :func:`normalize_per_token_type` -- dispatch +
  byte-gathering.

This is the **vectorized rewrite** of the per-source Python-loop encoders in
:mod:`tokenizer.aligned_data.loader.decoded.custom_float`
(``from_float16`` / ``from_bfloat16`` / ``from_float32`` / ``from_float64`` /
``from_float80`` / ``from_float128`` / ``from_int``). The oracle remains the
source of truth for per-chunk byte equivalence; per-chunk values match the
oracle exactly. The ONLY documented divergence is the F128 chunk count:
the batch path always emits 2 chunks per finite F128 source (1 per NaN/Inf),
whereas the oracle's ``_split_to_chunks`` short-circuits to 1 chunk when the
effective mantissa fits in u64 (i.e. F128 +/-0 and denormals with
bit_length <= 64). The per-chunk normalization formula is identical.
"""

from ._dispatch import normalize_per_token_type

__all__ = ["normalize_per_token_type"]
