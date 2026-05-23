"""Stage 3 stub — bulk u8 buffer + 2D indexers + vectorized FP normalization
to the f96 sidecar shape.

Operates over ``Stage2Batch`` and produces ``Stage3Batch`` (D9). The
batch-shared arrays (``inline_bytes``, per-TokenType ``(significand, sign_exp)``,
``identities_flat_caller_local``) live at level 1; per-level-4 slices into
them are recorded on each ``Stage3CallTarget``.

Iteration order: sections → variants → call_targets in DFS encounter order,
the same linearization stage 1 used.

1. Compute ``inline_bytes`` total size: ``1 + sum_over_level4(surviving.inline_byte_count)``.
   Leading slot 0 is the zero pad referenced by short-payload indexers.
2. Per level-4 call_target in linear order: extract surviving inline bytes via
   ALG-1 (narrowing u16→u8 assignment); write into ``inline_bytes`` at the
   call_target's allocated offset. Record ``Stage3CallTarget.inline_byte_slice``.
3. Build ``identity_idx_2d`` per ALG-5 (skipping prepend slots — written by
   stage 4 in ALG-9). View-cast to ``identities_flat_caller_local: u16[N]``.
   Per-level-4 ``identity_slice`` includes the leading prepend slot.
4. Build per-TokenType number ``idx_2d`` arrays per ALG-7 + ALG-8:
     - VC2 + F128: K rows per source (K from stage 2's chunk-count
       predictions; VC2 byte layout per ALG-8 with optional MSB short chunk).
     - F16 / BF16 / F32 / F64: 1 row per source.
     - F80: 1 row per source (10 bytes; 5 big-endian u16 limbs).
5. Vectorized per-TokenType FP normalization to ``(u64 significand, u32
   sign_exponent)`` per ALG-7. Bit-field extraction → renormalize the mantissa
   so the leading 1 sits at bit 63 ("without integer bit") → pack sign +
   biased exponent via the existing ``pack_sign_exp`` convention. Denormal +
   NaN/Inf branches handled per-TokenType. Stage 3 produces per-TokenType
   ``(significand: u64, sign_exp: u32)`` array pairs.
6. Multi-chunk per-source bookkeeping (VC2 exponent-base = chunk_index × 64;
   F128 finite emits 2 chunks, NaN/Inf emits 1 via ``_encode_infnan``).
7. Per-level-4 ``number_chunk_slices`` records each TokenType slice into the
   level-1 ``numbers_per_TokenType[T]`` arrays.

Body is intentionally a ``NotImplementedError`` — Phase 3 subagents fill this
in. See ``batch_decode_plan.md`` ``Stage 3`` + ALG-1 + ALG-5 + ALG-7 + ALG-8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Stage2Batch, Stage3Batch


def build_bulk_bytes(
    stage2: "Stage2Batch",
) -> "Stage3Batch":
    """Stage 3: inline-byte concat via narrowing assignment (ALG-1) + per-
    TokenType ``idx_2d`` construction (ALG-5 + ALG-7 + ALG-8) + vectorized FP
    normalization to the f96 sidecar shape.

    Produces ``Stage3Batch`` with ``inline_bytes``,
    ``identities_flat_caller_local`` (pre-remap), and per-TokenType
    ``(significand, sign_exponent)`` array pairs populated.
    """

    raise NotImplementedError(
        "Phase 3 — see batch_decode_plan.md '## Stages — algorithm sketch' "
        "Stage 3 + ALG-1 + ALG-5 + ALG-7 + ALG-8."
    )
