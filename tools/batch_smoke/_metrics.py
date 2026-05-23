"""Per-session :func:`batch_decode` invocation + shape capture.

Single concern: given an open :class:`BinarySession` + the binary's
matched-arm count, run :func:`batch_decode` once and snapshot the
:class:`BatchDecodeResult` shape into a flat JSON-serialisable dict.
The driver layer (``tools.run_batch_smoke``) owns argument parsing,
session lifecycle, and JSON writing.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from tokenizer.aligned_data.loader.batch_decode import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession


def matched_section_pointers(
    matched_count: int, max_functions: int
) -> List[SectionPointerSpec]:
    """Build the first ``min(matched_count, max_functions)`` matched
    pointers in deterministic ``idx`` order.

    Pure data shaping -- no sampling here; the variant axis sampling
    lives inside :func:`batch_decode`.
    """
    take = min(matched_count, max_functions)
    return [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=i)
        for i in range(take)
    ]


def _result_metrics(
    result: BatchDecodeResult, *, section_count: int, wall_seconds: float
) -> Dict[str, Any]:
    """Snapshot the public :class:`BatchDecodeResult` shape into a flat
    JSON-serialisable dict.
    """
    tokens_shape = list(result.tokens.shape)
    return {
        "batch_size": int(tokens_shape[0]),
        "tokens_shape": tokens_shape,
        "total_identity_chunks": int(result.identities.shape[0]),
        "total_number_chunks": int(result.numbers_significant.shape[0]),
        "section_count": int(section_count),
        "wall_seconds": round(float(wall_seconds), 6),
    }


def _empty_block(context_len: int) -> Dict[str, Any]:
    """Empty-batch metric block for binaries with zero matched
    sections. Keeps the JSON schema stable across such binaries.
    """
    return {
        "batch_size": 0,
        "tokens_shape": [0, context_len],
        "total_identity_chunks": 0,
        "total_number_chunks": 0,
        "section_count": 0,
        "wall_seconds": 0.0,
    }


def collect_session_metrics(
    session: BinarySession,
    matched_count: int,
    *,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: VariantPadding,
    max_functions_per_binary: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Run :func:`batch_decode` on the open session and return its
    metrics dict.

    Caller owns the session lifecycle. ``matched_count`` is read off
    the binary's matched arm; passing it in rather than re-deriving
    here keeps the function single-concern (it never reaches outside
    the session for index metadata).
    """
    section_pointers = matched_section_pointers(
        matched_count, max_functions_per_binary
    )
    if not section_pointers:
        return _empty_block(context_len)

    t0 = time.monotonic()
    result = batch_decode(
        session,
        section_pointers,
        num_variants_per_section=num_variants_per_section,
        context_len=context_len,
        max_depth=max_depth,
        variant_padding=variant_padding,
        rng=rng,
    )
    wall = time.monotonic() - t0
    return _result_metrics(
        result, section_count=len(section_pointers), wall_seconds=wall
    )
