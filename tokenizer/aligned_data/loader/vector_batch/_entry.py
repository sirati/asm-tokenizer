"""Vectorized batch dataloader -- the entry orchestrator (plan C3).

Single concern: thread the vectorized path end to end -- sample the same
``(section, variant)`` rows ``batch_decode`` would, run the body-free
geometry prepass (plan C1), run the fused scatter (plan C2), and return a
typed result whose ``tokens`` + ``batch_idx_to_section_variant`` mirror
``batch_decode``'s. Backfill is a flag, DEFAULT OFF -- the optional
geometry-augmentation hook (TD) is left as a clean seam before the
scatter; this module never implements backfill.

Sampling parity (byte-identity contract): the SAME sampler
``batch_decode`` uses is reused verbatim -- :func:`...batch_decode.
_resolve_pointers.resolve_section_pointers` (RNG-sampled native variant
indices per section) + :func:`...batch_decode._section_walk._batch_idx.
compute_batch_idx_mapping` (the ALG-10 ``batch_idx`` layout). Driven from
the same ``rng`` the two paths draw IDENTICAL samples, so the geometry +
scatter assemble byte-identically against ``batch_decode`` with backfill
off (the entry harness proves this).

The geometry handles (columnar catalog + RLG3 geometry + ``_variants.bin``
+ ``_data.bin``) are opened body-free via the SAME readers the index
build uses (:func:`...sorted_index._prepass.read_section_variant_info`,
:class:`...realized_lengths.RealizedGeometryReader`) -- no bespoke BIN
parse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._resolve_pointers import (
    resolve_section_pointers,
)
from tokenizer.aligned_data.loader.batch_decode._batch_layout import (
    compute_batch_idx_mapping,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.realized_lengths import (
    RealizedGeometryReader,
)
from tokenizer.aligned_data.sorted_index._prepass import (
    read_section_variant_info,
)

from ._geometry import compute_batch_geometry
from ._scatter import build_dense_sidecars, scatter_batch_tokens
from .session_handles import VectorBatchHandles


__all__ = ["VectorBatchResult", "vector_batch_tokens"]


#: The sentinel a PAD_NULL padding row carries in
#: ``batch_idx_to_section_variant`` (matches ``Stage1Batch``).
_PADDING_SENTINEL = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class VectorBatchResult:
    """The vectorized path's full result (backfill OFF).

    Mirrors the ``batch_decode`` ``BatchDecodeResult`` fields the
    byte-identity harness compares: the ``u16[B, L]`` token tensor, the
    ``u32[B, 2]`` ``(section_idx, variant_idx)`` mapping (padding rows
    hold the ``(UINT32_MAX, UINT32_MAX)`` sentinel), AND the DENSE
    sidecars -- the post-remap caller-local->counter identity array (+
    offsets), the ``(significand, sign_exp)`` numeric arrays (+ offsets),
    and the optional per-Category FID sidecars. Every array is
    byte-identical to ``batch_decode`` with backfill off.
    """

    tokens: np.ndarray
    batch_idx_to_section_variant: np.ndarray
    identities: np.ndarray
    identity_row_offsets: np.ndarray
    numbers_significant: np.ndarray
    numbers_sign_exponent: np.ndarray
    number_row_offsets: np.ndarray
    fid_sidecar: Optional[np.ndarray]
    fid_row_offsets: Optional[np.ndarray]
    fid_per_category_counts: Optional[np.ndarray]


def vector_batch_tokens(
    session,
    section_pointers: List[SectionPointerSpec],
    *,
    handles: VectorBatchHandles,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    rng: Optional[np.random.Generator] = None,
    augment_geometry=None,
    include_fid_sidecar: bool = False,
) -> VectorBatchResult:
    """Sample -> geometry prepass -> fused scatter -> token + dense sidecars.

    Parameters
    ----------
    session:
        The :class:`BinarySession` the sampler resolves pointers +
        variants through (the SAME object ``batch_decode`` samples on).
    section_pointers:
        The MATCHED-arm section pointers to batch (one ``(arm, idx)``
        per section). Only ``SectionKind.MATCHED`` is supported by the
        geometry path; an unmatched pointer raises.
    handles:
        The opened body-free + body geometry handles
        (:class:`.session_handles.VectorBatchHandles`): the columnar
        catalog + ``section_offsets``, the RLG3 geometry reader, and the
        ``_variants.bin`` / ``_data.bin`` uint8 views.
    num_variants_per_section / context_len / max_depth / variant_padding /
    rng:
        The same knobs ``batch_decode`` takes; ``rng`` defaults to a
        fresh non-reproducible generator (pass an explicit one for
        deterministic / equivalence-tested sampling).
    augment_geometry:
        OPTIONAL backfill seam (default ``None`` = backfill OFF). When
        provided it is a callable ``BatchGeometry -> BatchGeometry``
        applied AFTER the prepass and BEFORE the scatter (TD builds the
        backfill transform separately). This module never implements
        backfill; it only leaves the hook.
    include_fid_sidecar:
        When True, the dense pass also produces the per-Category FID
        sidecars (``fid_sidecar`` / ``fid_row_offsets`` /
        ``fid_per_category_counts``), matching ``batch_decode``'s
        same-named flag. Default ``False`` (those fields are ``None``).

    Returns
    -------
    VectorBatchResult
        The token tensor + the ``batch_idx_to_section_variant`` mapping
        + the dense identity / numeric sidecars (+ optional FID sidecars).
    """
    if rng is None:
        rng = np.random.default_rng()
    if section_pointers and any(
        sp.arm is not SectionKind.MATCHED for sp in section_pointers
    ):
        raise NotImplementedError(
            "vector_batch_tokens supports MATCHED-arm pointers only; the "
            "RLG3 geometry + columnar catalog are matched-arm scoped"
        )

    # --- sample EXACTLY as batch_decode does (shared sampler + rng) ------
    resolved = resolve_section_pointers(
        session,
        section_pointers,
        num_variants_per_section=num_variants_per_section,
        rng=rng,
    )
    batch_idx_to_section_variant, batch_size = compute_batch_idx_mapping(
        resolved,
        num_variants_per_section=num_variants_per_section,
        variant_padding=variant_padding,
        rng=rng,
    )

    # --- per-batch-row (catalog section idx, NATIVE variant idx) ---------
    root_sections, root_variants = _rows_to_catalog_nodes(
        batch_idx_to_section_variant, resolved
    )

    geometry = compute_batch_geometry(
        cols=handles.cols,
        section_offsets=handles.section_offsets,
        geometry=handles.geometry,
        variants_u8=handles.variants_u8,
        root_sections=root_sections,
        root_sampled_variants=root_variants,
        seq_len=context_len,
        max_depth=max_depth,
    )
    if augment_geometry is not None:
        geometry = augment_geometry(geometry)

    scattered = scatter_batch_tokens(
        geometry,
        cols=handles.cols,
        data_u8=handles.data_u8,
        variants_u8=handles.variants_u8,
    )

    # Re-expand the per-included-row tensor onto the full batch (padding
    # rows stay all-zero). The geometry runs only over the non-padding
    # rows; ``batch_idx_to_section_variant`` carries the padding layout.
    tokens = _expand_to_batch(
        scattered.tokens,
        batch_idx_to_section_variant,
        batch_size=batch_size,
        context_len=context_len,
    )

    # Dense identity + numeric sidecars from the SAME expanded bodies the
    # token scatter produced (no _data.bin re-read / re-expand). Already
    # placed onto the full batch (padding rows zero-length).
    dense = build_dense_sidecars(
        geometry,
        scattered.expanded,
        cols=handles.cols,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        batch_size=batch_size,
        include_fid_sidecar=include_fid_sidecar,
    )
    return VectorBatchResult(
        tokens=tokens,
        batch_idx_to_section_variant=batch_idx_to_section_variant,
        identities=dense.identities,
        identity_row_offsets=dense.identity_row_offsets,
        numbers_significant=dense.numbers_significant,
        numbers_sign_exponent=dense.numbers_sign_exponent,
        number_row_offsets=dense.number_row_offsets,
        fid_sidecar=dense.fid_sidecar,
        fid_row_offsets=dense.fid_row_offsets,
        fid_per_category_counts=dense.fid_per_category_counts,
    )


def _rows_to_catalog_nodes(batch_idx_to_section_variant, resolved):
    """Per NON-padding batch row, ``(catalog_section_idx, native_variant)``.

    ``batch_idx_to_section_variant`` column 0 is the position in
    ``resolved``; column 1 is the SLOT into that section's
    ``sampled_variant_indices`` (post-sampling, NOT the native variant).
    The prepass needs the catalog section index (MATCHED ``idx``) + the
    NATIVE variant index, so we map both through ``resolved``.
    """
    mapping = np.asarray(batch_idx_to_section_variant, dtype=np.int64)
    is_padding = mapping[:, 0] == int(_PADDING_SENTINEL)
    real = mapping[~is_padding]
    sec_out = np.empty(real.shape[0], dtype=np.int64)
    var_out = np.empty(real.shape[0], dtype=np.int64)
    for i, (resolved_pos, slot) in enumerate(real.tolist()):
        rs = resolved[resolved_pos]
        sec_out[i] = int(rs.idx)
        var_out[i] = int(rs.sampled_variant_indices[slot])
    return sec_out, var_out


def _expand_to_batch(
    row_tokens, batch_idx_to_section_variant, *, batch_size, context_len
):
    """Place the per-non-padding-row token tensor into the full batch.

    The geometry + scatter run over the non-padding rows in batch order;
    this scatters them back to their ``batch_idx`` positions, leaving
    padding rows (sentinel) at the all-zero null-content default. The
    non-padding rows keep their original batch order, so the i-th
    geometry row maps to the i-th non-padding batch row.
    """
    mapping = np.asarray(batch_idx_to_section_variant, dtype=np.int64)
    tokens = np.zeros((batch_size, context_len), dtype=np.uint16)
    if batch_size == 0 or context_len == 0:
        return tokens
    is_real = mapping[:, 0] != int(_PADDING_SENTINEL)
    real_rows = np.nonzero(is_real)[0]
    if real_rows.size != row_tokens.shape[0]:
        raise AssertionError(
            f"geometry produced {row_tokens.shape[0]} rows but the batch "
            f"mapping has {real_rows.size} non-padding rows"
        )
    tokens[real_rows] = row_tokens
    return tokens
