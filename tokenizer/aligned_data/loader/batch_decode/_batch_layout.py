"""Stage 1 sub-concern: compute the ``batch_idx -> (section_idx, variant_idx)``
layout per :class:`VariantPadding` policy (plan ALG-10).

This module owns ONE concern: turning a list of variant-sampled sections plus a
padding policy into the canonical ``u32[batch_size, 2]`` mapping that the four
stages all iterate through. Computed ONCE at the top of stage 1, AFTER per-
section variant sampling, BEFORE loading. Pure function -- no I/O, no session
access; the caller hands us already-sampled variant index lists per section.

Per ALG-10's policy table (verbatim from ``batch_decode_plan.md``):

* ``PAD_NULL``: ``batch_size = num_sections * num_variants_per_section``;
  ``mapping[s * nv + v] = (s, v)`` when section ``s`` has at least ``v + 1``
  sampled variants; ``(UINT32_MAX, UINT32_MAX)`` otherwise.
* ``RESAMPLE_WITHIN_SECTION``: same layout shape as ``PAD_NULL``; missing slots
  filled by random sampling with replacement from this section's available
  variant slots.
* ``RAGGED``: ``batch_size = total_real_variants``; dense mapping over the
  sampled variants only -- no padding rows.
* ``REDISTRIBUTE``: same shape as ``PAD_NULL``; sections with MORE sampled
  variants than ``num_variants_per_section`` donate extras to sections with
  fewer, so the final mapping is dense (no ``UINT32_MAX`` rows) but the
  ``section_id`` column is non-uniform across the linear layout.

The mapping's ``variant_idx`` column is the SLOT index into the section's
``sampled_variant_indices`` list (i.e. the index in
:class:`Stage1Section.variants`), NOT the original variant index inside the
section's underlying variants list. Callers recover the original variant idx
via ``sampled_variant_indices[section_idx][slot_v]``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ._resolve_pointers import ResolvedSection
from ._types import VariantPadding


# ---------------------------------------------------------------------------
# Public sentinel constant
# ---------------------------------------------------------------------------


UINT32_MAX: np.uint32 = np.uint32(0xFFFFFFFF)
"""Sentinel marker for padding rows in ``batch_idx_to_section_variant``.

Per plan ALG-10: padding rows hold ``(UINT32_MAX, UINT32_MAX)``. Only the
:attr:`VariantPadding.PAD_NULL` policy produces padding rows; the other three
policies yield a dense mapping.
"""


__all__ = [
    "UINT32_MAX",
    "compute_batch_idx_mapping",
]


# ---------------------------------------------------------------------------
# Per-policy layout builders
# ---------------------------------------------------------------------------


def _layout_pad_null(
    real_counts: np.ndarray,
    *,
    num_sections: int,
    num_variants_per_section: int,
) -> Tuple[np.ndarray, int]:
    """Linear ``s * nv + v`` layout; missing slots get ``(UINT32_MAX, UINT32_MAX)``."""

    batch_size = num_sections * num_variants_per_section
    mapping = np.full((batch_size, 2), UINT32_MAX, dtype=np.uint32)

    # Vectorize over sections: for each section s, fill the leading real_counts[s]
    # slots with (s, 0..real_counts[s]-1).
    section_ids = np.repeat(
        np.arange(num_sections, dtype=np.uint32), num_variants_per_section
    )
    slot_ids = np.tile(
        np.arange(num_variants_per_section, dtype=np.uint32), num_sections
    )
    # A row is "real" iff slot_v < real_counts[section_id].
    real_count_per_row = real_counts[section_ids]
    is_real = slot_ids < real_count_per_row
    mapping[is_real, 0] = section_ids[is_real]
    mapping[is_real, 1] = slot_ids[is_real]
    return mapping, batch_size


def _layout_resample(
    real_counts: np.ndarray,
    *,
    num_sections: int,
    num_variants_per_section: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """Linear ``s * nv + v`` layout; missing slots oversampled (with
    replacement) from this section's available slot indices.

    A section with zero sampled variants has no source pool to resample from;
    such slots fall back to the ``UINT32_MAX`` sentinel (same as ``PAD_NULL``).
    Realistically this should not occur when ``RESAMPLE_WITHIN_SECTION`` is
    used -- the caller would not request resampling for empty sections -- but
    the fallback keeps the function total.
    """

    batch_size = num_sections * num_variants_per_section
    mapping = np.full((batch_size, 2), UINT32_MAX, dtype=np.uint32)

    for s in range(num_sections):
        real = int(real_counts[s])
        row_base = s * num_variants_per_section
        # Fill the leading `real` slots with their direct (s, slot_v) entries.
        for v in range(min(real, num_variants_per_section)):
            mapping[row_base + v, 0] = np.uint32(s)
            mapping[row_base + v, 1] = np.uint32(v)
        # Resample remaining slots from this section's available slot indices.
        deficit = num_variants_per_section - real
        if deficit <= 0 or real <= 0:
            # No deficit, or no source pool -- leave any remaining slots as
            # UINT32_MAX (the latter case mirrors PAD_NULL for empty sections).
            continue
        resampled_slots = rng.integers(
            low=0, high=real, size=deficit, dtype=np.int64
        )
        for k, slot_v in enumerate(resampled_slots):
            mapping[row_base + real + k, 0] = np.uint32(s)
            mapping[row_base + real + k, 1] = np.uint32(slot_v)
    return mapping, batch_size


def _layout_ragged(
    real_counts: np.ndarray,
    *,
    num_sections: int,
) -> Tuple[np.ndarray, int]:
    """Dense mapping: one row per actually-sampled variant, no padding rows."""

    batch_size = int(real_counts.sum())
    mapping = np.empty((batch_size, 2), dtype=np.uint32)
    write_pos = 0
    for s in range(num_sections):
        real = int(real_counts[s])
        if real <= 0:
            continue
        mapping[write_pos : write_pos + real, 0] = np.uint32(s)
        mapping[write_pos : write_pos + real, 1] = np.arange(real, dtype=np.uint32)
        write_pos += real
    assert write_pos == batch_size
    return mapping, batch_size


def _layout_redistribute(
    real_counts: np.ndarray,
    *,
    num_sections: int,
    num_variants_per_section: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """Linear shape ``batch_size = num_sections * num_variants_per_section``;
    sections with MORE sampled variants than ``nv`` donate their excess slots
    to sections with FEWER. Result is dense (no ``UINT32_MAX`` rows) provided
    ``donor_pool >= deficit_total``.

    Donation algorithm:

    1. For each section, the leading ``min(real, nv)`` of its slots take the
       section's own slot indices ``0 .. min(real, nv) - 1``.
    2. Donor pool = all ``(donor_s, slot_v)`` where ``slot_v >= nv`` (i.e. the
       section had more sampled variants than the per-section budget).
    3. RNG-permute the donor pool, then assign donors in linear-row order to
       deficit slots (rows whose ``slot_v >= real_counts[s]``).

    When the donor pool is insufficient to cover the deficit (e.g. all
    sections are short), the remaining deficit slots stay
    ``(UINT32_MAX, UINT32_MAX)`` -- callers must ensure their sampling produced
    enough donors when they pick this policy.
    """

    batch_size = num_sections * num_variants_per_section
    mapping = np.full((batch_size, 2), UINT32_MAX, dtype=np.uint32)

    # Fill the "own" leading slots per section.
    section_ids = np.repeat(
        np.arange(num_sections, dtype=np.uint32), num_variants_per_section
    )
    slot_ids = np.tile(
        np.arange(num_variants_per_section, dtype=np.uint32), num_sections
    )
    real_count_per_row = real_counts[section_ids]
    is_own_real = slot_ids < real_count_per_row
    mapping[is_own_real, 0] = section_ids[is_own_real]
    mapping[is_own_real, 1] = slot_ids[is_own_real]

    # Build the donor pool: every (s, slot_v) with slot_v >= nv (i.e. the
    # section's sampled list extended past the per-section budget).
    donor_pairs = []
    for s in range(num_sections):
        real = int(real_counts[s])
        if real > num_variants_per_section:
            for slot_v in range(num_variants_per_section, real):
                donor_pairs.append((s, slot_v))
    if donor_pairs:
        donor_arr = np.array(donor_pairs, dtype=np.uint32)
        perm = rng.permutation(donor_arr.shape[0])
        donor_arr = donor_arr[perm]
    else:
        donor_arr = np.empty((0, 2), dtype=np.uint32)

    # Assign donors to deficit rows in linear-row order.
    deficit_rows = np.flatnonzero(~is_own_real)
    n_assign = min(deficit_rows.shape[0], donor_arr.shape[0])
    mapping[deficit_rows[:n_assign]] = donor_arr[:n_assign]
    return mapping, batch_size


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def compute_batch_idx_mapping(
    resolved_sections: List[ResolvedSection],
    *,
    num_variants_per_section: int,
    variant_padding: VariantPadding,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    """Build ``batch_idx_to_section_variant`` per :class:`VariantPadding`
    policy (plan ALG-10).

    Parameters
    ----------
    resolved_sections:
        One :class:`ResolvedSection` per section pointer in the request, with
        ``sampled_variant_indices`` already populated by the variant-sampling
        step (sibling 1a / ``_resolve_pointers``).
    num_variants_per_section:
        Per-section variant budget. The four policies interpret this
        differently; see :class:`VariantPadding` + this module's docstring.
    variant_padding:
        Policy enum controlling the layout shape + the missing-slot semantics.
    rng:
        Required by :attr:`VariantPadding.RESAMPLE_WITHIN_SECTION` and
        :attr:`VariantPadding.REDISTRIBUTE`. Passed unconditionally so the
        caller's seeding contract is uniform across policies.

    Returns
    -------
    (mapping, batch_size):
        ``mapping`` is ``np.uint32[batch_size, 2]`` with columns
        ``(section_idx, slot_v)`` where ``slot_v`` is the position inside the
        section's ``sampled_variant_indices`` list. Padding rows hold
        ``(UINT32_MAX, UINT32_MAX)`` -- only :attr:`VariantPadding.PAD_NULL`
        produces padding rows in normal operation; the other policies yield a
        dense mapping (modulo the empty-section / no-donor fallback noted on
        the per-policy helpers).
    """

    num_sections = len(resolved_sections)
    if num_variants_per_section < 0:
        raise ValueError(
            f"num_variants_per_section must be >= 0 (got {num_variants_per_section})"
        )

    if num_sections == 0:
        return np.empty((0, 2), dtype=np.uint32), 0

    real_counts = np.array(
        [len(s.sampled_variant_indices) for s in resolved_sections],
        dtype=np.int64,
    )

    if variant_padding is VariantPadding.PAD_NULL:
        return _layout_pad_null(
            real_counts,
            num_sections=num_sections,
            num_variants_per_section=num_variants_per_section,
        )
    if variant_padding is VariantPadding.RESAMPLE_WITHIN_SECTION:
        return _layout_resample(
            real_counts,
            num_sections=num_sections,
            num_variants_per_section=num_variants_per_section,
            rng=rng,
        )
    if variant_padding is VariantPadding.RAGGED:
        return _layout_ragged(
            real_counts,
            num_sections=num_sections,
        )
    if variant_padding is VariantPadding.REDISTRIBUTE:
        return _layout_redistribute(
            real_counts,
            num_sections=num_sections,
            num_variants_per_section=num_variants_per_section,
            rng=rng,
        )
    raise ValueError(f"unknown VariantPadding policy: {variant_padding!r}")
