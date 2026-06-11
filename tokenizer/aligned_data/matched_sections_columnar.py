"""Columnar (vectorized) decode of ``sections.bin`` sections.

Single concern: given the ``sections.bin`` blob as a uint8 array and a
batch of section byte offsets, decode every section's header, jump
table, call_target table, variant headers, and per-call entries into
flat numpy arrays -- no per-section Python objects, no sequential walk.

The wire layout is owned by :mod:`.matched_sections_bin`
(:func:`~.matched_sections_bin.parse_section_bin` is the scalar source
of truth); this module mirrors it field-for-field and the equivalence
test (``tests/test_matched_sections_columnar.py``) pins the two
decoders together. Layout constants are imported, never restated.

Flattening convention: arrays are grouped per level with CSR-style
``*_offsets`` arrays (exclusive prefix sums) tying each level to its
parent, in on-disk order throughout:

* section level -- parallel to the input ``section_offsets``;
* call_target level -- section ``s`` owns rows
  ``ct_offsets[s] : ct_offsets[s + 1]``;
* variant level -- section ``s`` owns rows
  ``var_offsets[s] : var_offsets[s + 1]``;
* per-call-entry level -- flat variant ``v`` owns rows
  ``pce_offsets[v] : pce_offsets[v + 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matched_sections_bin import (
    CALL_TARGET_ENTRY_SIZE,
    PER_CALL_ENTRY_SIZE,
    SECTION_ALIGNMENT,
    SECTION_HEADER_SIZE,
    VARIANT_HEADER_SIZE,
)


__all__ = ["ColumnarSections", "parse_sections_columnar"]


# Flag-field bit layout mirror (kept in lockstep with
# ``matched_sections_bin._pack_flags`` via the equivalence test).
_FLAG_IS_MATCHED_BIT = 0
_FLAG_TYPE_SHIFT = 1
_FLAG_TYPE_MASK = 0b11


@dataclass(frozen=True)
class ColumnarSections:
    """Flat columnar view of a batch of parsed sections.

    See the module docstring for the CSR flattening convention. All
    integer columns are numpy arrays; ``*_offsets`` arrays have one
    more entry than their level has parents and start at 0.
    """

    # --- section level ---------------------------------------------------
    function_name_ptr: np.ndarray
    """``u32[n_sections]`` -- the section header's FID field."""

    n_call_targets: np.ndarray
    """``i64[n_sections]``."""

    n_variants: np.ndarray
    """``i64[n_sections]``."""

    # --- call_target level -------------------------------------------------
    ct_offsets: np.ndarray
    """``i64[n_sections + 1]`` CSR offsets into the ``ct_*`` columns."""

    ct_function_name_ptr: np.ndarray
    """``u32[total_cts]``."""

    ct_function_section_ptr: np.ndarray
    """``u32[total_cts]``."""

    ct_type: np.ndarray
    """``u8[total_cts]`` -- raw :class:`CallTargetType` values."""

    ct_is_matched: np.ndarray
    """``bool[total_cts]``."""

    # --- variant level -----------------------------------------------------
    var_offsets: np.ndarray
    """``i64[n_sections + 1]`` CSR offsets into the ``var_*`` columns."""

    var_ref_offset: np.ndarray
    """``u32[total_variants]`` -- the vkey."""

    var_data_offset_shifted: np.ndarray
    """``u32[total_variants]`` -- ``record_offset >> 4`` into _data.bin."""

    var_n_calls: np.ndarray
    """``i64[total_variants]``."""

    # --- per-call-entry level -----------------------------------------------
    pce_offsets: np.ndarray
    """``i64[total_variants + 1]`` CSR offsets into the ``pce_*`` columns."""

    pce_called_idx: np.ndarray
    """``u16[total_entries]`` -- index into the owning section's
    call_target table."""

    pce_section_variant_index: np.ndarray
    """``u16[total_entries]`` -- resolved callee variant index (or the
    ``MISSING_VARIANT_INDEX`` sentinel)."""


def _u16(b: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Gather little-endian u16 values at byte indices ``idx``."""
    return b[idx].astype(np.int64) | (b[idx + 1].astype(np.int64) << 8)


def _u32(b: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Gather little-endian u32 values at byte indices ``idx``."""
    return (
        b[idx].astype(np.int64)
        | (b[idx + 1].astype(np.int64) << 8)
        | (b[idx + 2].astype(np.int64) << 16)
        | (b[idx + 3].astype(np.int64) << 24)
    )


def _csr(counts: np.ndarray) -> np.ndarray:
    """Exclusive prefix-sum offsets (CSR) for ``counts``."""
    out = np.zeros(counts.size + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def _flat_member_addresses(
    base_per_parent: np.ndarray,
    counts: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-member byte addresses for fixed-stride tables.

    ``base_per_parent[p]`` is parent ``p``'s table start; parent ``p``
    owns ``counts[p]`` members of ``stride`` bytes each. Returns
    ``(parent_of_member, member_addresses)`` flattened in parent order.
    """
    offsets = _csr(counts)
    total = int(offsets[-1])
    parent = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    within = np.arange(total, dtype=np.int64) - offsets[parent]
    return parent, base_per_parent[parent] + stride * within


def parse_sections_columnar(
    blob_u8: np.ndarray,
    section_offsets: np.ndarray,
    section_lengths: np.ndarray | None = None,
) -> ColumnarSections:
    """Decode the sections at ``section_offsets`` into columnar arrays.

    Parameters
    ----------
    blob_u8:
        The full ``sections.bin`` as a 1-D uint8 array (prelude
        included; offsets are absolute file offsets, exactly as stored
        in ``_index.bin``).
    section_offsets:
        Integer array of section byte offsets (any integer dtype).
    section_lengths:
        Optional parallel byte lengths (from the matched locator).
        When given, every section's computed end (header + jump table +
        tables + alignment pad) is validated against it; a mismatch
        raises :class:`ValueError` naming the first offending section
        -- catching decoder/layout drift loudly instead of returning
        garbage columns.

    Returns
    -------
    ColumnarSections
        Flat columnar arrays in on-disk order (see class docs).
    """
    b = blob_u8
    offs = np.asarray(section_offsets, dtype=np.int64).reshape(-1)
    n_sections = offs.size
    if n_sections == 0:
        e = np.zeros(0, dtype=np.int64)
        z16 = np.zeros(0, dtype=np.uint16)
        return ColumnarSections(
            function_name_ptr=np.zeros(0, dtype=np.uint32),
            n_call_targets=e,
            n_variants=e.copy(),
            ct_offsets=np.zeros(1, dtype=np.int64),
            ct_function_name_ptr=np.zeros(0, dtype=np.uint32),
            ct_function_section_ptr=np.zeros(0, dtype=np.uint32),
            ct_type=np.zeros(0, dtype=np.uint8),
            ct_is_matched=np.zeros(0, dtype=bool),
            var_offsets=np.zeros(1, dtype=np.int64),
            var_ref_offset=np.zeros(0, dtype=np.uint32),
            var_data_offset_shifted=np.zeros(0, dtype=np.uint32),
            var_n_calls=e.copy(),
            pce_offsets=np.zeros(1, dtype=np.int64),
            pce_called_idx=z16,
            pce_section_variant_index=z16.copy(),
        )

    # --- section headers --------------------------------------------------
    function_name_ptr = _u32(b, offs).astype(np.uint32)
    n_call_targets = _u16(b, offs + 4)
    n_variants = _u16(b, offs + 6)

    # --- jump table (per-variant n_calls) ----------------------------------
    jt_base = offs + SECTION_HEADER_SIZE
    var_section, jt_addr = _flat_member_addresses(jt_base, n_variants, 2)
    var_n_calls = _u16(b, jt_addr)
    var_offsets = _csr(n_variants)

    # --- call_target table --------------------------------------------------
    # Vector form of ``_padded_jump_table_bytes`` (the scalar helper
    # remains the source of truth; the equivalence test pins the two).
    jt_padded_bytes = ((n_variants + 1) // 2) * 4
    ct_base = jt_base + jt_padded_bytes
    _ct_section, ct_addr = _flat_member_addresses(
        ct_base, n_call_targets, CALL_TARGET_ENTRY_SIZE
    )
    ct_function_name_ptr = _u32(b, ct_addr).astype(np.uint32)
    ct_function_section_ptr = _u32(b, ct_addr + 4).astype(np.uint32)
    ct_flags = _u16(b, ct_addr + 8)
    ct_type = ((ct_flags >> _FLAG_TYPE_SHIFT) & _FLAG_TYPE_MASK).astype(
        np.uint8
    )
    ct_is_matched = ((ct_flags >> _FLAG_IS_MATCHED_BIT) & 1).astype(bool)
    ct_offsets = _csr(n_call_targets)

    # --- variant blocks ------------------------------------------------------
    # Block v starts at: variants_region_start(section) + v_within * 8
    # + 4 * (sum of n_calls of PRIOR variants in the same section).
    region_base = ct_base + CALL_TARGET_ENTRY_SIZE * n_call_targets
    calls_excl = _csr(var_n_calls)[:-1]  # global exclusive cumsum
    # Rebase the global cumsum to each section's first variant.
    section_first_excl = np.zeros(n_sections, dtype=np.int64)
    has_vars = n_variants > 0
    section_first_excl[has_vars] = calls_excl[var_offsets[:-1][has_vars]]
    total_vars = int(var_offsets[-1])
    within_var = (
        np.arange(total_vars, dtype=np.int64) - var_offsets[var_section]
    )
    vb_addr = (
        region_base[var_section]
        + VARIANT_HEADER_SIZE * within_var
        + PER_CALL_ENTRY_SIZE
        * (calls_excl - section_first_excl[var_section])
    )
    var_ref_offset = _u32(b, vb_addr).astype(np.uint32)
    var_data_offset_shifted = _u32(b, vb_addr + 4).astype(np.uint32)

    # --- per-call entries ------------------------------------------------------
    pce_base = vb_addr + VARIANT_HEADER_SIZE
    _pce_var, pce_addr = _flat_member_addresses(
        pce_base, var_n_calls, PER_CALL_ENTRY_SIZE
    )
    pce_called_idx = _u16(b, pce_addr).astype(np.uint16)
    pce_section_variant_index = _u16(b, pce_addr + 2).astype(np.uint16)
    pce_offsets = _csr(var_n_calls)

    # --- optional end-offset validation ------------------------------------
    if section_lengths is not None:
        lens = np.asarray(section_lengths, dtype=np.int64).reshape(-1)
        per_section_calls = np.diff(_csr(var_n_calls)[var_offsets])
        end = (
            ct_base
            + CALL_TARGET_ENTRY_SIZE * n_call_targets
            + VARIANT_HEADER_SIZE * n_variants
            + PER_CALL_ENTRY_SIZE * per_section_calls
        )
        rem = end % SECTION_ALIGNMENT
        end = end + np.where(rem, SECTION_ALIGNMENT - rem, 0)
        bad = np.nonzero(end - offs != lens)[0]
        if bad.size:
            i = int(bad[0])
            raise ValueError(
                f"sections.bin columnar decode drift: section at offset "
                f"{int(offs[i])} computes byte length {int(end[i] - offs[i])} "
                f"but the locator says {int(lens[i])}"
            )

    return ColumnarSections(
        function_name_ptr=function_name_ptr,
        n_call_targets=n_call_targets,
        n_variants=n_variants,
        ct_offsets=ct_offsets,
        ct_function_name_ptr=ct_function_name_ptr,
        ct_function_section_ptr=ct_function_section_ptr,
        ct_type=ct_type,
        ct_is_matched=ct_is_matched,
        var_offsets=var_offsets,
        var_ref_offset=var_ref_offset,
        var_data_offset_shifted=var_data_offset_shifted,
        var_n_calls=var_n_calls,
        pce_offsets=pce_offsets,
        pce_called_idx=pce_called_idx,
        pce_section_variant_index=pce_section_variant_index,
    )
