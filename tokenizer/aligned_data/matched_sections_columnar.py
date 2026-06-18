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
from functools import cached_property

import numpy as np

from .matched_sections_bin import (
    CALL_TARGET_ENTRY_SIZE,
    PER_CALL_ENTRY_SIZE,
    SECTION_ALIGNMENT,
    SECTION_HEADER_SIZE,
    VARIANT_HEADER_SIZE,
    _SECTION_DUPLICATED_BIT,
    _SECTION_FID_MASK,
)


__all__ = [
    "ColumnarSections",
    "LazyColumnarSections",
    "parse_sections_columnar",
    "parse_sections_columnar_lazy",
    "read_n_variants_columnar",
]


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
    """``u32[n_sections]`` -- the section header's FID field (the
    duplicated-marker bit 31 is masked off; the clean line number)."""

    is_duplicated: np.ndarray
    """``bool[n_sections]`` -- the section header's duplicated marker
    (bit 31 of the raw FID field; see
    ``matched_sections_bin._SECTION_DUPLICATED_BIT``)."""

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

    @cached_property
    def sec_of_var(self) -> np.ndarray:
        """``i64[total_variants]`` -- owning SECTION index per flat variant.

        ``np.repeat(arange(n_sections), n_variants)``: the variant-major
        inverse of ``var_offsets``. Corpus-sized (one entry per variant)
        and cols-invariant, so it is memoised once per catalog rather than
        rebuilt by every consumer per batch -- the no-reparse contract.
        Computed on first access (the frozen dataclass stores the cache in
        ``__dict__``, leaving the declared fields immutable).
        """
        return np.repeat(
            np.arange(self.n_variants.size, dtype=np.int64), self.n_variants
        )

    def pce_variant(self) -> np.ndarray:
        """``i64[total_entries]`` -- owning flat variant index per entry.

        The CSR inverse of ``pce_offsets``: entry ``e`` belongs to flat
        variant ``v`` iff ``pce_offsets[v] <= e < pce_offsets[v + 1]``.
        The parser builds exactly this array (to lay out the per-call
        entries in variant order) and then discards it; consumers that
        need the entry-to-variant map call here instead of rebuilding
        an identical ``np.repeat`` -- the no-reparse contract. Derived
        on access (no stored state; the dataclass stays frozen).
        """
        return np.repeat(
            np.arange(self.var_n_calls.size, dtype=np.int64),
            self.var_n_calls,
        )

    def missing_variant_index_count(self) -> int:
        """Per-call entries carrying the ``MISSING_VARIANT_INDEX`` sentinel.

        The whole-catalog count of dropped splice edges (a data-quality
        diagnostic). The eager catalog has every entry materialised, so it
        is one masked sum; the lazy twin overrides this to a touched-bounded
        running total (it never has the full catalog resident).
        """
        from .matched_sections_bin import MISSING_VARIANT_INDEX

        return int(
            (self.pce_section_variant_index == MISSING_VARIANT_INDEX).sum()
        )

    def ensure_sections(self, section_indices) -> None:
        """No-op: the eager catalog already has every section materialised.

        The lazy twin (:class:`...matched_sections_columnar_lazy.
        LazyColumnarSections`) overrides this to fill the touched sections'
        heavy columns on first touch. Defining it here -- as a no-op --
        lets every consumer call ``cols.ensure_sections(secs)`` uniformly
        against EITHER catalog with no type-test branching: the eager
        catalog simply has nothing left to fill.
        """
        return None


def _u16(b: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Gather little-endian u16 values at byte indices ``idx``.

    Combines in ``uint16`` (the field width) rather than ``int64`` --
    each byte is <= 255 so the ``<< 8`` of the high byte never wraps
    the carrier. Callers that do address/size arithmetic on the result
    upcast explicitly.
    """
    return b[idx].astype(np.uint16) | (b[idx + 1].astype(np.uint16) << 8)


def _u32(b: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Gather little-endian u32 values at byte indices ``idx``.

    Combines in ``uint32`` (the field width); the high byte's ``<< 24``
    of a value <= 255 stays inside u32. The result is the field's exact
    unsigned value -- pointers use the full u32 range.
    """
    return (
        b[idx].astype(np.uint32)
        | (b[idx + 1].astype(np.uint32) << 8)
        | (b[idx + 2].astype(np.uint32) << 16)
        | (b[idx + 3].astype(np.uint32) << 24)
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

    ``parent`` is ``int32`` (member counts fit 31 bits at any catalog
    scale) and ``member_addresses`` is ``uint32`` -- every address is a
    valid byte offset into a ``sections.bin`` whose offsets the wire
    format caps at the u32 ``function_section_ptr`` range, so the
    addressing never wraps. Halves both flat arrays from ``int64`` (they
    are sized by the catalog's total member counts -- tens of millions
    of per-call entries at corpus scale).
    """
    offsets = _csr(counts)
    total = int(offsets[-1])
    parent = np.repeat(np.arange(counts.size, dtype=np.int32), counts)
    within = np.arange(total, dtype=np.uint32) - offsets[parent].astype(
        np.uint32
    )
    addr = base_per_parent[parent].astype(np.uint32) + np.uint32(stride) * within
    return parent, addr


def read_n_variants_columnar(
    blob_u8: np.ndarray,
    section_offsets: np.ndarray,
) -> np.ndarray:
    """Per-section variant count at ``section_offsets`` -- header-only.

    The ``n_variants`` field is the section header's third member
    (``<IHH`` = raw_fid, n_call_targets, n_variants), i.e. a u16 at
    ``offset + 6`` -- so the whole batch's counts are one vectorized
    :func:`_u16` gather, reading NO jump table / call_target table /
    variant block. The header-only twin of :func:`parse_sections_columnar`
    for a consumer that needs only ``n_variants`` (the vector_batch
    resolve's variant-sampling driver): it pays neither the full scalar
    :func:`...matched_sections_bin.parse_section_bin` object build per
    section NOR the columnar decoder's jump-table / table / per-call
    walk, both of which are dead work when only the count is wanted.

    The ``+ 6`` field offset is the same one
    :func:`parse_sections_columnar` reads (``_u16(b, offs + 6)``); both
    mirror the scalar ``<IHH`` unpack in
    :func:`...matched_sections_bin.parse_section_bin`. ``blob_u8`` is the
    whole-file uint8 array; ``section_offsets`` are absolute file
    offsets. Returns ``i64[len(section_offsets)]`` parallel to the input.
    """
    offs = np.asarray(section_offsets, dtype=np.int64).reshape(-1)
    if offs.size == 0:
        return np.zeros(0, dtype=np.int64)
    return _u16(blob_u8, offs + 6).astype(np.int64)


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
            is_duplicated=np.zeros(0, dtype=bool),
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
    # Counts widen to int64 immediately: they are per-section (tiny) and
    # feed byte-address/size arithmetic below where a u16 carrier would
    # wrap (e.g. CALL_TARGET_ENTRY_SIZE * n_call_targets). The big flat
    # arrays this module narrows are the per-member ADDRESS columns, not
    # these.
    raw_function_name_ptr = _u32(b, offs).astype(np.uint32)
    # Bit 31 is the per-section duplicated marker; mask it off so the FID
    # column is the clean line number, and surface the bit separately.
    is_duplicated = (
        raw_function_name_ptr & np.uint32(_SECTION_DUPLICATED_BIT)
    ).astype(bool)
    function_name_ptr = (
        raw_function_name_ptr & np.uint32(_SECTION_FID_MASK)
    ).astype(np.uint32)
    n_call_targets = _u16(b, offs + 4).astype(np.int64)
    n_variants = _u16(b, offs + 6).astype(np.int64)

    # --- jump table (per-variant n_calls) ----------------------------------
    jt_base = offs + SECTION_HEADER_SIZE
    var_section, jt_addr = _flat_member_addresses(jt_base, n_variants, 2)
    var_n_calls = _u16(b, jt_addr).astype(np.int64)
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
        np.arange(total_vars, dtype=np.uint32)
        - var_offsets[var_section].astype(np.uint32)
    )
    # vb_addr is a valid in-blob byte offset (<= the u32 ptr range), so
    # it lands in uint32 -- halving this total_vars-sized address array.
    vb_addr = (
        region_base[var_section].astype(np.uint32)
        + np.uint32(VARIANT_HEADER_SIZE) * within_var
        + np.uint32(PER_CALL_ENTRY_SIZE)
        * (calls_excl - section_first_excl[var_section]).astype(np.uint32)
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
        is_duplicated=is_duplicated,
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
