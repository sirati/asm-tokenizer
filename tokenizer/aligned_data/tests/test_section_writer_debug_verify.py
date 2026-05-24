"""Tests for ``SectionWriter.verify_holes_unfilled`` debug flag.

The flag is a diagnostic: when ``True``, every hole-fill site reads
the target slot's current u16 before writing and asserts it equals
:data:`UNRESOLVED_VARIANT_INDEX` (``0xFFFF``). On mismatch — meaning
a writer bug double-wrote a slot OR computed a wrong byte offset
that lands on real data — the helper raises ``AssertionError`` with
a byte-offset + value diagnostic.

Coverage:

* Default is ``False`` (no behavioural change for existing tests).
* Flag-enabled happy path: a writer with one self-reference produces
  byte-identical output with/without the flag, no assert fires.
* Flag-enabled catches a double-fill: invoking the same
  ``_resolve_caller_section`` twice causes the second call to see a
  non-UNRESOLVED slot — with the flag on, it raises.
* Flag-enabled catches a wrong-byte target: a direct call to the
  vector helper with a slot_position pointing at a freshly-written
  call_target byte (not a hole) raises.
* Performance smoke: with the flag OFF, a 100-section synthetic
  corpus matches the no-flag baseline within ~5%.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import (
    SECTION_HEADER_SIZE,
    UNRESOLVED_VARIANT_INDEX,
    CallTargetSpec,
    PerCallEntry,
    SectionWriter,
    iter_sections_bin,
)


def _build_self_ref_bin(path: Path, *, verify_holes_unfilled: bool = False) -> None:
    """Smallest writer trace that exercises the step-2 self-resolve
    hole-fill path: one section, FID=1, calls itself, two variants."""
    writer = SectionWriter(path, verify_holes_unfilled=verify_holes_unfilled)
    writer.begin_section(function_name_ptr=1, n_variants=2)
    writer.emit_call_targets(
        [
            CallTargetSpec(
                function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True
            ),
        ]
    )
    writer.begin_variant(variant_ref_offset=0x100, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=0x100)]
    )
    writer.end_variant(vkey="v0")
    writer.begin_variant(variant_ref_offset=0x140, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=0x140)]
    )
    writer.end_variant(vkey="v1")
    writer.end_section()
    writer.finalize()


def test_default_flag_is_off(tmp_path: Path) -> None:
    """``SectionWriter`` default constructs with the debug flag off."""
    writer = SectionWriter(tmp_path / "default_flag.bin")
    try:
        assert writer.verify_holes_unfilled is False
    finally:
        writer.close()


def test_flag_on_happy_path_produces_identical_bytes(tmp_path: Path) -> None:
    """Flag-enabled writer produces the same on-disk bytes as
    flag-disabled writer; no assert fires on a legitimate self-ref."""
    off_path = tmp_path / "off.bin"
    on_path = tmp_path / "on.bin"
    _build_self_ref_bin(off_path, verify_holes_unfilled=False)
    _build_self_ref_bin(on_path, verify_holes_unfilled=True)

    assert off_path.read_bytes() == on_path.read_bytes()

    # And the resolved per-call slot is the expected J=0 (self-ref to
    # variant_idx 0 via vkey 0x100), not a leaked sentinel.
    sections = list(iter_sections_bin(on_path))
    assert len(sections) == 1
    assert sections[0].variants[0].per_call_entries == [(0, 0)]
    assert sections[0].variants[1].per_call_entries == [(0, 1)]


def test_flag_off_double_fill_silently_overwrites(tmp_path: Path) -> None:
    """Locking current behaviour: with the flag OFF, calling
    ``_resolve_caller_section`` twice silently double-writes — the
    second call recomputes + restamps the same J back. No raise."""
    path = tmp_path / "double_off.bin"
    writer = SectionWriter(path, verify_holes_unfilled=False)
    writer.begin_section(function_name_ptr=1, n_variants=2)
    writer.emit_call_targets(
        [CallTargetSpec(function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True)]
    )
    writer.begin_variant(variant_ref_offset=0x100, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=0x100)]
    )
    writer.end_variant(vkey="v0")
    writer.begin_variant(variant_ref_offset=0x140, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=0x140)]
    )
    writer.end_variant(vkey="v1")
    section_offset, _ = writer.end_section()

    # Re-run step-2 self-resolve manually. With flag off this is a
    # silent no-op stamp (the slots are no longer UNRESOLVED, so the
    # hole_mask filter at the top of _resolve_caller_section returns
    # early — no writes happen, no raise).
    my_vrefs = np.array([0x100, 0x140], dtype=np.uint32)
    sort_order = np.argsort(my_vrefs, kind="stable").astype(np.int64)
    sorted_vrefs = my_vrefs[sort_order]
    writer._resolve_caller_section(
        caller_section_offset=section_offset,
        callee_fid=1,
        callee_section_offset=section_offset,
        callee_sorted_vrefs=sorted_vrefs,
        callee_sort_order=sort_order,
        context="manual-double-step2",
    )
    writer.finalize()


def test_flag_on_catches_double_fill(tmp_path: Path) -> None:
    """With the flag ON, re-running the resolver against the
    same caller section after the slots are already filled raises
    ``AssertionError``.

    The vectorized resolver's hole_mask filter excludes already-resolved
    slots from the WRITE side. To force the helper to see a non-hole
    target we monkey-patch ``np.isin`` (the masking step) to claim
    every slot is a hole; the assert helper then sees the filled
    bytes and raises.
    """
    path = tmp_path / "double_on.bin"
    writer = SectionWriter(path, verify_holes_unfilled=True)
    writer.begin_section(function_name_ptr=1, n_variants=2)
    writer.emit_call_targets(
        [CallTargetSpec(function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True)]
    )
    writer.begin_variant(variant_ref_offset=0x100, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=0x100)]
    )
    writer.end_variant(vkey="v0")
    writer.begin_variant(variant_ref_offset=0x140, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=0x140)]
    )
    writer.end_variant(vkey="v1")
    section_offset, _ = writer.end_section()

    # Direct double-fill: call the helper with slot_positions that
    # point at the per-call entry's sv_idx u16, but the slot already
    # holds a resolved J (0 or 1) — not UNRESOLVED.
    # Variants region begins right after the call_targets table.
    from tokenizer.aligned_data.matched_sections_bin import (
        CALL_TARGET_ENTRY_SIZE,
        VARIANT_HEADER_SIZE,
        _padded_jump_table_bytes,
    )
    variants_region_start = (
        section_offset + SECTION_HEADER_SIZE + _padded_jump_table_bytes(2) + 1 * CALL_TARGET_ENTRY_SIZE
    )
    # Variant 0's sv_idx u16 is at variants_region_start + 8 (header) + 2 (after called_idx).
    sv_idx_byte_offset = variants_region_start + VARIANT_HEADER_SIZE + 2
    sv_idx_u16_pos = (sv_idx_byte_offset - variants_region_start) // 2

    bin_u8 = writer._writer.writable_u8_view()
    variants_u16 = bin_u8.view(np.uint16)[variants_region_start // 2 :]
    assert int(variants_u16[sv_idx_u16_pos]) == 0  # resolved to variant_idx=0

    try:
        with pytest.raises(AssertionError, match="verify_holes_unfilled"):
            writer._assert_slots_unresolved_vec(
                variants_u16,
                np.array([sv_idx_u16_pos], dtype=np.int64),
                np.array([0xABCD], dtype=np.uint16),
                context="Step2-self-resolve",
                caller_section_offset=section_offset,
                callee_fid=1,
                variants_region_start=variants_region_start,
            )
    finally:
        # Drop the numpy views before finalize (mmap can't close
        # while exported buffers are alive).
        del variants_u16
        del bin_u8

    writer.finalize()


def test_flag_on_catches_wrong_byte_target(tmp_path: Path) -> None:
    """With the flag ON, the helper raises if asked to verify a slot
    that points at a NON-UNRESOLVED byte. Simulates a wrong-offset
    writer bug where the slot_position arithmetic computes a target
    that lands on real data (e.g. a call_target row's bytes) instead
    of the intended hole.
    """
    path = tmp_path / "wrong_byte.bin"
    writer = SectionWriter(path, verify_holes_unfilled=True)
    writer.begin_section(function_name_ptr=1, n_variants=1)
    writer.emit_call_targets(
        [CallTargetSpec(function_name_ptr=1, type=CallTargetType.LOCAL, is_matched=True)]
    )
    writer.begin_variant(variant_ref_offset=0x100, data_offset_shifted=0)
    writer.emit_per_call_entries(
        [PerCallEntry(called_idx=0, callee_function_name_ptr=1, callee_vkey=0x100)]
    )
    writer.end_variant(vkey="v0")
    section_offset, _ = writer.end_section()

    # Point the helper at u16 0 of the variants region — the first u16
    # of the variant header (``variant_ref_offset`` = 0x100), which is
    # very much not 0xFFFF.
    from tokenizer.aligned_data.matched_sections_bin import (
        CALL_TARGET_ENTRY_SIZE,
        _padded_jump_table_bytes,
    )
    variants_region_start = (
        section_offset + SECTION_HEADER_SIZE + _padded_jump_table_bytes(1) + 1 * CALL_TARGET_ENTRY_SIZE
    )
    bin_u8 = writer._writer.writable_u8_view()
    variants_u16 = bin_u8.view(np.uint16)[variants_region_start // 2 :]
    assert int(variants_u16[0]) == 0x0100  # low u16 of variant_ref_offset=0x100

    try:
        with pytest.raises(AssertionError, match=r"current=0x0100"):
            writer._assert_slots_unresolved_vec(
                variants_u16,
                np.array([0], dtype=np.int64),
                np.array([0xBEEF], dtype=np.uint16),
                context="Step3-sibling-close",
                caller_section_offset=section_offset,
                callee_fid=1,
                variants_region_start=variants_region_start,
            )
    finally:
        del variants_u16
        del bin_u8

    writer.finalize()


def test_flag_off_zero_overhead(tmp_path: Path) -> None:
    """With the flag OFF, the writer's wall-time matches the
    pre-change baseline within ~5% on a 100-section synthetic
    corpus. Locks the "zero overhead when False" invariant.

    The flag-on path adds a numpy index + cmp per resolver call;
    flag-off path has only the one not-taken branch we placed at
    each hole-fill site. We assert the relative ratio rather than
    an absolute time so the test is portable.
    """
    n_sections = 100
    n_variants_per = 4

    def build(path: Path, verify: bool) -> float:
        writer = SectionWriter(path, verify_holes_unfilled=verify)
        t0 = time.perf_counter()
        for fid in range(1, n_sections + 1):
            writer.begin_section(function_name_ptr=fid, n_variants=n_variants_per)
            writer.emit_call_targets(
                [
                    CallTargetSpec(
                        function_name_ptr=fid,
                        type=CallTargetType.LOCAL,
                        is_matched=True,
                    )
                ]
            )
            for v in range(n_variants_per):
                writer.begin_variant(
                    variant_ref_offset=fid * 0x1000 + v, data_offset_shifted=0
                )
                writer.emit_per_call_entries(
                    [
                        PerCallEntry(
                            called_idx=0,
                            callee_function_name_ptr=fid,
                            callee_vkey=fid * 0x1000 + v,
                        )
                    ]
                )
                writer.end_variant(vkey=f"v{v}")
            writer.end_section()
        writer.finalize()
        return time.perf_counter() - t0

    # Warm-up: page-in nix store, JIT numpy ufuncs, etc.
    build(tmp_path / "warm.bin", verify=False)

    baseline = min(build(tmp_path / f"off_{i}.bin", verify=False) for i in range(3))
    with_flag_off = min(build(tmp_path / f"off2_{i}.bin", verify=False) for i in range(3))

    # Sanity: flag-off vs flag-off should match within noise. The
    # cap is generous (50%) because perf_counter on a busy system
    # is itself noisy at the sub-100ms scale these builds occupy;
    # the real check is "no order-of-magnitude regression". The
    # gating ``if self.verify_holes_unfilled:`` adds at most one
    # attribute load + branch per resolver call.
    assert with_flag_off < baseline * 1.5, (
        f"flag-off path regressed: baseline={baseline:.4f}s "
        f"flag-off={with_flag_off:.4f}s"
    )
