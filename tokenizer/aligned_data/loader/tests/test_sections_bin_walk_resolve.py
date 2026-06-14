"""Focused tests for ``_sections_bin_walk.resolve_func_name_or_raise``.

Pins the duplicated-section-marker masking contract of the single
shared FID->name resolver: a section-header FID carrying bit 31 (the
duplicated-section marker) must resolve to the SAME name as its clean
low-31-bit form, while a genuinely-absent (post-mask) FID must still
raise the sidecar-drift error. The masking is a no-op for clean FIDs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.aligned_data.matched_sections_bin import (
    _SECTION_DUPLICATED_BIT,
    _SECTION_FID_MASK,
)
from tokenizer.aligned_data.loader._sections_bin_walk import (
    resolve_func_name_or_raise,
)


def test_resolve_clean_fid_returns_name() -> None:
    """A clean (no bit-31) FID present in the sidecar resolves as-is."""
    name = resolve_func_name_or_raise(
        0x70, {0x70: "foo"}, Path("/tmp/fake_sections.bin"), cursor=16
    )
    assert name == "foo"


def test_resolve_duplicated_fid_masks_bit31_then_resolves() -> None:
    """A FID with the duplicated marker (bit 31) set resolves to the
    SAME name as the masked low-31-bit FID in the sidecar, instead of
    raising sidecar-drift. (busybox hit 0x80000070, clamconf 0x80000022.)
    """
    clean = 0x70
    raw = _SECTION_DUPLICATED_BIT | clean
    assert raw == 0x80000070
    # The sidecar only ever holds the clean line number.
    name = resolve_func_name_or_raise(
        raw, {clean: "foo"}, Path("/tmp/fake_sections.bin"), cursor=16
    )
    assert name == "foo"


def test_resolve_absent_masked_fid_still_raises() -> None:
    """A genuinely-absent FID (after the bit-31 mask) still raises the
    migration-pointing sidecar-drift error -- masking does not weaken
    the real drift guard.
    """
    clean = 0x22
    raw = _SECTION_DUPLICATED_BIT | clean
    # The masked FID (0x22) is NOT in the sidecar -> genuine drift.
    with pytest.raises(ValueError, match="re-run memmap_builder"):
        resolve_func_name_or_raise(
            raw, {0x99: "other"}, Path("/tmp/fake_sections.bin"), cursor=32
        )
    # The error reports the CLEAN (masked) FID, not the raw bit-31 value.
    with pytest.raises(ValueError, match=rf"function_name_ptr={clean}\b"):
        resolve_func_name_or_raise(
            raw, {0x99: "other"}, Path("/tmp/fake_sections.bin"), cursor=32
        )


def test_mask_is_noop_for_every_clean_fid() -> None:
    """Sanity: the mask never alters a FID inside the legal 31-bit
    range, so non-duplicated sections are unaffected.
    """
    for clean in (0, 1, 0x70, _SECTION_FID_MASK):
        name = resolve_func_name_or_raise(
            clean, {clean: "fn"}, Path("/tmp/fake_sections.bin"), cursor=0
        )
        assert name == "fn"
