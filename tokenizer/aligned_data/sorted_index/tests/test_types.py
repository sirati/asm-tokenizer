"""Tests for sorted_index._types: LengthReduction + cross-binary types."""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.sorted_index._types import (
    LengthReduction,
    MultiBinaryBatchDecodeResult,
    MultiBinarySectionPointer,
    ReductionKind,
)


# ---------------------------------------------------------------------------
# filename_tag
# ---------------------------------------------------------------------------


def test_filename_tag_max() -> None:
    assert LengthReduction(ReductionKind.MAX).filename_tag() == "max"


@pytest.mark.parametrize(
    "percentile, expected",
    [
        (1, "p01"),
        (5, "p05"),
        (50, "p50"),
        (95, "p95"),
        (99, "p99"),
    ],
)
def test_filename_tag_percentile_zero_padded(percentile: int, expected: str) -> None:
    assert (
        LengthReduction(ReductionKind.PERCENTILE, percentile).filename_tag()
        == expected
    )


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


def test_reduce_max_returns_max_int() -> None:
    lengths = np.array([3, 7, 1, 5], dtype=np.uint32)
    assert LengthReduction(ReductionKind.MAX).reduce(lengths) == 7


def test_reduce_percentile_lower_method_returns_input_value() -> None:
    # method="lower" guarantees the returned value is one of the inputs.
    lengths = np.array([3, 7, 1, 5], dtype=np.uint32)
    # Sorted: [1, 3, 5, 7]. 50th percentile lower -> index floor(0.5*3)=1 -> 3.
    assert LengthReduction(ReductionKind.PERCENTILE, 50).reduce(lengths) == 3
    # 75th percentile lower -> index floor(0.75*3)=2 -> 5.
    assert LengthReduction(ReductionKind.PERCENTILE, 75).reduce(lengths) == 5


def test_reduce_empty_input_returns_zero_for_both_kinds() -> None:
    empty = np.empty(0, dtype=np.uint32)
    assert LengthReduction(ReductionKind.MAX).reduce(empty) == 0
    assert LengthReduction(ReductionKind.PERCENTILE, 50).reduce(empty) == 0


def test_reduce_returns_python_int_not_numpy_scalar() -> None:
    lengths = np.array([10], dtype=np.uint32)
    result_max = LengthReduction(ReductionKind.MAX).reduce(lengths)
    result_pct = LengthReduction(ReductionKind.PERCENTILE, 50).reduce(lengths)
    assert type(result_max) is int
    assert type(result_pct) is int


# ---------------------------------------------------------------------------
# __post_init__ validation
# ---------------------------------------------------------------------------


def test_post_init_rejects_max_with_percentile() -> None:
    with pytest.raises(ValueError, match="MAX reduction takes no percentile"):
        LengthReduction(ReductionKind.MAX, percentile=5)


def test_post_init_rejects_percentile_without_value() -> None:
    with pytest.raises(ValueError, match="PERCENTILE reduction requires"):
        LengthReduction(ReductionKind.PERCENTILE)


@pytest.mark.parametrize("bad", [0, 100, -1, 101, 1000])
def test_post_init_rejects_out_of_range_percentile(bad: int) -> None:
    with pytest.raises(ValueError, match="percentile must be in"):
        LengthReduction(ReductionKind.PERCENTILE, bad)


def test_post_init_accepts_boundary_percentiles() -> None:
    # boundaries 1 and 99 are accepted; 100 must be funnelled through
    # parse_reduction (canonicalises to MAX).
    LengthReduction(ReductionKind.PERCENTILE, 1)
    LengthReduction(ReductionKind.PERCENTILE, 99)


# ---------------------------------------------------------------------------
# Hashability + dict-key usability
# ---------------------------------------------------------------------------


def test_length_reduction_is_hashable_and_dict_keyable() -> None:
    keys = {
        LengthReduction(ReductionKind.MAX): "max",
        LengthReduction(ReductionKind.PERCENTILE, 95): "p95",
        LengthReduction(ReductionKind.PERCENTILE, 5): "p05",
    }
    # Same value -> same hash bucket; lookup succeeds.
    assert keys[LengthReduction(ReductionKind.MAX)] == "max"
    assert keys[LengthReduction(ReductionKind.PERCENTILE, 95)] == "p95"
    assert keys[LengthReduction(ReductionKind.PERCENTILE, 5)] == "p05"
    assert len(keys) == 3


def test_length_reduction_equality() -> None:
    assert LengthReduction(ReductionKind.MAX) == LengthReduction(ReductionKind.MAX)
    assert LengthReduction(ReductionKind.PERCENTILE, 50) == LengthReduction(
        ReductionKind.PERCENTILE, 50
    )
    assert LengthReduction(ReductionKind.PERCENTILE, 50) != LengthReduction(
        ReductionKind.PERCENTILE, 95
    )
    assert LengthReduction(ReductionKind.MAX) != LengthReduction(
        ReductionKind.PERCENTILE, 99
    )


# ---------------------------------------------------------------------------
# MultiBinarySectionPointer
# ---------------------------------------------------------------------------


def test_multi_binary_section_pointer_is_frozen() -> None:
    ptr = MultiBinarySectionPointer(
        binary_name="coreutils",
        section_pointer=SectionPointerSpec(arm=SectionKind.MATCHED, idx=7),
    )
    with pytest.raises((AttributeError, Exception)):
        ptr.binary_name = "other"  # type: ignore[misc]


def test_multi_binary_section_pointer_is_hashable() -> None:
    ptr1 = MultiBinarySectionPointer(
        binary_name="coreutils",
        section_pointer=SectionPointerSpec(arm=SectionKind.MATCHED, idx=7),
    )
    ptr2 = MultiBinarySectionPointer(
        binary_name="coreutils",
        section_pointer=SectionPointerSpec(arm=SectionKind.MATCHED, idx=7),
    )
    ptr3 = MultiBinarySectionPointer(
        binary_name="other",
        section_pointer=SectionPointerSpec(arm=SectionKind.MATCHED, idx=7),
    )
    assert hash(ptr1) == hash(ptr2)
    assert ptr1 == ptr2
    assert ptr1 != ptr3
    # Usable as dict key.
    bag = {ptr1: 1, ptr3: 2}
    assert bag[ptr2] == 1


# ---------------------------------------------------------------------------
# MultiBinaryBatchDecodeResult
# ---------------------------------------------------------------------------


def _make_dummy_inner() -> BatchDecodeResult:
    """Minimal BatchDecodeResult instance for frozen-check purposes."""
    return BatchDecodeResult(
        tokens=np.zeros((1, 1), dtype=np.uint16),
        identities=np.zeros(0, dtype=np.uint16),
        identity_row_offsets=np.zeros(2, dtype=np.uint32),
        numbers_significant=np.zeros(0, dtype=np.uint64),
        numbers_sign_exponent=np.zeros(0, dtype=np.uint32),
        number_row_offsets=np.zeros(2, dtype=np.uint32),
        batch_idx_to_section_variant=np.zeros((1, 2), dtype=np.uint32),
        fid_sidecar=None,
        fid_row_offsets=None,
        fid_per_category_counts=None,
        block_runlength=None,
        block_runlength_row_offsets=None,
        insn_runlength=None,
        insn_runlength_row_offsets=None,
        intermediate=None,
    )


def test_multi_binary_batch_decode_result_is_frozen() -> None:
    inner = _make_dummy_inner()
    result = MultiBinaryBatchDecodeResult(
        inner=inner,
        binary_id_per_row=np.zeros(1, dtype=np.uint32),
        binary_names=["coreutils"],
    )
    with pytest.raises((AttributeError, Exception)):
        result.binary_names = ["other"]  # type: ignore[misc]
    # And the inner content stays addressable.
    assert result.inner is inner
    assert result.binary_names == ["coreutils"]
    assert result.binary_id_per_row.dtype == np.uint32
