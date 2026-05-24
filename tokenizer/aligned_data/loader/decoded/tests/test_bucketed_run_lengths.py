"""Tests for :class:`BucketedRunLengthCollector`.

The collector is a pow2-bucketed batched dispatcher around
:func:`run_lengths`. The invariant under test in every case is
byte-identical equivalence between the bucketed result and the
single-call :func:`run_lengths` API.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.decoded.run_lengths import run_lengths


def _random_mask(rng: np.random.Generator, length: int) -> np.ndarray:
    """1D bool mask of given length with mask[0] == False."""
    mask = rng.random(length) < 0.4
    mask[0] = False
    return mask


# ---------------------------------------------------------------------------
# Single-mask round trip
# ---------------------------------------------------------------------------


def test_single_mask_round_trip() -> None:
    """The bucketed result for one mask equals the single-call result."""
    rng = np.random.default_rng(0)
    mask = _random_mask(rng, 17)
    collector = BucketedRunLengthCollector()
    h = collector.add(mask)
    results = collector.flush()
    expected = run_lengths(mask)
    np.testing.assert_array_equal(results[h], expected)
    assert results[h].dtype == np.uint16


def test_length_one_mask() -> None:
    """A mask of length 1 is always all-False (precondition); the
    bucket is pow2=1, the result is ``[0]``."""
    mask = np.array([False])
    collector = BucketedRunLengthCollector()
    h = collector.add(mask)
    results = collector.flush()
    np.testing.assert_array_equal(results[h], np.array([0], dtype=np.uint16))


def test_exact_pow2_length_mask() -> None:
    """A mask of length exactly ``2**k`` (no padding overhead).

    Mask of length 8 lands in bucket 8; padding contributes zero rows
    in the bucket buffer's column space."""
    mask = np.array(
        [False, True, True, False, False, True, False, False], dtype=bool
    )
    collector = BucketedRunLengthCollector()
    h = collector.add(mask)
    results = collector.flush()
    expected = run_lengths(mask)
    np.testing.assert_array_equal(results[h], expected)


# ---------------------------------------------------------------------------
# Multiple masks
# ---------------------------------------------------------------------------


def test_multiple_masks_same_bucket() -> None:
    """Two masks of length 5 + 7 both land in bucket 8; both results
    must equal their respective single-call values."""
    rng = np.random.default_rng(1)
    masks = [_random_mask(rng, 5), _random_mask(rng, 7), _random_mask(rng, 8)]
    collector = BucketedRunLengthCollector()
    handles = [collector.add(m) for m in masks]
    results = collector.flush()
    for h, m in zip(handles, masks):
        np.testing.assert_array_equal(results[h], run_lengths(m))


def test_multiple_masks_multiple_buckets() -> None:
    """Stress: 100 seeds, random sizes in [1, 1024]. Every result must
    byte-match its single-call equivalent."""
    for seed in range(100):
        rng = np.random.default_rng(seed)
        n_masks = int(rng.integers(1, 32))
        masks = [
            _random_mask(rng, int(rng.integers(1, 1025)))
            for _ in range(n_masks)
        ]
        collector = BucketedRunLengthCollector()
        handles = [collector.add(m) for m in masks]
        results = collector.flush()
        for h, m in zip(handles, masks):
            np.testing.assert_array_equal(results[h], run_lengths(m))


# ---------------------------------------------------------------------------
# Postpend-False invariance
# ---------------------------------------------------------------------------


def test_postpend_false_invariance() -> None:
    """Padding a mask with trailing ``False`` values is the engine that
    makes the bucket scheme correct. Verify directly: pad a mask with
    a random tail and slice the result back to ``[:L]``; it must match
    the un-padded single call.

    The collector relies on this invariant internally; this test pins
    it from the outside so a regression in :func:`run_lengths` would
    immediately surface.
    """
    rng = np.random.default_rng(42)
    for seed in range(50):
        rng = np.random.default_rng(seed)
        original_len = int(rng.integers(1, 256))
        tail_len = int(rng.integers(0, 256))
        original = _random_mask(rng, original_len)
        padded = np.zeros(original_len + tail_len, dtype=bool)
        padded[:original_len] = original
        np.testing.assert_array_equal(
            run_lengths(padded)[:original_len], run_lengths(original)
        )


# ---------------------------------------------------------------------------
# API edge cases
# ---------------------------------------------------------------------------


def test_flush_no_adds_returns_empty_dict() -> None:
    """No :meth:`add` calls -> flush returns ``{}``."""
    collector = BucketedRunLengthCollector()
    assert collector.flush() == {}


def test_flush_twice_second_is_noop() -> None:
    """Two consecutive flushes: second returns ``{}``."""
    collector = BucketedRunLengthCollector()
    collector.add(np.array([False, True, False]))
    first = collector.flush()
    assert len(first) == 1
    second = collector.flush()
    assert second == {}


def test_flush_clears_state_for_reuse() -> None:
    """After flush, internal storage is cleared; a second add/flush
    cycle works as if the collector were fresh. Prevents memory leaks
    in long-running batch loops."""
    collector = BucketedRunLengthCollector()
    h1 = collector.add(np.array([False, True, False]))
    r1 = collector.flush()
    np.testing.assert_array_equal(
        r1[h1], np.array([0, 1, 0], dtype=np.uint16)
    )
    # Second cycle on the SAME collector instance.
    h2 = collector.add(np.array([False, False, True, True]))
    r2 = collector.flush()
    np.testing.assert_array_equal(
        r2[h2], np.array([0, 0, 2, 0], dtype=np.uint16)
    )
    # Handle namespace is independent per cycle (resets to 0 after flush).
    assert h2 == 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_inputs_same_handles_and_outputs() -> None:
    """Same add order -> same handles + byte-identical outputs across
    two independent collector instances."""
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    masks_a = [_random_mask(rng_a, int(rng_a.integers(1, 256))) for _ in range(20)]
    masks_b = [_random_mask(rng_b, int(rng_b.integers(1, 256))) for _ in range(20)]
    for m_a, m_b in zip(masks_a, masks_b):
        np.testing.assert_array_equal(m_a, m_b)

    c_a = BucketedRunLengthCollector()
    c_b = BucketedRunLengthCollector()
    h_a = [c_a.add(m) for m in masks_a]
    h_b = [c_b.add(m) for m in masks_b]
    assert h_a == h_b
    r_a = c_a.flush()
    r_b = c_b.flush()
    for h in h_a:
        np.testing.assert_array_equal(r_a[h], r_b[h])


# ---------------------------------------------------------------------------
# Precondition rejection
# ---------------------------------------------------------------------------


def test_add_rejects_non_1d() -> None:
    collector = BucketedRunLengthCollector()
    with pytest.raises(ValueError, match="1D"):
        collector.add(np.zeros((2, 3), dtype=bool))


def test_add_rejects_wrong_dtype() -> None:
    collector = BucketedRunLengthCollector()
    with pytest.raises(ValueError, match="bool"):
        collector.add(np.zeros(3, dtype=np.uint8))


def test_add_rejects_zero_length() -> None:
    """Mirrors :func:`run_lengths`'s zero-length rejection."""
    collector = BucketedRunLengthCollector()
    with pytest.raises(ValueError, match="zero-length"):
        collector.add(np.zeros(0, dtype=bool))


def test_add_rejects_first_position_true() -> None:
    """``mask[0] == True`` violates :func:`run_lengths`'s precondition."""
    collector = BucketedRunLengthCollector()
    with pytest.raises(ValueError, match="mask\\[0\\] must be False"):
        collector.add(np.array([True, False]))
