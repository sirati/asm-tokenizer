import numpy as np
import pytest

from tokenizer.aligned_data.loader.decoded.run_lengths import (
    inline_data_runlength_after_real_tokens,
    run_lengths,
)


def test_single_false_element():
    mask = np.array([False])
    out = run_lengths(mask)
    np.testing.assert_array_equal(out, np.array([0], dtype=np.uint16))
    assert out.dtype == np.uint16


def test_all_false_short():
    mask = np.zeros(5, dtype=bool)
    out = run_lengths(mask)
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.uint16))
    assert out.dtype == np.uint16


def test_all_false_long():
    mask = np.zeros(1024, dtype=bool)
    out = run_lengths(mask)
    np.testing.assert_array_equal(out, np.zeros(1024, dtype=np.uint16))


def test_single_true_run_at_position_1_length_1():
    mask = np.array([False, True, False, False])
    out = run_lengths(mask)
    expected = np.array([0, 1, 0, 0], dtype=np.uint16)
    np.testing.assert_array_equal(out, expected)


def test_single_true_run_at_position_2_length_3():
    mask = np.array([False, False, True, True, True, False, False])
    out = run_lengths(mask)
    expected = np.array([0, 0, 3, 0, 0, 0, 0], dtype=np.uint16)
    np.testing.assert_array_equal(out, expected)


def test_single_true_run_at_tail():
    # Run extends to the final position; the "final True implicitly succeeded
    # by False" branch must close the run correctly.
    mask = np.array([False, True, True, True])
    out = run_lengths(mask)
    expected = np.array([0, 3, 0, 0], dtype=np.uint16)
    np.testing.assert_array_equal(out, expected)


def test_single_true_at_tail_length_1():
    mask = np.array([False, False, True])
    out = run_lengths(mask)
    expected = np.array([0, 0, 1], dtype=np.uint16)
    np.testing.assert_array_equal(out, expected)


def test_multiple_true_runs():
    # Runs at positions (1, length 3) and (5, length 1).
    mask = np.array([False, True, True, True, False, True, False])
    out = run_lengths(mask)
    expected = np.array([0, 3, 0, 0, 0, 1, 0], dtype=np.uint16)
    np.testing.assert_array_equal(out, expected)


def test_runs_adjacent_only_by_one_false():
    mask = np.array([False, True, True, False, True, True, True, False, True])
    out = run_lengths(mask)
    expected = np.array([0, 2, 0, 0, 3, 0, 0, 0, 1], dtype=np.uint16)
    np.testing.assert_array_equal(out, expected)


def test_max_length_under_u16():
    # Run of 65535 fits exactly in u16; surrounding zeros so first pos is False.
    n = 65535
    mask = np.zeros(n + 2, dtype=bool)
    mask[1 : 1 + n] = True
    out = run_lengths(mask)
    assert out[1] == n
    assert out[0] == 0
    assert out[1 + n] == 0
    # All other positions inside the run are zero.
    assert (out[2 : 1 + n] == 0).all()


def test_first_position_true_raises():
    mask = np.array([True, False, True])
    with pytest.raises(AssertionError):
        run_lengths(mask)


def test_first_position_true_length_1_raises():
    mask = np.array([True])
    with pytest.raises(AssertionError):
        run_lengths(mask)


def test_empty_mask_raises_indexerror():
    # The first-position assertion can't be evaluated on a zero-length input;
    # numpy raises naturally. We pin that behavior so a future "silent
    # empty-array return" wouldn't slip through.
    mask = np.zeros(0, dtype=bool)
    with pytest.raises((IndexError, ValueError)):
        run_lengths(mask)


def test_batched_two_rows():
    # The helper documents the `...` axis form; verify it preserves leading
    # batch dims and computes per-row runs independently.
    mask = np.array(
        [
            [False, True, True, False, True],
            [False, False, True, True, True],
        ]
    )
    out = run_lengths(mask)
    expected = np.array(
        [
            [0, 2, 0, 0, 1],
            [0, 0, 3, 0, 0],
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(out, expected)


def test_batched_three_dims():
    mask = np.zeros((2, 2, 6), dtype=bool)
    mask[0, 0, 1:4] = True
    mask[0, 1, 5] = True
    mask[1, 0, 2:6] = True
    mask[1, 1, :] = False
    out = run_lengths(mask)
    expected = np.zeros((2, 2, 6), dtype=np.uint16)
    expected[0, 0, 1] = 3
    expected[0, 1, 5] = 1
    expected[1, 0, 2] = 4
    np.testing.assert_array_equal(out, expected)


def test_inline_data_runlength_after_real_tokens_basic():
    # Synthetic: stream of 7 tokens, real at positions {1, 3, 4}; inline runs
    # immediately after them are of length 1, 0, 2 respectively (the run after
    # the last real token at index 4 is "outside" the slice and is not
    # returned by this helper).
    real_mask = np.array([False, True, False, True, True, False, False])
    runlen = np.array([0, 0, 1, 0, 0, 2, 0], dtype=np.uint16)
    got = inline_data_runlength_after_real_tokens(runlen, real_mask)
    expected = np.array([1, 0, 2], dtype=np.uint16)
    np.testing.assert_array_equal(got, expected)


def test_inline_data_runlength_after_real_tokens_no_reals():
    real_mask = np.zeros(4, dtype=bool)
    runlen = np.zeros(4, dtype=np.uint16)
    got = inline_data_runlength_after_real_tokens(runlen, real_mask)
    assert got.shape == (0,)


def test_inline_data_runlength_after_real_tokens_real_at_tail():
    # Real token at the final position is excluded from the returned slice
    # (real_mask[:-1] drops it). Callers handle "tail real token's trailing
    # run is zero" separately.
    real_mask = np.array([False, True, False, True])
    runlen = np.array([0, 0, 1, 0], dtype=np.uint16)
    got = inline_data_runlength_after_real_tokens(runlen, real_mask)
    expected = np.array([1], dtype=np.uint16)
    np.testing.assert_array_equal(got, expected)


def test_run_lengths_then_wrapper_end_to_end():
    # Stream of 7 tokens: real at {1, 4, 5}, the rest inline; inline-mask
    # satisfies the first-pos-False precondition because position 0 is real.
    real_mask = np.array([False, True, False, False, True, True, False])
    inline_mask = np.array([False, False, True, True, False, False, True])
    runlen = run_lengths(inline_mask)
    expected_runlen = np.array([0, 0, 2, 0, 0, 0, 1], dtype=np.uint16)
    np.testing.assert_array_equal(runlen, expected_runlen)
    after_real = inline_data_runlength_after_real_tokens(runlen, real_mask)
    np.testing.assert_array_equal(after_real, np.array([2, 0, 1], dtype=np.uint16))
