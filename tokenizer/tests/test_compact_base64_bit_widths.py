"""Round-trip regression tests for ``ndarray_to_base64`` / ``base64_to_ndarray``.

The historical ``_pack_bits_vec`` routine carried a documented "bugged for
larger bits" disclaimer that capped its allowed bit-width at 12 — but in
practice the *exact* width 11 (and only 11) also corrupted values that
straddled a 32-bit word boundary. That corruption surfaced as silent
mapping-table breakage during vocab unification (per-binary id → unified id
mappings whose codomain happened to fit in 11 bits got high bits dropped),
which in turn led memmap_builder to emit token streams with inline-digit
bytes at function-body starts.

This test pins:

* every supported bit-width (2 through 32) round-trips byte-for-byte under
  several deterministic data shapes,
* the concrete 11-bit cross-word corpus pattern that surfaced the bug.

The vec packer has since been rewritten to cover ALL widths (the 11-bit
scalar detour is gone); byte-identity against the legacy scalar writer is
pinned in ``test_compact_base64_pack_vec.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tokenizer.compact_base64_utils import (
    base64_to_ndarray,
    base64_to_ndarray_vec,
    ndarray_to_base64,
)


@pytest.mark.parametrize("bits", list(range(2, 21)))
@pytest.mark.parametrize("size", [50, 393, 1000])
def test_round_trip_at_every_bit_width(bits: int, size: int) -> None:
    """Random ndarrays at each supported bit-width round-trip exactly."""
    max_val = (1 << bits) - 1
    rng = np.random.default_rng(seed=bits * 1000 + size)
    values = rng.integers(0, max_val + 1, size=size, dtype=np.int64)
    values[0] = max_val   # force the max so the encoder picks ``bits``

    encoded = ndarray_to_base64(values)
    decoded_scalar = base64_to_ndarray(encoded)
    decoded_vec = base64_to_ndarray_vec(encoded)

    np.testing.assert_array_equal(decoded_scalar, values)
    np.testing.assert_array_equal(decoded_vec, values)


def test_bits_11_with_cross_word_boundary_value() -> None:
    """Concrete regression: legacy-corpus mapping arrays produced bits=11 +
    a value at the cross-word position (256/257) that the vec path corrupted
    silently (272 → 16 = lost high 6 bits)."""
    values = np.zeros(393, dtype=np.int32)
    values[:256] = np.arange(256)
    values[256] = 264   # block_v2 unified id
    values[257] = 272   # Block_Def unified id (used to round-trip as 16)
    values[258] = 530
    values[262] = 1000  # used to round-trip as 488

    encoded = ndarray_to_base64(values)
    decoded = base64_to_ndarray(encoded)

    np.testing.assert_array_equal(decoded.astype(np.int64), values.astype(np.int64))
    assert int(decoded[257]) == 272, (
        "regression: bits=11 vec packer lost high bits of value at 257"
    )
    assert int(decoded[262]) == 1000, (
        "regression: bits=11 vec packer lost high bit of value at 262"
    )
