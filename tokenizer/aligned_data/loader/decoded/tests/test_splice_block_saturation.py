"""Block-identity compaction kills u16 saturation under deep splices.

Plan reference: ``## Locked-in decisions`` item 28 (a) -- the old
running-max + rebase scheme produced a per-Category counter that grew
without bound across the splice tree; at corpus scale (smoke at 50
funcs/binary depth 3 emitted 9180 sentinels) it saturated u16. The new
design concatenates verbatim and runs first-occurrence-wins
compaction at the top, so the OUTPUT identity space is bounded by the
PHYSICAL number of distinct identities in the spliced view, not by
``depth × per-function counter``.

The test below builds a synthetic deep splice where the unbounded
running-max scheme would have crossed the u16 ceiling, and confirms
the new compaction produces a dense ``[0, N)`` range with N == the
number of distinct staged identities.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.loader.decoded.extract import _StagingDecoded
from tokenizer.aligned_data.loader.decoded.splice import splice_with_callees
from tokenizer.tokens import Category


class _StubCallTarget:
    def __init__(self, function_section_ptr: int) -> None:
        self.function_section_ptr = function_section_ptr
        self.is_matched = True


class _StubVariant:
    """Single-variant default mirroring ``test_splice._StubVariant``."""

    def __init__(
        self,
        per_call_entries: List[Tuple[int, int]],
        variant_ref_offset: int = 0,
    ) -> None:
        self.per_call_entries = per_call_entries
        self.variant_ref_offset = variant_ref_offset


class _StubSection:
    def __init__(self, call_targets: List[_StubCallTarget]) -> None:
        self.call_targets = call_targets
        # Default single-variant section: every call_target maps to
        # J=0 in the only variant. Mirrors test_splice._StubSection so
        # the saturation tests pass through the new walker contract
        # without behavior change.
        self.variants = [
            _StubVariant(
                per_call_entries=[(i, 0) for i in range(len(call_targets))],
                variant_ref_offset=0,
            )
        ]


_DEFAULT_PRIMARY_VARIANT_IDX = 0
_DEFAULT_SELECTION_VKEYS = frozenset({0})


def _make_staging_with_block_ids(
    *, func_name: str, real_tokens, block_ids
) -> _StagingDecoded:
    identities: Dict[Category, np.ndarray] = {
        c: np.empty(0, dtype=np.uint16) for c in Category
    }
    identities[Category.BLOCK] = np.array(block_ids, dtype=np.uint16)
    return _StagingDecoded(
        real_tokens=np.array(real_tokens, dtype=np.uint16),
        identities=identities,
        numbers_significant=np.empty(0, dtype=np.uint64),
        numbers_sign_exponent=np.empty(0, dtype=np.uint32),
        func_name=func_name,
        metadata={},
    )


def test_block_compaction_no_saturation_at_depth_3() -> None:
    """A 4-level chain where each level emits many BLOCK identities
    saturating the per-function counter at 50 (rare-but-possible in a
    big function). Old rebase scheme: depth-3 callee's identities get
    offset by ~150 in the running max; with deeper splices the running
    max accumulates beyond u16. Compaction collapses the FOUR functions'
    BLOCK arrays to a dense ``[0, K)`` where K == count of distinct
    BLOCK identities across the whole tree.

    The synthetic block ids are chosen so the depth=3 chain has 4
    physical functions × 50 ids = 200 staged slots but only 50 distinct
    values (each function emits ids 0..49); after compaction the output
    array length is 200 but the value space is exactly ``[0, 50)`` --
    far short of the u16 ceiling.
    """
    block_ids = list(range(50))
    a = _make_staging_with_block_ids(
        func_name="A", real_tokens=[300], block_ids=block_ids
    )
    b = _make_staging_with_block_ids(
        func_name="B", real_tokens=[400], block_ids=block_ids
    )
    c = _make_staging_with_block_ids(
        func_name="C", real_tokens=[500], block_ids=block_ids
    )
    d = _make_staging_with_block_ids(
        func_name="D", real_tokens=[600], block_ids=block_ids
    )

    table: Dict[int, Tuple[_StagingDecoded, _StubSection]] = {
        100: (a, _StubSection([_StubCallTarget(200)])),
        200: (b, _StubSection([_StubCallTarget(300)])),
        300: (c, _StubSection([_StubCallTarget(400)])),
        400: (d, _StubSection([])),
    }

    def decode_callee_to_staging(
        offset: int, arm: str, callee_variant_index: int
    ):
        # Single-variant section default: J must always be 0.
        assert callee_variant_index == 0, (
            f"saturation test stub expects callee_variant_index=0; "
            f"got {callee_variant_index}"
        )
        return table[offset]

    def is_callee_present(offset: int, arm: str) -> bool:
        return offset in table

    out = splice_with_callees(
        root_staging=a,
        root_arm="matched",
        root_section=_StubSection([_StubCallTarget(200)]),
        root_section_offset=100,
        decode_callee_to_staging=decode_callee_to_staging,
        is_callee_present=is_callee_present,
        max_depth=3,
        primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
        initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
    )

    block = out.identities[Category.BLOCK]
    # Staged length: 4 functions x 50 ids = 200 slots.
    assert block.size == 200
    assert block.dtype == np.uint16
    # Each function's block_ids list is identical, so compaction maps
    # all four function's slots through the same first-occurrence
    # mapping -> the output is the same [0..49] pattern repeated 4x.
    expected = np.tile(np.arange(50, dtype=np.uint16), 4)
    np.testing.assert_array_equal(block, expected)
    # Compacted value space is dense [0, 50) -- well under the u16
    # ceiling. A regression that re-introduced running-max rebase would
    # push the max past 200 (or higher under deeper / wider corpora).
    assert int(block.max()) == 49


def test_block_compaction_disjoint_ids_dense_output() -> None:
    """Deep chain where each level's BLOCK ids are DISJOINT (no overlap).

    Worst-case input: every staged identity is unique, so compaction
    produces a 1:1 mapping. The output length equals total staged
    slots, and the value range equals ``[0, total_slots)``. This is
    where the u16 ceiling is closest to mattering on real corpora; the
    test pins that the algorithm doesn't artificially limit beyond
    the staged count.
    """
    # Four levels, 50 ids each, but each level's ids are in a disjoint
    # 50-wide window: A=0..49, B=100..149, C=200..249, D=300..349.
    levels = [list(range(base, base + 50)) for base in (0, 100, 200, 300)]
    stagings = [
        _make_staging_with_block_ids(
            func_name=f"L{i}", real_tokens=[300 + i], block_ids=ids
        )
        for i, ids in enumerate(levels)
    ]
    table: Dict[int, Tuple[_StagingDecoded, _StubSection]] = {
        100: (stagings[0], _StubSection([_StubCallTarget(200)])),
        200: (stagings[1], _StubSection([_StubCallTarget(300)])),
        300: (stagings[2], _StubSection([_StubCallTarget(400)])),
        400: (stagings[3], _StubSection([])),
    }

    def decode_callee_to_staging(
        offset: int, arm: str, callee_variant_index: int
    ):
        # Single-variant section default: J must always be 0.
        assert callee_variant_index == 0, (
            f"saturation test stub expects callee_variant_index=0; "
            f"got {callee_variant_index}"
        )
        return table[offset]

    def is_callee_present(offset: int, arm: str) -> bool:
        return offset in table

    out = splice_with_callees(
        root_staging=stagings[0],
        root_arm="matched",
        root_section=_StubSection([_StubCallTarget(200)]),
        root_section_offset=100,
        decode_callee_to_staging=decode_callee_to_staging,
        is_callee_present=is_callee_present,
        max_depth=3,
        primary_variant_idx=_DEFAULT_PRIMARY_VARIANT_IDX,
        initial_selection_vkeys=_DEFAULT_SELECTION_VKEYS,
    )
    block = out.identities[Category.BLOCK]
    assert block.size == 200
    assert block.dtype == np.uint16
    # All 200 staged ids are unique -> compaction produces [0, 200).
    np.testing.assert_array_equal(block, np.arange(200, dtype=np.uint16))
    assert int(block.max()) == 199
