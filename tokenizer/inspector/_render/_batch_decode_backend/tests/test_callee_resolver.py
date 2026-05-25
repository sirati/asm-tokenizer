"""Tests for InlineCallEntry.callee_section_pointer resolution.

Pins:

* For LOCAL / PLT call sites, the walker passes the call_target's
  ``function_section_ptr`` through the caller-supplied
  ``callee_arm_resolver`` closure; the returned
  :class:`SectionPointerSpec` lands on
  :attr:`InlineCallEntry.callee_section_pointer`.
* For EXTERN call sites, ``callee_section_pointer`` is always
  ``None`` -- there is no body to inline (the UI hides expansion).
* When the resolver returns ``None`` (cross-arm or missing section),
  ``callee_section_pointer`` is ``None`` too.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import CallTarget
from tokenizer.inspector._render._batch_decode_backend._row_walk import (
    render_row_blocks,
)
from tokenizer.inspector._render._protocol import InlineCallEntry
from tokenizer.tokens import Category

from ._row_walk_fixtures import (
    EMPTY_FID_COUNTS,
    EMPTY_FID_SIDECAR,
    EMPTY_NUMBERS,
    LOCAL_FUNC,
    EXT_FUNC,
    PLT_FUNC,
    BLOCK_V2,
    make_fid_table,
    make_result,
    vocab_stub,
)


def _make_call_target(
    *, type_: CallTargetType, function_name_ptr: int, function_section_ptr: int
) -> CallTarget:
    """Bare-minimum :class:`CallTarget` for resolver dispatch tests."""
    ct = MagicMock(spec=CallTarget)
    ct.type = type_
    ct.function_name_ptr = function_name_ptr
    ct.function_section_ptr = function_section_ptr
    return ct


def _walk(
    *,
    tokens: np.ndarray,
    identities: np.ndarray,
    call_targets_section: list[CallTarget],
    callee_arm_resolver,
    fid_sidecar: np.ndarray | None = None,
    per_category_counts: np.ndarray | None = None,
):
    """Shorthand: single-row walk + single-CT span, configurable
    resolver + call_targets_section."""
    numbers_sig, numbers_se = EMPTY_NUMBERS
    fid_sc = fid_sidecar if fid_sidecar is not None else EMPTY_FID_SIDECAR
    pcc = (
        per_category_counts if per_category_counts is not None
        else EMPTY_FID_COUNTS
    )
    return render_row_blocks(
        result=make_result(
            tokens_row=tokens, identities=identities,
            numbers_sig=numbers_sig, numbers_se=numbers_se,
        ),
        row=0, n_axis=0,
        partial_cut_lengths=[int(tokens.shape[0])],
        call_targets_per_ct=[call_targets_section],
        vocab_manager=vocab_stub(),
        fid_table=make_fid_table(
            per_category_counts=pcc, sidecar=fid_sc,
        ),
        line_to_name={1: "callee_one"},
        line_to_provider={42: "libc"},
        callee_arm_resolver=callee_arm_resolver,
    )


def test_local_func_resolves_via_resolver_to_section_pointer() -> None:
    """A LOCAL_FUNC token with counter 0 looks up
    ``call_targets_section[0]`` (the first LOCAL slot), passes its
    ``function_section_ptr`` through the resolver, and lands the
    returned spec on the InlineCallEntry.
    """
    expected_spec = SectionPointerSpec(arm=SectionKind.MATCHED, idx=7)
    captured: list[int] = []

    def resolver(offset: int) -> SectionPointerSpec:
        captured.append(int(offset))
        return expected_spec

    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16),
        identities=np.asarray([0, 0], dtype=np.uint16),
        call_targets_section=[
            _make_call_target(
                type_=CallTargetType.LOCAL,
                function_name_ptr=1,
                function_section_ptr=0xDEAD,
            ),
        ],
        callee_arm_resolver=resolver,
        # FID sidecar must have at least one entry for LOCAL_FUNC.
        fid_sidecar=np.asarray([1], dtype=np.uint32),
        per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
    )
    items = blocks[0].items
    call_entries = [it for it in items if isinstance(it, InlineCallEntry)]
    assert len(call_entries) == 1
    assert call_entries[0].kind is CallTargetType.LOCAL
    assert call_entries[0].callee_section_pointer == expected_spec
    assert captured == [0xDEAD]


def test_plt_func_resolves_via_resolver_to_section_pointer() -> None:
    """A PLT_FUNC token follows the same resolution path as LOCAL_FUNC
    (the resolver is type-agnostic between LOCAL and PLT)."""
    expected_spec = SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=3)

    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, PLT_FUNC, 0], dtype=np.uint16),
        identities=np.asarray([0, 0], dtype=np.uint16),
        call_targets_section=[
            _make_call_target(
                type_=CallTargetType.PLT,
                function_name_ptr=1,
                function_section_ptr=0xBEEF,
            ),
        ],
        callee_arm_resolver=lambda _offset: expected_spec,
        fid_sidecar=np.asarray([1], dtype=np.uint32),
        per_category_counts=np.asarray([[0, 1, 0]], dtype=np.uint32),
    )
    items = blocks[0].items
    call_entries = [it for it in items if isinstance(it, InlineCallEntry)]
    assert len(call_entries) == 1
    assert call_entries[0].kind is CallTargetType.PLT
    assert call_entries[0].callee_section_pointer == expected_spec


def test_ext_func_keeps_callee_section_pointer_none() -> None:
    """EXT_FUNC tokens never inline (no body); the resolver is NOT
    invoked for them and the entry carries
    ``callee_section_pointer=None``."""
    invoked = []

    def resolver(offset: int):
        invoked.append(offset)
        return SectionPointerSpec(arm=SectionKind.MATCHED, idx=99)

    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, EXT_FUNC, 0], dtype=np.uint16),
        identities=np.asarray([0, 0], dtype=np.uint16),
        call_targets_section=[
            _make_call_target(
                type_=CallTargetType.EXTERN,
                function_name_ptr=42,
                function_section_ptr=42,
            ),
        ],
        callee_arm_resolver=resolver,
        fid_sidecar=np.asarray([42], dtype=np.uint32),
        per_category_counts=np.asarray([[0, 0, 1]], dtype=np.uint32),
    )
    items = blocks[0].items
    call_entries = [it for it in items if isinstance(it, InlineCallEntry)]
    assert len(call_entries) == 1
    assert call_entries[0].kind is CallTargetType.EXTERN
    assert call_entries[0].callee_section_pointer is None
    assert invoked == []  # resolver never invoked for EXTERN


def test_local_func_resolver_returning_none_yields_none_pointer() -> None:
    """When the resolver returns ``None`` (cross-arm pointer / missing
    section), the InlineCallEntry's
    ``callee_section_pointer`` is ``None`` -- the UI hides
    expansion."""
    blocks = _walk(
        tokens=np.asarray([BLOCK_V2, LOCAL_FUNC, 0], dtype=np.uint16),
        identities=np.asarray([0, 0], dtype=np.uint16),
        call_targets_section=[
            _make_call_target(
                type_=CallTargetType.LOCAL,
                function_name_ptr=1,
                function_section_ptr=0xC0DE,
            ),
        ],
        callee_arm_resolver=lambda _offset: None,
        fid_sidecar=np.asarray([1], dtype=np.uint32),
        per_category_counts=np.asarray([[1, 0, 0]], dtype=np.uint32),
    )
    items = blocks[0].items
    call_entries = [it for it in items if isinstance(it, InlineCallEntry)]
    assert len(call_entries) == 1
    assert call_entries[0].callee_section_pointer is None
