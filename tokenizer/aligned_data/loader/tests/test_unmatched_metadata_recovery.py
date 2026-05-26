"""Regression: ``build_unmatched_function_data`` recovers per-slot axes.

Unmatched sections store one ``_unmatched_data.bin`` record per
variant; each record's ``variant_ref_offset`` points at the variant
identity in ``_variants.bin``. Pre-fix the loader hard-coded
``arch / compiler / compilerversion / opt = "unknown"`` for every
unmatched record, so the inspector's ``label_axes`` rendered
"unknown unknown vunknown -unknown" even when the on-wire variant
tokens carried the real axes.

The fix resolves the per-slot ``variant_ref`` and pulls the
canonical-4 axes off the resolver row directly, so each
:class:`FunctionData` carries its own per-slot identity. Multi-variant
unmatched sections (cross-arch sibling builds of the same function)
therefore surface distinct labels per record.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader._session_parsers import (
    build_unmatched_function_data,
)
from tokenizer.aligned_data.matched_sections_bin import (
    CallTarget,
    Section,
    VariantBlock,
)
from tokenizer.aligned_data.call_target_type import CallTargetType


# Fake resolver: maps the per-slot hex refs to the canonical-4 axes
# the real :func:`variant_resolver.get_variant_by_ref` would return.
def _make_resolve_ref(table: dict[str, dict]) -> "callable":
    def _resolve(ref: str):
        return table.get(ref)
    return _resolve


def _make_section(
    *,
    variant_refs: list[int],
    call_targets: list[CallTarget] | None = None,
) -> Section:
    """A minimal :class:`Section` with one :class:`VariantBlock` per ref."""
    variants = [
        VariantBlock(
            variant_ref_offset=ref,
            data_offset_shifted=0,
            per_call_entries=[],
        )
        for ref in variant_refs
    ]
    return Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=call_targets or [],
        variants=variants,
    )


def _tokens_stub() -> np.ndarray:
    """Three-token instruction stream landing in the rep band (>=272)."""
    return np.array([300, 301, 302], dtype=np.uint16)


def test_unmatched_metadata_recovers_per_slot_axes():
    """Per-slot ``variant_ref`` resolves to canonical-4 axes on each
    :class:`FunctionData`. Cross-arm sibling variants (arm32 + x86)
    therefore surface distinct labels instead of the legacy
    ``"unknown"`` placeholder.
    """
    variant_refs = [0xA, 0xB]  # one arm32 variant, one x86 variant
    section = _make_section(variant_refs=variant_refs)
    resolve_ref = _make_resolve_ref({
        "a": {
            "arch": "arm32", "compiler": "clang",
            "compilerversion": "5.0", "opt": "O0",
            "filename": "die-arm32-clang-5.0-O0",
            "variant_tokens": np.array([400, 401], dtype=np.uint16),
        },
        "b": {
            "arch": "x86", "compiler": "gcc",
            "compilerversion": "13.2.0", "opt": "O2",
            "filename": "die-x86-gcc-13.2.0-O2",
            "variant_tokens": np.array([500, 501], dtype=np.uint16),
        },
    })

    fd_arm = build_unmatched_function_data(
        section,
        "die",
        start=0x100,
        tokens=_tokens_stub(),
        insn_rl=np.array([1, 2], dtype=np.uint8),
        block_rl=np.array([3], dtype=np.uint8),
        variant_slot=0,
        resolve_ref=resolve_ref,
        line_to_name={},
    )
    fd_x86 = build_unmatched_function_data(
        section,
        "die",
        start=0x200,
        tokens=_tokens_stub(),
        insn_rl=np.array([1, 2], dtype=np.uint8),
        block_rl=np.array([3], dtype=np.uint8),
        variant_slot=1,
        resolve_ref=resolve_ref,
        line_to_name={},
    )

    # Per-slot canonical-4 axes — no "unknown" sentinel leakage.
    assert fd_arm.metadata["arch"] == "arm32"
    assert fd_arm.metadata["compiler"] == "clang"
    assert fd_arm.metadata["compilerversion"] == "5.0"
    assert fd_arm.metadata["opt"] == "O0"

    assert fd_x86.metadata["arch"] == "x86"
    assert fd_x86.metadata["compiler"] == "gcc"
    assert fd_x86.metadata["compilerversion"] == "13.2.0"
    assert fd_x86.metadata["opt"] == "O2"

    # ``variant_tokens`` on each :class:`FunctionData` matches the
    # per-slot variant's on-wire stream, not the section's first
    # variant.
    np.testing.assert_array_equal(
        fd_arm.variant_tokens, np.array([400, 401], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        fd_x86.variant_tokens, np.array([500, 501], dtype=np.uint16),
    )

    # ``filename`` from the resolver row threads through too — the
    # inspector's "Variant Header" row reads this for display.
    assert fd_arm.metadata["filename"] == "die-arm32-clang-5.0-O0"
    assert fd_x86.metadata["filename"] == "die-x86-gcc-13.2.0-O2"


def test_unmatched_metadata_falls_back_to_unknown_on_resolver_miss():
    """Legacy datasets without ``_variants.bin`` see ``resolve_ref``
    return ``None`` for every slot; the loader falls back to the
    ``"unknown"`` placeholder so :meth:`VariantInfo.from_function_data_metadata`
    still produces a coherent identity.
    """
    section = _make_section(variant_refs=[0xA])
    resolve_ref = _make_resolve_ref({})  # nothing resolves

    fd = build_unmatched_function_data(
        section,
        "die",
        start=0x100,
        tokens=_tokens_stub(),
        insn_rl=np.array([1, 2], dtype=np.uint8),
        block_rl=np.array([3], dtype=np.uint8),
        variant_slot=0,
        resolve_ref=resolve_ref,
        line_to_name={},
    )

    assert fd.metadata["arch"] == "unknown"
    assert fd.metadata["compiler"] == "unknown"
    assert fd.metadata["compilerversion"] == "unknown"
    assert fd.metadata["opt"] == "unknown"
    # No resolver row -> empty variant_tokens stream.
    assert fd.variant_tokens.shape == (0,)
    assert fd.variant_tokens.dtype == np.uint16


def test_unmatched_metadata_preserves_legacy_section_wide_fields():
    """``variant_refs`` / ``variants`` / ``call_targets`` are still
    section-wide (every variant block), preserving the legacy metadata
    contract. The per-slot axis recovery rides alongside them, NOT
    in place of them.
    """
    call_targets = [
        CallTarget(
            function_name_ptr=42,
            function_section_ptr=0x1000,
            type=CallTargetType.LOCAL,
            is_matched=True,
        ),
    ]
    section = Section(
        function_name_ptr=0,
        section_offset=0,
        call_targets=call_targets,
        variants=[
            VariantBlock(
                variant_ref_offset=0xA,
                data_offset_shifted=0,
                per_call_entries=[(0, 7)],
            ),
            VariantBlock(
                variant_ref_offset=0xB,
                data_offset_shifted=0,
                per_call_entries=[(0, 3)],
            ),
        ],
    )
    resolve_ref = _make_resolve_ref({
        "a": {
            "arch": "arm32", "compiler": "clang",
            "compilerversion": "5.0", "opt": "O0",
            "filename": "f-arm32",
            "variant_tokens": np.array([400], dtype=np.uint16),
        },
        "b": {
            "arch": "x86", "compiler": "gcc",
            "compilerversion": "13", "opt": "O2",
            "filename": "f-x86",
            "variant_tokens": np.array([500], dtype=np.uint16),
        },
    })
    fd = build_unmatched_function_data(
        section,
        "f",
        start=0x100,
        tokens=_tokens_stub(),
        insn_rl=np.array([1, 2], dtype=np.uint8),
        block_rl=np.array([3], dtype=np.uint8),
        variant_slot=0,
        resolve_ref=resolve_ref,
        line_to_name={42: "callee"},
    )
    # Section-wide variant_refs (legacy shape).
    assert fd.metadata["variant_refs"] == ["a", "b"]
    # Both rows resolved into variants[].
    assert len(fd.metadata["variants"]) == 2
    # call_targets aggregated across every variant's per_call_entries.
    assert fd.metadata["call_targets"] == [
        [0, 0x1000, 7, 1],
        [0, 0x1000, 3, 1],
    ]
    assert fd.metadata["called"] == ["callee"]
