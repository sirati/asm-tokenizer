"""Tests for :mod:`tokenizer.aligned_data.loader.batch_decode._auto_size`.

Single concern: assert that :func:`compute_auto_sizes` (and its
pure-data sibling :func:`auto_size_from_resolved`) reduce a request's
section list to the tightest ``(num_variants_per_section, context_len)``
pair without skipping any variant -- the loader's natural sizing.

Mirrors the fake-session style used in :mod:`test_resolve_pointers` so
the unit test stays in-tree with the rest of the batch_decode unit
suite. A final smoke test wires the helper into the real
:func:`batch_decode` end-to-end via the shared
:mod:`_session_fixture` corpus to prove the auto-sized pair lets the
pipeline complete without :class:`IndexError` / shape mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    batch_decode,
    compute_auto_sizes,
)
from tokenizer.aligned_data.loader.batch_decode._auto_size import (
    CONTEXT_LEN_HEADROOM,
    SizingSpec,
    auto_size_from_resolved,
)
from tokenizer.aligned_data.loader.batch_decode._resolve_pointers import (
    ResolvedSection,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.function_data import FunctionData
from tokenizer.aligned_data.loader.matched_function import MatchedFunction
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.loader.tests._session_fixture import (
    build_synthetic_binary,
)
from tokenizer.aligned_data.matched_sections_bin import (
    Section,
    VariantBlock,
)


# ---------------------------------------------------------------------------
# Fake-session helpers (mirror test_resolve_pointers.py so the helper is
# tested at the same boundary the resolver itself is).
# ---------------------------------------------------------------------------


def _make_variant(ref_offset: int) -> VariantBlock:
    return VariantBlock(
        variant_ref_offset=ref_offset,
        data_offset_shifted=0,
        per_call_entries=[],
    )


def _make_function_data(
    name: str, n_body_tokens: int, n_variant_tokens: int = 0
) -> FunctionData:
    return FunctionData(
        func_name=name,
        metadata={"arch": "x86_64", "compiler": "gcc", "opt": "O2"},
        tokens=np.full(n_body_tokens, 300, dtype=np.uint16),
        insn_runlength=np.array([n_body_tokens], dtype=np.uint32),
        block_runlength=np.array([n_body_tokens], dtype=np.uint32),
        variant_tokens=np.zeros(n_variant_tokens, dtype=np.uint16),
    )


def _make_section(n_variants: int, section_offset: int = 0) -> Section:
    return Section(
        function_name_ptr=0,
        section_offset=section_offset,
        call_targets=[],
        variants=[_make_variant(ref_offset=i + 1) for i in range(n_variants)],
    )


@dataclass
class _FakeSession:
    """Hand-rolled stand-in for :class:`BinarySession` that exposes only
    the two private load helpers :func:`resolve_section_pointers`
    touches.
    """

    matched_sections: Dict[int, Section] = field(default_factory=dict)
    matched_functions: Dict[int, MatchedFunction] = field(default_factory=dict)
    unmatched_sections: Dict[int, Section] = field(default_factory=dict)
    unmatched_variant_function_data: Dict[int, List[FunctionData]] = field(
        default_factory=dict
    )

    def add_matched(
        self,
        idx: int,
        section: Section,
        variant_function_data: List[FunctionData],
    ) -> None:
        self.matched_sections[idx] = section
        self.matched_functions[idx] = MatchedFunction(
            func_name=f"matched_{idx}", variants=variant_function_data
        )

    def add_unmatched(
        self,
        idx: int,
        section: Section,
        variant_function_data: List[FunctionData],
    ) -> None:
        # Unmatched sections store one record per variant; the fake
        # mirrors the matched-arm shape so the resolver can iterate
        # ``section.variants`` in parallel with the per-variant body
        # list.
        self.unmatched_sections[idx] = section
        self.unmatched_variant_function_data[idx] = variant_function_data

    def _load_matched_section_and_variants(
        self, idx: int
    ) -> Tuple[Section, int, MatchedFunction]:
        section = self.matched_sections[idx]
        return section, section.section_offset, self.matched_functions[idx]

    def _load_unmatched_section_and_all_variants(
        self, idx: int
    ) -> Tuple[Section, int, List[FunctionData]]:
        section = self.unmatched_sections[idx]
        return (
            section,
            section.section_offset,
            self.unmatched_variant_function_data[idx],
        )


# ---------------------------------------------------------------------------
# auto_size_from_resolved -- pure reducer
# ---------------------------------------------------------------------------


def _resolved(
    arm: SectionKind,
    idx: int,
    section: Section,
    variant_function_data: List[FunctionData],
) -> ResolvedSection:
    return ResolvedSection(
        arm=arm,
        idx=idx,
        section=section,
        sampled_variant_indices=list(range(len(variant_function_data))),
        function_data_per_sampled_variant=variant_function_data,
    )


def test_auto_size_from_resolved_takes_max_variant_count_across_sections():
    """``num_variants_per_section`` is the per-section ``len(variants)``
    max -- not a sum, not the first entry."""
    section_a = _make_section(n_variants=2)
    section_b = _make_section(n_variants=5)
    section_c = _make_section(n_variants=3)
    fds_a = [_make_function_data(f"a{i}", n_body_tokens=4) for i in range(2)]
    fds_b = [_make_function_data(f"b{i}", n_body_tokens=4) for i in range(5)]
    fds_c = [_make_function_data(f"c{i}", n_body_tokens=4) for i in range(3)]
    resolved = [
        _resolved(SectionKind.MATCHED, 0, section_a, fds_a),
        _resolved(SectionKind.MATCHED, 1, section_b, fds_b),
        _resolved(SectionKind.MATCHED, 2, section_c, fds_c),
    ]

    spec = auto_size_from_resolved(resolved)

    assert isinstance(spec, SizingSpec)
    assert spec.num_variants_per_section == 5


def test_auto_size_from_resolved_context_len_adds_headroom_to_longest():
    """``context_len`` is the longest ``len(tokens) + variant_tokens``
    across every variant of every section, plus :data:`CONTEXT_LEN_HEADROOM`."""
    section_a = _make_section(n_variants=2)
    section_b = _make_section(n_variants=1)
    fds_a = [
        _make_function_data("a0", n_body_tokens=7, n_variant_tokens=2),
        _make_function_data("a1", n_body_tokens=4, n_variant_tokens=1),
    ]
    fds_b = [
        # Longest full token stream lives here: 11 + 3 = 14.
        _make_function_data("b0", n_body_tokens=11, n_variant_tokens=3),
    ]
    resolved = [
        _resolved(SectionKind.MATCHED, 0, section_a, fds_a),
        _resolved(SectionKind.MATCHED, 1, section_b, fds_b),
    ]

    spec = auto_size_from_resolved(resolved)

    longest_full_stream = 11 + 3
    assert spec.context_len == longest_full_stream + CONTEXT_LEN_HEADROOM
    assert spec.context_len >= longest_full_stream + CONTEXT_LEN_HEADROOM


def test_auto_size_from_resolved_empty_list_returns_zero_count_plus_headroom():
    """Empty resolved list -- zero variant count + bare headroom
    context_len. Callers are expected to treat an empty request as a
    no-op upstream; the helper itself does not raise."""
    spec = auto_size_from_resolved([])
    assert spec.num_variants_per_section == 0
    assert spec.context_len == CONTEXT_LEN_HEADROOM


def test_sizing_spec_is_frozen():
    """The handoff dataclass must be immutable -- consistent with the
    rest of the batch_decode dataclass backbone."""
    spec = SizingSpec(num_variants_per_section=2, context_len=64)
    with pytest.raises(Exception):
        spec.context_len = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_auto_sizes -- session-bound entry
# ---------------------------------------------------------------------------


def test_compute_auto_sizes_matched_dispatch_returns_section_max():
    """:func:`compute_auto_sizes` round-trips through
    :func:`resolve_section_pointers` for matched pointers and reduces
    the resolved list per the rules tested above."""
    section = _make_section(n_variants=4, section_offset=64)
    variant_fds = [
        _make_function_data(f"m_v{v}", n_body_tokens=5 + v) for v in range(4)
    ]
    session = _FakeSession()
    session.add_matched(0, section, variant_fds)

    spec = compute_auto_sizes(
        session,  # type: ignore[arg-type]
        [SectionPointerSpec(arm=SectionKind.MATCHED, idx=0)],
    )

    assert spec.num_variants_per_section == 4
    # Longest body is the last one (5 + 3 = 8) with no variant_tokens.
    assert spec.context_len == 8 + CONTEXT_LEN_HEADROOM


def test_compute_auto_sizes_unmatched_dispatch_single_variant():
    """An unmatched section with one variant surfaces its body's
    token-length unchanged through the auto-size reducer."""
    section = _make_section(n_variants=1, section_offset=32)
    fd = _make_function_data("u_only", n_body_tokens=9, n_variant_tokens=2)
    session = _FakeSession()
    session.add_unmatched(7, section, [fd])

    spec = compute_auto_sizes(
        session,  # type: ignore[arg-type]
        [SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=7)],
    )

    assert spec.num_variants_per_section == 1
    assert spec.context_len == (9 + 2) + CONTEXT_LEN_HEADROOM


def test_compute_auto_sizes_unmatched_dispatch_multi_variant_no_index_error():
    """Regression: an unmatched section with N>1 variants MUST resolve
    without IndexError. Pre-fix the resolver's per-record loader
    returned a 1-element FunctionData list while the section advertised
    N variants, so the cap-bypass sampling step indexed out of range
    on every multi-variant unmatched section (surfaced as
    ``list index out of range`` when the inspector navigated to an
    unmatched callee whose BIN section carried more than one variant).
    """
    section = _make_section(n_variants=3, section_offset=64)
    fds = [
        _make_function_data(f"u_v{v}", n_body_tokens=5 + v, n_variant_tokens=1)
        for v in range(3)
    ]
    session = _FakeSession()
    session.add_unmatched(0, section, fds)

    spec = compute_auto_sizes(
        session,  # type: ignore[arg-type]
        [SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=0)],
    )

    assert spec.num_variants_per_section == 3
    # Longest body is variant 2 (7 + 1 = 8).
    assert spec.context_len == 8 + CONTEXT_LEN_HEADROOM


def test_compute_auto_sizes_mixed_arms_max_across_request():
    """A request mixing matched + unmatched pointers picks the max
    variant count / token length across every section regardless of
    arm."""
    sec_m = _make_section(n_variants=3, section_offset=64)
    sec_u = _make_section(n_variants=1, section_offset=128)
    fds_m = [
        _make_function_data(f"m{v}", n_body_tokens=4, n_variant_tokens=1)
        for v in range(3)
    ]
    fd_u = _make_function_data("u", n_body_tokens=20, n_variant_tokens=4)
    session = _FakeSession()
    session.add_matched(0, sec_m, fds_m)
    session.add_unmatched(0, sec_u, [fd_u])

    spec = compute_auto_sizes(
        session,  # type: ignore[arg-type]
        [
            SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
            SectionPointerSpec(arm=SectionKind.UNMATCHED, idx=0),
        ],
    )

    assert spec.num_variants_per_section == 3
    assert spec.context_len == (20 + 4) + CONTEXT_LEN_HEADROOM


def test_compute_auto_sizes_enumerates_all_variants_not_just_sample():
    """The cap-bypass bound inside :func:`compute_auto_sizes` MUST make
    the resolver enumerate every variant -- otherwise the longest
    variant could be skipped by the sampler and ``context_len`` would
    be too small. This is the regression guard for the headroom-only
    failure mode."""
    section = _make_section(n_variants=20)
    # The last variant is the longest -- a random sample with seed 0
    # would almost certainly skip it.
    variant_fds = [
        _make_function_data(f"v{v}", n_body_tokens=(50 if v == 19 else 3))
        for v in range(20)
    ]
    session = _FakeSession()
    session.add_matched(0, section, variant_fds)

    spec = compute_auto_sizes(
        session,  # type: ignore[arg-type]
        [SectionPointerSpec(arm=SectionKind.MATCHED, idx=0)],
    )

    assert spec.num_variants_per_section == 20
    assert spec.context_len == 50 + CONTEXT_LEN_HEADROOM


# ---------------------------------------------------------------------------
# Regression vs the inspector's old hand-peek formula
# ---------------------------------------------------------------------------


def _old_inspector_peek(
    arm: SectionKind, session: _FakeSession, idx: int
) -> Tuple[int, int]:
    """Mirror of the inspector's pre-helper peek (now removed).

    Reproduces the exact ``(n_variants, context_len)`` it used to
    compute: longest variant's ``len(tokens)`` + 64 headroom, clamped
    at 64.
    """
    if arm is SectionKind.MATCHED:
        _section, _offset, matched = session._load_matched_section_and_variants(idx)
        variant_lengths = [len(v.tokens) for v in matched.variants]
        n_variants = len(matched.variants)
    else:
        _section, _offset, fd = session._load_unmatched_record_and_section(idx)
        variant_lengths = [len(fd.tokens)]
        n_variants = 1
    longest = max(variant_lengths) if variant_lengths else 0
    return n_variants, max(longest + CONTEXT_LEN_HEADROOM, CONTEXT_LEN_HEADROOM)


def test_compute_auto_sizes_matches_old_inspector_peek_when_variant_tokens_empty():
    """When every variant's ``variant_tokens`` is empty (the common
    case the old inspector peeked) :func:`compute_auto_sizes` MUST
    produce the same ``(n_variants, context_len)`` pair the inspector
    used to compute by hand. Sole tolerated drift: when the new helper
    sees non-empty ``variant_tokens`` (which the old peek ignored) it
    sizes UP -- that's the bug fix this helper exists to land, not a
    regression."""
    section = _make_section(n_variants=3, section_offset=64)
    variant_fds = [
        _make_function_data(f"v{v}", n_body_tokens=6 + v) for v in range(3)
    ]
    session = _FakeSession()
    session.add_matched(0, section, variant_fds)

    spec = compute_auto_sizes(
        session,  # type: ignore[arg-type]
        [SectionPointerSpec(arm=SectionKind.MATCHED, idx=0)],
    )
    old_n, old_ctx = _old_inspector_peek(SectionKind.MATCHED, session, 0)

    assert spec.num_variants_per_section == old_n
    assert spec.context_len == old_ctx


# ---------------------------------------------------------------------------
# End-to-end smoke -- ``compute_auto_sizes`` feeds ``batch_decode``
# ---------------------------------------------------------------------------


def test_compute_auto_sizes_drives_batch_decode_end_to_end(tmp_path) -> None:
    """The helper's output, threaded verbatim into :func:`batch_decode`,
    yields a result whose ``batch_size`` equals
    ``len(section_pointers) * sizing.num_variants_per_section``.

    Validates the design intent (loader-natural sizing is exactly what
    the pipeline wants) on the shared synthetic corpus.
    """
    fb = build_synthetic_binary(tmp_path)
    section_pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
    ]

    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as session:
        sizing = compute_auto_sizes(session, section_pointers)
        result = batch_decode(
            session,
            section_pointers=section_pointers,
            num_variants_per_section=sizing.num_variants_per_section,
            context_len=sizing.context_len,
            max_depth=0,
            rng=np.random.default_rng(seed=0),
        )

    expected_batch_size = (
        len(section_pointers) * sizing.num_variants_per_section
    )
    assert result.tokens.shape == (expected_batch_size, sizing.context_len)
    assert sizing.num_variants_per_section >= 1
    assert sizing.context_len > CONTEXT_LEN_HEADROOM
