"""Pass-1 walker integration with the FunctionNamesRegistry.

The walkers add every function name they emit (header function + called-
function references) to the shared registry so pass 2 can resolve the
section-CSV cells to 1-indexed sidecar line numbers.

These tests drive the walkers with synthetic :class:`ParsedRecord`
instances + a real :class:`ArmDedupState` writing to a tmp_path bin.
The bin is fresh per test (the writer's grow-on-demand mapping handles
the small fixtures with no truncation).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.parsed_record_iter import Matched, ParsedRecord
from tokenizer.memmap_builder._dedup import open_arm_dedup_state
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry
from tokenizer.memmap_builder.passes import (
    process_matched_function,
    process_unmatched_function,
)


@dataclass(frozen=True)
class _FakeVKey:
    """Minimal hashable vkey stand-in; identity is enough for the tests."""

    label: str


def _make_record(
    func_name: str,
    called: list[str],
    *,
    n_tokens: int = 4,
    token_fill: int = 0,
    extern_libraries: "dict[str, str] | None" = None,
) -> ParsedRecord:
    """One synthetic ParsedRecord that will round-trip through the encoder."""
    tokens = np.full(n_tokens, token_fill, dtype=np.uint16)
    block_runlength = np.array([1, 2], dtype=np.uint8)
    insn_runlength = np.array([3, 4], dtype=np.uint8)
    # Content hash is not used by the registry-wiring path; any
    # deterministic value works as long as identical bytes produce the
    # same hash (so the dedup gate fires when expected).
    content_hash = int(tokens.tobytes().__hash__() & 0xFFFFFFFFFFFFFFFF)
    typed_called = sorted(
        {(name, CallTargetType.LOCAL) for name in called},
        key=lambda nt: (nt[0], nt[1].value),
    )
    return ParsedRecord(
        func_name=func_name,
        insn_runlength=insn_runlength,
        block_runlength=block_runlength,
        tokens=tokens,
        called_funcs=typed_called,
        extern_libraries=extern_libraries or {},
        content_hash=content_hash,
    )


def _make_arm_state(tmp_path, name: str):
    state = open_arm_dedup_state(tmp_path / f"{name}_data.bin")
    return state


def test_matched_walker_records_header_and_callees_on_emit(tmp_path):
    """Function survives encoding (distinct bytes per variant) → its
    name + every referenced callee land in the registry."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    matched = Matched(
        func_name="header_fn",
        records={
            0: _make_record("header_fn", ["alpha_callee", "beta_callee"], token_fill=1),
            1: _make_record("header_fn", ["alpha_callee", "gamma_callee"], token_fill=2),
        },
    )

    registry = FunctionNamesRegistry()
    state = _make_arm_state(tmp_path, "matched")
    try:
        entry = process_matched_function(matched, [vkey_a, vkey_b], state, registry)
        assert entry is not None
    finally:
        state.writer.finalize()

    registry.finalize()
    expected = {"header_fn", "alpha_callee", "beta_callee", "gamma_callee"}
    assert set(registry._sorted) == expected  # noqa: SLF001 — test-side white-box


def test_matched_walker_skips_registry_when_function_dropped(tmp_path):
    """All variants encode to the same bytes → dedup collapses to one
    offset → the matched arm drops the function and no name is recorded."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    rec = _make_record("deduped_fn", ["only_callee"], token_fill=42)
    matched = Matched(func_name="deduped_fn", records={0: rec, 1: rec})

    registry = FunctionNamesRegistry()
    state = _make_arm_state(tmp_path, "matched_drop")
    try:
        entry = process_matched_function(matched, [vkey_a, vkey_b], state, registry)
    finally:
        state.writer.finalize()

    assert entry is None
    registry.finalize()
    assert registry._sorted == ()  # noqa: SLF001


def test_unmatched_walker_records_header_and_callees(tmp_path):
    """Single-variant function → header + its callees land in the registry."""
    vkey_a = _FakeVKey("a")
    rec = _make_record("unmatched_fn", ["x_callee", "y_callee"])
    records = {0: rec}

    registry = FunctionNamesRegistry()
    state = _make_arm_state(tmp_path, "unmatched")
    try:
        entries = process_unmatched_function(
            "unmatched_fn", records, [vkey_a], state, registry
        )
    finally:
        state.writer.finalize()

    assert len(entries) == 1
    registry.finalize()
    assert set(registry._sorted) == {"unmatched_fn", "x_callee", "y_callee"}  # noqa: SLF001


def test_unmatched_walker_handles_multi_variant_records(tmp_path):
    """Matched fallback path: every variant from the original Matched
    flows through here as the records dict; each adds its callees."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    records = {
        0: _make_record("partial_fn", ["alpha_callee"], token_fill=10),
        1: _make_record("partial_fn", ["beta_callee", "gamma_callee"], token_fill=20),
    }

    registry = FunctionNamesRegistry()
    state = _make_arm_state(tmp_path, "unmatched_multi")
    try:
        entries = process_unmatched_function(
            "partial_fn", records, [vkey_a, vkey_b], state, registry
        )
    finally:
        state.writer.finalize()

    assert len(entries) == 2
    registry.finalize()
    assert set(registry._sorted) == {  # noqa: SLF001
        "partial_fn",
        "alpha_callee",
        "beta_callee",
        "gamma_callee",
    }


def test_matched_entry_dict_shape_matches_phase3_contract(tmp_path):
    """Matched walker entry dict carries the typed-tuple keys Phase 3.2
    consumes: ``unique_called`` sorted ``list[(name, type)]``, per-variant
    ``called`` set of typed tuples, and an ``extern_libraries`` dict
    populated from the records' library info."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    rec_a = _make_record(
        "fn",
        ["loc_a"],
        token_fill=1,
        extern_libraries={"shared_extern": "libfoo.so"},
    )
    rec_b = _make_record(
        "fn",
        ["loc_b"],
        token_fill=2,
        extern_libraries={"shared_extern": "libfoo.so", "only_b": "libbar.so"},
    )
    matched = Matched(func_name="fn", records={0: rec_a, 1: rec_b})

    registry = FunctionNamesRegistry()
    state = _make_arm_state(tmp_path, "matched_shape")
    try:
        entry = process_matched_function(matched, [vkey_a, vkey_b], state, registry)
    finally:
        state.writer.finalize()

    assert entry is not None
    assert entry["func_name"] == "fn"
    assert entry["unique_called"] == [
        ("loc_a", CallTargetType.LOCAL),
        ("loc_b", CallTargetType.LOCAL),
    ]
    assert entry["extern_libraries"] == {
        "shared_extern": "libfoo.so",
        "only_b": "libbar.so",
    }
    by_vkey = {vd["vkey"]: vd for vd in entry["version_data"]}
    assert by_vkey[vkey_a]["called"] == {("loc_a", CallTargetType.LOCAL)}
    assert by_vkey[vkey_b]["called"] == {("loc_b", CallTargetType.LOCAL)}


def test_matched_extern_library_mismatch_first_wins_and_logs(tmp_path):
    """If two variants report different libraries for the same EXTERN
    name, the union picks the first variant's value and writes one warn
    line into ``error_log``."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    rec_a = _make_record(
        "fn",
        ["any"],
        token_fill=1,
        extern_libraries={"conflicting": "libfirst.so"},
    )
    rec_b = _make_record(
        "fn",
        ["any"],
        token_fill=2,
        extern_libraries={"conflicting": "libsecond.so"},
    )
    matched = Matched(func_name="fn", records={0: rec_a, 1: rec_b})

    registry = FunctionNamesRegistry()
    state = _make_arm_state(tmp_path, "matched_lib_conflict")
    error_log = io.StringIO()
    try:
        entry = process_matched_function(
            matched,
            [vkey_a, vkey_b],
            state,
            registry,
            error_log=error_log,
        )
    finally:
        state.writer.finalize()

    assert entry is not None
    assert entry["extern_libraries"] == {"conflicting": "libfirst.so"}
    log = error_log.getvalue()
    assert "WARN:" in log
    assert "extern library mismatch" in log
    assert "fn" in log


def test_unmatched_entry_dict_shape_matches_phase3_contract(tmp_path):
    """Per-variant unmatched entries carry typed ``called`` sets and
    a per-record ``extern_libraries`` dict (no union — single source)."""
    vkey_a = _FakeVKey("a")
    rec = _make_record(
        "fn",
        ["loc"],
        extern_libraries={"ext_known": "libfoo.so"},
    )
    entries = process_unmatched_function(
        "fn",
        {0: rec},
        [vkey_a],
        _make_arm_state(tmp_path, "unmatched_shape"),
        FunctionNamesRegistry(),
    )

    assert len(entries) == 1
    e = entries[0]
    assert e["func_name"] == "fn"
    assert e["vkey"] is vkey_a
    assert e["called"] == {("loc", CallTargetType.LOCAL)}
    assert e["extern_libraries"] == {"ext_known": "libfoo.so"}


def test_matched_walker_skips_dotL_prefix(tmp_path):
    """``.L``-prefixed local-label functions are never recorded."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    matched = Matched(
        func_name=".Llocal",
        records={
            0: _make_record(".Llocal", ["any_callee"], token_fill=1),
            1: _make_record(".Llocal", ["any_callee"], token_fill=2),
        },
    )

    registry = FunctionNamesRegistry()
    state = _make_arm_state(tmp_path, "matched_dotL")
    try:
        entry = process_matched_function(matched, [vkey_a, vkey_b], state, registry)
    finally:
        state.writer.finalize()

    assert entry is None
    registry.finalize()
    assert registry._sorted == ()  # noqa: SLF001
