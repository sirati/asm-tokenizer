"""Scaling guard: the unmatched pass-2 stays LINEAR under heavy-duplicate
input.

Owner constraint: the memmap builder must run in linear time in the input
size; the only permitted non-linearity is the content-dedup hashmap. A
canonical name that maps to M genuinely-distinct functions (each built in
S streams) is the z3-style pathological input. Before the distinct-section
fix the unmatched grouper keyed on the bare name, so all M*S bodies
collapsed into ONE section whose per-group pass-2 work was superlinear in
the group size -> the build wedged on z3. The fix keys the grouper on
``(func_name, occurrence)`` so the input is M distinct small sections,
restoring the small-group distribution the algorithm was always linear on.

This test scales M at 1x / 2x / 4x / 8x and fits a log-log slope of build
wall-time vs M. Linear is slope ~= 1; quadratic is ~= 2. We assert the
slope stays well under quadratic with margin. The companion standalone
benchmark (``bench_unmatched_linearity.py`` at the repo root) prints the
full timing table for manual inspection at larger M.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.index_format import write_index_prelude
from tokenizer.aligned_data.matched_sections_bin import SectionWriter
from tokenizer.aligned_data.parsed_record_iter import ParsedRecord
from tokenizer.memmap_builder._dedup import (
    finalize_arm_dedup_state,
    open_arm_dedup_state,
)
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry
from tokenizer.memmap_builder.passes import (
    build_function_lookup_table,
    process_unmatched_function,
    write_unmatched_sections_pass2,
)

from ._fixtures import StubVariants

_DUP = "__cxx_global_var_init"
_N_STREAMS = 6


def _record(func_name, occurrence, *, seed, called=()):
    t = seed % 60000
    return ParsedRecord(
        func_name=func_name,
        occurrence=occurrence,
        insn_runlength=np.array([seed % 250 + 4], dtype=np.uint16),
        block_runlength=np.array([seed % 250 + 3], dtype=np.uint16),
        tokens=np.array([t, t + 1, t + 2], dtype=np.uint16),
        called_funcs=list(called),
        extern_libraries={},
        content_hash=seed,  # globally unique -> distinct sections, no dedup link
    )


def _build_entries(tmp: Path, m_occurrences: int):
    """M distinct bodies of one duplicated name (each in S streams) plus M
    distinct caller functions each calling the duplicated name."""
    version_keys = [("v", s) for s in range(_N_STREAMS)]
    registry = FunctionNamesRegistry()
    state = open_arm_dedup_state(tmp / "u_data.bin")
    error_log = io.StringIO()
    entries = []
    seed = 1
    dup_call = [(_DUP, CallTargetType.LOCAL)]
    for occ in range(m_occurrences):
        for stream in range(_N_STREAMS):
            seed += 1
            entries.extend(
                process_unmatched_function(
                    _DUP, {stream: _record(_DUP, occ, seed=seed)},
                    version_keys, state, registry, error_log=error_log,
                )
            )
        seed += 1
        entries.extend(
            process_unmatched_function(
                f"caller_{occ}",
                {0: _record(f"caller_{occ}", 0, seed=seed, called=dup_call)},
                version_keys, state, registry, error_log=error_log,
            )
        )
    finalize_arm_dedup_state(state)
    registry.finalize()
    registry.write_sidecar(tmp, "bench")
    return entries, registry


def _time_pass2(tmp: Path, m_occurrences: int) -> float:
    entries, registry = _build_entries(tmp, m_occurrences)
    function_lookup = build_function_lookup_table([], entries)
    sectioned = {e["func_name"] for e in entries}
    section_writer = SectionWriter(tmp / "bench_sections.bin")
    t0 = time.perf_counter()
    try:
        with open(tmp / "u_sec.csv", "w", newline="", encoding="ascii") as sf, \
             open(tmp / "u_idx.bin", "wb") as idxf:
            write_csv_prelude(sf)
            write_index_prelude(idxf)
            write_unmatched_sections_pass2(
                entries, function_lookup, sf, idxf, io.StringIO(),
                StubVariants(), registry, section_writer,
                ExternProviderRegistry(), set(), sectioned,
                duplicated_names={_DUP},
            )
        section_writer.finalize()
    except BaseException:
        section_writer.close()
        raise
    return time.perf_counter() - t0


def test_unmatched_pass2_scales_linearly(tmp_path: Path) -> None:
    """Build wall-time grows ~linearly (NOT quadratically) with the number
    of distinct duplicated functions."""
    base = 300
    scales = [1, 2, 4, 8]
    ms: list[int] = []
    ts: list[float] = []
    for k in scales:
        m = base * k
        d = tmp_path / f"m{m}"
        d.mkdir()
        ts.append(_time_pass2(d, m))
        ms.append(m)

    slope = float(np.polyfit(np.log(np.array(ms, float)),
                             np.log(np.array(ts, float)), 1)[0])
    # Linear ~= 1.0, quadratic ~= 2.0. A name-merge regression (one giant
    # group) drives the slope toward 2; the distinct-section grouping keeps
    # it near 1. Generous ceiling absorbs timing noise yet catches a
    # quadratic regression decisively.
    assert slope < 1.5, (
        f"unmatched pass-2 scaling looks superlinear: log-log slope={slope:.3f} "
        f"(M={ms}, wall_s={[round(t, 4) for t in ts]}); a name-merge regression "
        f"in group_unmatched_entries_by_function would collapse duplicated "
        f"functions into one giant group again"
    )
