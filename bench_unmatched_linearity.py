"""Scaling benchmark: unmatched pass-2 wall-time vs heavy-duplicate input.

Reproduces the z3-style pathological input (one canonical name mapping to
M genuinely-distinct functions, each built in S streams = S arch-variants)
and times the full pass-1 + pass-2 unmatched build at 1x / 2x / 4x / 8x M.

Before the distinct-section fix the unmatched grouper keyed on the bare
name, so all M*S bodies collapsed into ONE section whose per-group pass-2
work was superlinear in the group size -> wall-time quadratic in M. After
the fix the grouper keys on ``(func_name, occurrence)``, so the input is M
distinct sections each with S arch-variants -> the small-group
distribution the algorithm was always linear on. The expected signature:
doubling M roughly doubles wall-time (slope ~1 on a log-log fit), NOT ~2.

Run inside the nix dev-shell:
    python bench_unmatched_linearity.py
Optionally pass a base scale: ``python bench_unmatched_linearity.py 200``.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

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
from tokenizer.memmap_builder.tests._fixtures import StubVariants

_DUP = "__cxx_global_var_init"
_N_STREAMS = 6  # arch-variants per distinct body (z3-like)


def _record(
    func_name: str,
    occurrence: int,
    *,
    seed: int,
    called=(),
) -> ParsedRecord:
    """Distinct-body ParsedRecord; body bytes vary by seed so content-dedup
    does not collapse across distinct functions (it still links genuinely
    byte-identical bodies, which is the allowed non-linearity)."""
    t = seed % 60000
    return ParsedRecord(
        func_name=func_name,
        occurrence=occurrence,
        insn_runlength=np.array([seed % 250 + 4], dtype=np.uint16),
        block_runlength=np.array([seed % 250 + 3], dtype=np.uint16),
        tokens=np.array([t, t + 1, t + 2], dtype=np.uint16),
        called_funcs=list(called),
        extern_libraries={},
        # content_hash stays globally unique so distinct bodies are NOT
        # content-deduped (the benchmark stresses distinct sections, not
        # the dedup link); the uint16 token wrap only affects body bytes.
        content_hash=seed,
    )


def _build_unmatched_entries(m_occurrences: int):
    """M distinct bodies of one duplicated name (each in S streams), PLUS
    M distinct unmatched CALLER functions that each call the duplicated
    name.

    Mirrors the z3 wedge: a single canonical name maps to many distinct
    functions, and many call sites reference it. On the OLD name-only
    grouping the dup collapses into ONE giant callee section, so every
    caller's per-call resolution re-parses that whole section
    (O(callers x group) -> quadratic). On the fixed grouping the dup is M
    distinct small sections and a call into a duplicated callee is stamped
    MISSING_VARIANT_INDEX with NO re-parse -> linear.

    Drives the REAL pass-1 walker so the entry dicts (incl. the
    ``occurrence`` ordinal) are produced the production way.
    """
    version_keys = [("v", s) for s in range(_N_STREAMS)]
    registry = FunctionNamesRegistry()
    tmp = TemporaryDirectory()
    state = open_arm_dedup_state(Path(tmp.name) / "u_data.bin")
    error_log = io.StringIO()
    entries = []
    seed = 1
    dup_call = [(_DUP, CallTargetType.LOCAL)]
    for occ in range(m_occurrences):
        for stream in range(_N_STREAMS):
            seed += 1
            rec = _record(_DUP, occ, seed=seed)
            entries.extend(
                process_unmatched_function(
                    _DUP, {stream: rec}, version_keys, state, registry,
                    error_log=error_log,
                )
            )
        # One distinct caller per occurrence, each calling the dup. These
        # are genuinely-single-variant unmatched functions (occurrence 0),
        # so they are NOT in duplicated_names; their edge into the
        # duplicated callee is the stressed back-patch path.
        seed += 1
        caller = _record(f"caller_{occ}", 0, seed=seed, called=dup_call)
        entries.extend(
            process_unmatched_function(
                f"caller_{occ}", {0: caller}, version_keys, state, registry,
                error_log=error_log,
            )
        )
    finalize_arm_dedup_state(state)
    registry.finalize()
    return entries, registry, tmp


def _time_pass2(m_occurrences: int) -> float:
    entries, registry, tmp = _build_unmatched_entries(m_occurrences)
    try:
        registry.write_sidecar(Path(tmp.name), "bench")
        function_lookup = build_function_lookup_table([], entries)
        sectioned = {e["func_name"] for e in entries}
        variants = StubVariants()
        extern_providers = ExternProviderRegistry()
        bin_path = Path(tmp.name) / "bench_sections.bin"
        section_writer = SectionWriter(bin_path)
        sec_csv = Path(tmp.name) / "u_sec.csv"
        idx_bin = Path(tmp.name) / "u_idx.bin"
        t0 = time.perf_counter()
        try:
            with open(sec_csv, "w", newline="", encoding="ascii") as sf, \
                 open(idx_bin, "wb") as idxf:
                write_csv_prelude(sf)
                write_index_prelude(idxf)
                write_unmatched_sections_pass2(
                    entries, function_lookup, sf, idxf, io.StringIO(),
                    variants, registry, section_writer, extern_providers,
                    set(), sectioned,
                    duplicated_names={_DUP},
                )
            section_writer.finalize()
        except BaseException:
            section_writer.close()
            raise
        return time.perf_counter() - t0
    finally:
        tmp.cleanup()


def main() -> None:
    base = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    scales = [1, 2, 4, 8]
    print(f"base M={base}  streams/body={_N_STREAMS}  (M*S bodies total)")
    print(f"{'scale':>6} {'M':>8} {'bodies':>9} {'wall_s':>10} {'ratio':>8}")
    prev = None
    times = []
    for k in scales:
        m = base * k
        # warm + measure (single timed run; the work dwarfs noise at these M)
        dt = _time_pass2(m)
        times.append((m, dt))
        ratio = (dt / prev) if prev else float("nan")
        print(f"{k:>5}x {m:>8} {m * _N_STREAMS:>9} {dt:>10.4f} {ratio:>8.2f}")
        prev = dt
    # Log-log slope: linear ~1.0, quadratic ~2.0.
    ms = np.array([m for m, _ in times], dtype=float)
    ts = np.array([t for _, t in times], dtype=float)
    slope = np.polyfit(np.log(ms), np.log(ts), 1)[0]
    print(f"\nlog-log slope = {slope:.3f}  (≈1 linear, ≈2 quadratic)")


if __name__ == "__main__":
    main()
