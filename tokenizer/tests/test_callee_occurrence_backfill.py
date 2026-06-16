"""Tests for the callee-occurrence backfill (producer-side disambiguator).

Concern under test: ``callee_occurrence_backfill.backfill_callee_occurrence``
stamps the callee ``occurrence`` ordinal onto ``local_funcs`` call-target
entries whose resolved entry ``addr`` belongs to a duplicated canonical
name, leaving everything else byte-for-byte untouched.

The fixtures synthesize v2 CSVs the way ``main_loop`` writes them (a
``version=2`` prelude, a header, function rows, and an interleaved
``vocabulary`` row), so no disassembler is needed.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from tokenizer.callee_occurrence_backfill import backfill_callee_occurrence


# ---------------------------------------------------------------------------
# Fixtures: faithful v2 CSV synthesis
# ---------------------------------------------------------------------------
def _metadata_cell(local_funcs: list[dict]) -> str:
    """One metadata JSON cell with the exact kwargs main_loop uses."""
    return json.dumps({"local_funcs": local_funcs}, separators=(",", ":"))


def _write_csv(path, rows: list[list[str]], *, prelude: bool = True) -> None:
    """Write a v2-shaped CSV (prelude + header + the given function/vocab rows)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        if prelude:
            writer.writerow(["version=2"])
        writer.writerow(
            [
                "function_name",
                "occurrence",
                "tokens_base64",
                "block_runlength_base64",
                "instruction_runlength_base64",
                "metadata",
            ]
        )
        for row in rows:
            writer.writerow(row)


def _function_row(name: str, occurrence: int, local_funcs: list[dict]) -> list[str]:
    return [name, str(occurrence), "AAAA", "AA", "AA", _metadata_cell(local_funcs)]


def _read_local_funcs(path, function_name: str) -> list[dict]:
    """Pull the parsed ``local_funcs`` list for a given function row."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0] == function_name:
                return json.loads(row[-1])["local_funcs"]
    raise AssertionError(f"row {function_name!r} not found")


# ---------------------------------------------------------------------------
# Guardrail 1 + 3: empty map / no-duplicate CSV is byte-identical
# ---------------------------------------------------------------------------
def test_empty_map_leaves_file_untouched(tmp_path):
    csv_path = tmp_path / "out.csv"
    _write_csv(
        csv_path,
        [
            _function_row("caller", 0, [{"name": "foo", "addr": "0x1000"}]),
            ["vocabulary", "", "x", "y", "z", "{}"],
            _function_row("foo", 0, []),
        ],
    )
    original = csv_path.read_bytes()
    backfill_callee_occurrence(csv_path, {})
    assert csv_path.read_bytes() == original


def test_no_matching_addr_is_byte_identical(tmp_path):
    # Non-empty map, but no local_funcs addr matches => every row passes
    # through verbatim; output must be byte-identical to the input.
    csv_path = tmp_path / "out.csv"
    _write_csv(
        csv_path,
        [
            _function_row("caller", 0, [{"name": "foo", "addr": "0x1000"}]),
            _function_row("foo", 0, []),
        ],
    )
    original = csv_path.read_bytes()
    backfill_callee_occurrence(csv_path, {0xDEAD: 1})
    assert csv_path.read_bytes() == original


# ---------------------------------------------------------------------------
# Core: occurrence injection on a duplicated-name callee
# ---------------------------------------------------------------------------
def test_injects_occurrence_on_duplicated_callee(tmp_path):
    csv_path = tmp_path / "out.csv"
    # ``dup`` exists at occurrence 0 (addr 0x1000) and occurrence 1
    # (addr 0x2000). ``caller`` calls the SECOND body (0x2000).
    _write_csv(
        csv_path,
        [
            _function_row(
                "caller",
                0,
                [
                    {"name": "dup", "addr": "0x2000"},
                    {"name": "single", "addr": "0x3000"},
                ],
            ),
            _function_row("dup", 0, []),
            _function_row("dup", 1, []),
            _function_row("single", 0, []),
        ],
    )
    # Map carries ONLY duplicated-name addrs (0x1000->0, 0x2000->1); the
    # non-duplicated ``single`` (0x3000) is absent by construction.
    backfill_callee_occurrence(csv_path, {0x1000: 0, 0x2000: 1})

    entries = _read_local_funcs(csv_path, "caller")
    by_name = {e["name"]: e for e in entries}
    assert by_name["dup"]["occurrence"] == 1
    # Non-duplicated callee must NOT gain an occurrence key.
    assert "occurrence" not in by_name["single"]


def test_injection_changes_only_targeted_rows(tmp_path):
    # A function row whose local_funcs contains no matching addr must be
    # left verbatim even while another row IS rewritten.
    csv_path = tmp_path / "out.csv"
    _write_csv(
        csv_path,
        [
            _function_row("caller_a", 0, [{"name": "dup", "addr": "0x2000"}]),
            _function_row("caller_b", 0, [{"name": "other", "addr": "0x9000"}]),
            _function_row("dup", 0, []),
            _function_row("dup", 1, []),
        ],
    )
    # Capture caller_b's exact serialized line before the rewrite.
    raw_before = csv_path.read_text(encoding="utf-8")
    line_b_before = next(
        line for line in raw_before.splitlines(keepends=True) if line.startswith("caller_b,")
    )

    backfill_callee_occurrence(csv_path, {0x1000: 0, 0x2000: 1})

    raw_after = csv_path.read_text(encoding="utf-8")
    line_b_after = next(
        line for line in raw_after.splitlines(keepends=True) if line.startswith("caller_b,")
    )
    assert line_b_after == line_b_before  # untouched, byte-identical
    assert _read_local_funcs(csv_path, "caller_a")[0]["occurrence"] == 1


# ---------------------------------------------------------------------------
# Guardrail 5: parity — injected occurrence == the targeted body's CSV column
# ---------------------------------------------------------------------------
def _csv_occurrence_column(path, function_name: str, want_occurrence: int) -> int:
    """Return the ``occurrence`` column of the named body whose column == want."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0] == function_name and row[1] == str(want_occurrence):
                return int(row[1])
    raise AssertionError(f"{function_name} occurrence {want_occurrence} not found")


def test_injected_occurrence_matches_definition_column(tmp_path):
    # The producer records (func_addr -> occurrence) at the SAME point it
    # writes CSV column 1. This test pins the contract end to end: a call
    # into the SECOND body of a duplicated name must be stamped with the
    # occurrence that the second body's definition row actually carries in
    # its column 1.
    csv_path = tmp_path / "out.csv"
    addr_occ0, addr_occ1 = 0x1000, 0x2000
    _write_csv(
        csv_path,
        [
            _function_row("caller", 0, [{"name": "dup", "addr": hex(addr_occ1)}]),
            _function_row("dup", 0, []),  # body at addr_occ0, column == 0
            _function_row("dup", 1, []),  # body at addr_occ1, column == 1
        ],
    )
    # Map mirrors what main_loop records: addr -> its CSV column value.
    backfill_callee_occurrence(csv_path, {addr_occ0: 0, addr_occ1: 1})

    injected = _read_local_funcs(csv_path, "caller")[0]["occurrence"]
    definition_column = _csv_occurrence_column(csv_path, "dup", 1)
    assert injected == definition_column == 1


# ---------------------------------------------------------------------------
# Guardrail 4: atomicity / fail-loud
# ---------------------------------------------------------------------------
def test_failure_leaves_original_intact_and_raises(tmp_path, monkeypatch):
    csv_path = tmp_path / "out.csv"
    _write_csv(
        csv_path,
        [
            _function_row("caller", 0, [{"name": "dup", "addr": "0x2000"}]),
            _function_row("dup", 0, []),
            _function_row("dup", 1, []),
        ],
    )
    original = csv_path.read_bytes()

    import tokenizer.callee_occurrence_backfill as mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("inject failure")

    monkeypatch.setattr(mod, "_inject_into_cell", _boom)

    with pytest.raises(RuntimeError, match="inject failure"):
        backfill_callee_occurrence(csv_path, {0x2000: 1})

    # Original CSV untouched; no leftover temp file.
    assert csv_path.read_bytes() == original
    leftovers = [p for p in tmp_path.iterdir() if p.name != "out.csv"]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Vocab / prelude / header passthrough
# ---------------------------------------------------------------------------
def test_vocab_and_prelude_rows_pass_through(tmp_path):
    csv_path = tmp_path / "out.csv"
    _write_csv(
        csv_path,
        [
            _function_row("caller", 0, [{"name": "dup", "addr": "0x2000"}]),
            ["vocabulary", "", "v0", "v1", "v2", "{}"],
            _function_row("dup", 0, []),
            _function_row("dup", 1, []),
        ],
    )
    backfill_callee_occurrence(csv_path, {0x1000: 0, 0x2000: 1})

    text = csv_path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "version=2"
    assert any(line.startswith("vocabulary,") for line in text.splitlines())
    assert _read_local_funcs(csv_path, "caller")[0]["occurrence"] == 1
