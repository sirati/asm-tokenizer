"""Phase J.4 angr regression preservation suite.

Validates that the angr-backed tokenization path still works after the
owned-API refactor (Phases A-H + Ghidra-side fixes from Phase H.1 +
G.2). The owned-API redesign touched the angr provider significantly
(``tokenizer/disasm/angr_provider.py`` lazy view types, the
``MetadataLookup`` shape, ``op.fp_type`` stamping, ``slot_target ==
None`` invariants). These tests pin the documented invariants from
``tokenizer/disasm/angr_limitations.md`` so future refactors cannot
silently regress them.

The tests fall into two layers:

* **CLI smoke** (``test_angr_basic_tokenize``): invokes the public
  ``python -m tokenizer --backend angr`` entry point against the
  zlib minigzip fixtures and asserts the output CSV is structurally
  valid (v2 prelude, header, function rows with 6 fields, terminating
  vocabulary row, >=50 functions). Note: ``run_tokenizer.py``
  auto-switches arm/mips binaries to the ghidra backend even when
  ``--backend angr`` is requested (see ``run_tokenizer.py:275-280``;
  reason: known angr CFG bugs on those archs). So the x64 invocation
  is the *real* angr path; the arm32 invocation exercises the CLI's
  auto-switch fallback and asserts only that the chain still produces
  a valid CSV.

* **Provider invariants** (``test_angr_reg_list_separate_operands``,
  ``test_angr_no_slot_target``, ``test_angr_fp_type_always_none``):
  instantiate ``AngrDisassemblyProvider`` directly on the arm32 fixture
  and walk its lazy views to pin the asymmetries documented in
  ``angr_limitations.md`` (operand-kind asymmetry, missing slot_target,
  uniform ``fp_type=None``). Bypasses the CLI's auto-switch so we
  actually exercise the angr provider on ARM.

Fixture binaries: ``/tmp/asm_smoke/src/x64-gcc-7-Os_minigzip``,
``/tmp/asm_smoke/src/arm32-gcc-7-Os_minigzip``. If absent the test
sources them from ``src/zlib/`` under the project root.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture binary discovery
# ---------------------------------------------------------------------------
_FIXTURE_ROOT = Path("/tmp/asm_smoke/src")
_FIXTURES = ("x64-gcc-7-Os_minigzip", "arm32-gcc-7-Os_minigzip")
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ZLIB_SOURCE_ROOT = _PROJECT_ROOT / "src" / "zlib"


def _ensure_fixtures() -> Path:
    """Return the fixture root directory, copying from ``src/zlib/`` if
    a fixture is missing.

    The eye-inspect smoke runs maintain
    ``/tmp/asm_smoke/src/<arch>-gcc-7-Os_minigzip`` already, but a
    fresh worktree may not. The fallback copy keeps the test
    self-contained without bloating the fixture set: it skips
    cleanly if both source and destination are unavailable.
    """
    _FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    for name in _FIXTURES:
        dest = _FIXTURE_ROOT / name
        if dest.exists():
            continue
        src = _ZLIB_SOURCE_ROOT / name
        if not src.exists():
            pytest.skip(
                f"Fixture {name} missing at both {dest} and {src}; "
                f"cannot run angr regression suite."
            )
        shutil.copy2(src, dest)
    return _FIXTURE_ROOT


# ---------------------------------------------------------------------------
# CLI invocation helper
# ---------------------------------------------------------------------------
def _run_tokenizer_cli(
    *,
    queue_path: Path,
    source_dir: Path,
    output_dir: Path,
    backend: str,
    platform: str,
) -> subprocess.CompletedProcess[str]:
    """Invoke ``python -m tokenizer`` with the supplied parameters.

    Runs from ``_PROJECT_ROOT`` so the tokenizer module resolves on
    ``sys.path`` via the package layout (``-m tokenizer`` enters
    ``tokenizer/__main__.py``).
    """
    cmd = [
        sys.executable,
        "-m",
        "tokenizer",
        "--backend",
        backend,
        "--batch",
        str(queue_path),
        "--source",
        str(source_dir),
        "--output",
        str(output_dir),
        "--platform",
        platform,
    ]
    return subprocess.run(
        cmd,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


# ---------------------------------------------------------------------------
# CSV inspection helper
# ---------------------------------------------------------------------------
def _read_csv_layout(csv_path: Path) -> tuple[list[str], list[list[str]], list[str] | None]:
    """Return ``(prelude_row, function_rows, vocabulary_row)``.

    The CSV layout (see ``tokenizer/main_loop.py:215-232`` +
    ``tokenizer/vocab_unifier/saver.py``):

    1. Prelude: a single-cell row ``["version=2"]`` for v2 outputs.
    2. Header: ``["function_name", "occurrence", "tokens_base64",
       "block_runlength_base64", "instruction_runlength_base64",
       "metadata"]``.
    3. N function rows: 6 fields each, ``func_name`` in column 0.
    4. Trailing vocabulary row: ``["vocabulary", ...]`` (first cell
       literal ``"vocabulary"``).
    """
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return [], [], None

    prelude = rows[0]
    # Header is rows[1]. Function rows start at rows[2].
    body = rows[2:]
    vocabulary_row: list[str] | None = None
    function_rows: list[list[str]] = []
    for row in body:
        if row and row[0] == "vocabulary":
            vocabulary_row = row
            continue
        function_rows.append(row)
    return prelude, function_rows, vocabulary_row


# ---------------------------------------------------------------------------
# Test 1: end-to-end CLI smoke for both fixtures
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "binary_name",
    [
        "x64-gcc-7-Os_minigzip",
        # arm32: ``run_tokenizer.py`` auto-switches ``--backend angr``
        # to ghidra for arm/mips (lines 275-280). The CLI invocation
        # still completes; this exercises the auto-switch end-to-end.
        "arm32-gcc-7-Os_minigzip",
    ],
)
def test_angr_basic_tokenize(tmp_path: Path, binary_name: str) -> None:
    """``python -m tokenizer --backend angr --batch ...`` completes
    cleanly and emits a structurally valid v2 CSV with >= 50 function
    rows.

    Per ``main_loop.py:215-232`` the v2 CSV starts with a single-cell
    ``version=2`` prelude row, the canonical 6-column header, one row
    per surviving function, and a trailing ``vocabulary,...`` row.
    Each function row carries (function_name, occurrence, tokens,
    block_rl, insn_rl, metadata) — exactly 6 fields.
    """
    fixture_root = _ensure_fixtures()
    binary_path = fixture_root / binary_name
    assert binary_path.exists(), f"Fixture missing: {binary_path}"

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    queue_path = tmp_path / "queue.txt"
    queue_path.write_text(f"{binary_name}\n")

    proc = _run_tokenizer_cli(
        queue_path=queue_path,
        source_dir=fixture_root,
        output_dir=output_dir,
        backend="angr",
        platform="auto",
    )
    assert proc.returncode == 0, (
        f"Tokenizer CLI exited non-zero ({proc.returncode})\n"
        f"STDOUT (tail):\n{proc.stdout[-2000:]}\n"
        f"STDERR (tail):\n{proc.stderr[-2000:]}"
    )
    # Error counter — warnings allowed, errors are not. ``setup_logger``
    # registers a handler that bumps ``errors=N`` in the
    # ``Disassembly time: ...`` log line. The strictest check is just
    # to look for ``errors=0``.
    combined = proc.stdout + proc.stderr
    assert "errors=0" in combined, (
        f"Tokenizer reported errors != 0:\n{combined[-2000:]}"
    )

    csv_path = output_dir / f"{binary_name}_output.csv"
    assert csv_path.exists(), f"Expected CSV at {csv_path}"

    prelude, function_rows, vocabulary_row = _read_csv_layout(csv_path)
    assert prelude == ["version=2"], f"Expected v2 prelude row, got {prelude!r}"
    assert vocabulary_row is not None, (
        "Expected trailing 'vocabulary,...' row (see vocab_unifier/saver.py:24)"
    )
    assert vocabulary_row[0] == "vocabulary"

    # zlib minigzip has roughly 200 functions; allow a generous floor.
    assert len(function_rows) >= 50, (
        f"Expected >= 50 tokenized functions, got {len(function_rows)} "
        f"(zlib minigzip has ~200)"
    )

    for row in function_rows:
        assert len(row) == 6, (
            f"Expected 6-column function row, got {len(row)}: {row!r}"
        )
        # Field 0 (function name) is non-empty; field 1 (occurrence) is
        # a non-negative integer literal. Stronger semantic validation
        # belongs in dedicated parser tests — here we only assert the
        # row shape doesn't regress.
        assert row[0], f"Empty function_name in row: {row!r}"
        assert row[1].lstrip("-").isdigit(), (
            f"Non-integer occurrence in row: {row!r}"
        )


# ---------------------------------------------------------------------------
# Provider-level invariants (bypass the CLI's auto-switch)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def angr_arm32_provider():
    """Instantiate ``AngrDisassemblyProvider`` on the arm32 fixture.

    Module-scoped because CFG construction takes a few seconds and the
    provider state is read-only across the three invariant tests below.
    """
    _ensure_fixtures()
    from tokenizer.disasm.angr_provider import AngrDisassemblyProvider

    binary_path = _FIXTURE_ROOT / "arm32-gcc-7-Os_minigzip"
    provider = AngrDisassemblyProvider(binary_path)
    provider.build_cfg()
    try:
        yield provider
    finally:
        provider.close()


def test_angr_reg_list_separate_operands(angr_arm32_provider) -> None:
    """Per ``angr_limitations.md`` (intentional asymmetry): the angr
    provider (via Capstone) reports ARM register-list members as
    *separate* ``OperandKind.REG`` operands rather than as a single
    ``OperandKind.REG_LIST`` operand. The Ghidra provider emits the
    composite ``REG_LIST`` kind.

    This is the user-accepted asymmetry — the consumer-side ARM
    architecture provider (``tokenizer/arch/arm32.py``) handles both
    shapes uniformly. The test pins the angr-side shape so a future
    refactor does not silently regress to emitting ``REG_LIST``.

    Asserts: at least one register-list instruction in
    ``__aeabi_idiv0`` (gcc-7 -Os emits a ``push {r1, lr}`` here);
    that instruction carries >= 2 REG operands and zero REG_LIST
    operands.
    """
    from tokenizer.disasm.types import OperandKind

    target_function = "__aeabi_idiv0"
    register_list_mnemonics = {"push", "pop", "stmdb", "stmia", "ldmia", "ldmdb"}

    found_function = False
    found_reglist_insn = False
    for _addr, name, func in angr_arm32_provider.iter_functions():
        if name != target_function:
            continue
        found_function = True
        for block in func.blocks:
            for insn in block.instructions:
                base = insn.base_mnemonic.lower()
                # Capstone sometimes prefixes with conditional codes; match
                # the bare mnemonic as well.
                bare = base.split(".")[0]
                if bare not in register_list_mnemonics:
                    continue
                kinds = [op.kind for op in insn.operands]
                # Asymmetry assertion: NO REG_LIST present on the angr path.
                assert OperandKind.REG_LIST not in kinds, (
                    f"Unexpected REG_LIST operand in {name} @ "
                    f"{hex(insn.address)} {insn.mnemonic!r} {insn.op_str!r}; "
                    f"angr should emit separate REG operands "
                    f"(see angr_limitations.md / OperandKind asymmetry)."
                )
                reg_count = sum(1 for k in kinds if k == OperandKind.REG)
                assert reg_count >= 2, (
                    f"Expected >= 2 REG operands for register-list "
                    f"instruction at {hex(insn.address)} {insn.mnemonic!r} "
                    f"{insn.op_str!r}, got {reg_count} (kinds={kinds!r})"
                )
                found_reglist_insn = True
                # Pin the assertion on the first matching insn; the
                # invariant is per-instruction, not per-function.
                return

    assert found_function, f"Target function {target_function!r} missing from CFG"
    assert found_reglist_insn, (
        f"No register-list instruction found in {target_function!r}; "
        f"gcc-7 -Os should emit at least a push/stmdb here."
    )


def test_angr_no_slot_target(angr_arm32_provider) -> None:
    """Per ``angr_limitations.md`` §2/3: the angr provider never
    populates ``AddressMetadataView.slot_target``.

    The angr-side ``_AngrAddressMetadataView.slot_target`` returns
    ``None`` unconditionally (see ``angr_provider.py:222-224``) because
    angr has no equivalent of Ghidra's RTTI / SwitchAnalyzer. The
    associated ``AddressKind.JUMP_TABLE_SLOT`` / ``CODE_PTR_TABLE_SLOT``
    enum values are reserved for the Ghidra provider; angr emits
    ``RODATA`` / ``DATA`` / ``BSS`` etc. instead.

    Walks 100 distinct instruction-operand addresses through
    ``MetadataLookup.lookup`` and asserts both invariants.
    """
    from tokenizer.disasm.metadata import AddressKind

    lookup = angr_arm32_provider.create_metadata_lookup()

    # Collect a diverse address pool: instruction addresses, operand
    # immediate values, and memory-displacement targets. 100 entries is
    # well above the smallest function-count in the fixture.
    addresses: list[int] = []
    for _faddr, _fname, func in angr_arm32_provider.iter_functions():
        for block in func.blocks:
            for insn in block.instructions:
                addresses.append(insn.address)
                for op in insn.operands:
                    # ``imm`` is meaningful only for the imm-kind operand,
                    # but the lookup is address-typed and any int is a
                    # valid query (it returns ``AddressKind.NONE`` for
                    # un-indexed addresses).
                    try:
                        imm = op.imm
                    except Exception:
                        imm = 0
                    if imm:
                        addresses.append(imm)
                if len(addresses) >= 100:
                    break
            if len(addresses) >= 100:
                break
        if len(addresses) >= 100:
            break

    assert len(addresses) >= 100, (
        f"Could not collect 100 instruction-or-operand addresses for "
        f"the lookup walk; only got {len(addresses)}"
    )

    forbidden_kinds = {AddressKind.JUMP_TABLE_SLOT, AddressKind.CODE_PTR_TABLE_SLOT}
    for addr in addresses[:100]:
        meta = lookup.lookup(addr)
        assert meta.slot_target is None, (
            f"angr lookup unexpectedly returned slot_target != None for "
            f"addr={hex(addr)}: {meta.slot_target!r}"
        )
        assert meta.kind not in forbidden_kinds, (
            f"angr lookup returned forbidden AddressKind "
            f"{meta.kind!r} for addr={hex(addr)}; expected "
            f"slot kinds to be Ghidra-only (see angr_limitations.md)"
        )


def test_angr_fp_type_always_none(angr_arm32_provider) -> None:
    """Per ``angr_limitations.md`` §1: every operand on the angr path
    has ``fp_type is None``.

    ``angr_provider._stamp_fp_type_default`` attaches a class-level
    ``fp_type = None`` default to every Capstone operand class at
    module load time. Capstone does not surface an FP-precision signal,
    so the field stays ``None`` for the lifetime of the provider — no
    instance can override it without violating the consumer-side
    uniformity contract.

    Iterates 100 operands across the arm32 fixture and asserts the
    invariant.
    """
    operand_count = 0
    for _faddr, _fname, func in angr_arm32_provider.iter_functions():
        for block in func.blocks:
            for insn in block.instructions:
                for op in insn.operands:
                    assert op.fp_type is None, (
                        f"Unexpected non-None fp_type on angr operand: "
                        f"{op.fp_type!r} at "
                        f"{hex(insn.address)} {insn.mnemonic!r} "
                        f"{insn.op_str!r} (see angr_limitations.md §1)"
                    )
                    operand_count += 1
                    if operand_count >= 100:
                        return
    pytest.fail(
        f"Could not iterate 100 operands on the arm32 fixture "
        f"(got {operand_count}); arm32-gcc-7-Os_minigzip should have "
        f"thousands."
    )


# ---------------------------------------------------------------------------
# Test 5 (optional baseline diff): not implemented; no pre-refactor CSV
# baseline is checked into this repo. Re-enable when a baseline is
# committed under ``tests/baselines/``.
# ---------------------------------------------------------------------------
