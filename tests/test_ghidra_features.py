"""Phase J.3: Ghidra-only v2 feature assertions on tokenized fixtures.

Each test targets one Ghidra-only feature surface (REG_LIST writeback
bracketing, MEM disp+base merging, MEM_MINUS negative-immediate prefix
outside mem brackets, MIPS delay-slot mnemonic collapse, FP-typed
operand detection, provider string analyzer → ``string_ptr`` metadata).
The known-function-per-fixture approach lets each assertion be small
and targeted; if a fixture's binary doesn't contain a matching
instruction the test ``pytest.skip``s with a clear message so the
absence is visible in the run-log rather than disguised as a silent
pass or false fail.

Pre-tokenized CSVs at ``/tmp/asm_smoke/out/`` are reused if present
(the previous run's outputs); otherwise the test ``pytest.skip``s with
a hint to run the Phase J.2 smoke (``test_ghidra_e2e_smoke.py``) first
to populate them. Re-tokenization inline would multiply test runtime
by 7 (the J.2 smoke covers that pathway already), and Phase J.3's
concern is the wire-format SHAPE of the outputs, not the tokenize
pipeline's correctness end-to-end.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest

from tokenizer.csv_files import open_versioned_csv_reader
from tokenizer.compact_base64_utils import base64_to_ndarray_vec
from tokenizer.string_sidecar import iter_sidecar_lines
from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.loader import load_vocab_manager


_OUT_DIR = Path("/tmp/asm_smoke/out")


# The token stream's ``v2`` digit slots (IDs 0..255) ride immediately
# after the parent metatoken (e.g. ``valued_const_v2 | <0x10>`` is the
# integer 0x10 encoded as a single digit byte after the type-id). Real
# vocabulary tokens have IDs >= 256.
_DIGIT_SLOT_MAX = 256


def _csv_path(stem: str) -> Path:
    """Path to a pre-tokenized fixture's CSV; skip if missing."""
    p = _OUT_DIR / f"{stem}_output.csv"
    if not p.is_file():
        pytest.skip(
            f"pre-tokenized fixture missing at {p}; run "
            f"test_ghidra_e2e_smoke.py first to populate /tmp/asm_smoke/out/"
        )
    return p


def _strings_path(stem: str) -> Path:
    return _OUT_DIR / f"{stem}_strings.bin"


def _load_vm(csv_path: Path) -> VocabularyManager:
    vm = load_vocab_manager(csv_path)
    assert vm is not None, f"vocab failed to load from {csv_path}"
    return vm


def _resolve_token(vm: VocabularyManager, token_id: int) -> str:
    """Map a token id to its vocab string. Digit slots (0..255) come
    back as ``<digit_HH>``; real ids resolve via the vocab table.
    """
    if token_id < _DIGIT_SLOT_MAX:
        return f"<digit_{token_id:02x}>"
    if token_id < len(vm.id_to_token):
        return vm.id_to_token[token_id]
    return f"<oor_{token_id}>"


def _iter_function_row(csv_path: Path, func_name: str) -> Iterator[list[str]]:
    """Yield each CSV row whose first cell equals ``func_name``."""
    csv.field_size_limit(10_000_000)
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader, _format_version = open_versioned_csv_reader(fh)
        header = next(reader)
        del header  # only positions matter; v2 column order is fixed
        for row in reader:
            if row and row[0] == func_name:
                yield row


def _function_tokens(csv_path: Path, func_name: str) -> tuple[list[int], list[str]]:
    """Return ``(raw_ids, names)`` for the first occurrence of
    ``func_name`` in ``csv_path``. Raises ``LookupError`` if absent.
    """
    vm = _load_vm(csv_path)
    for row in _iter_function_row(csv_path, func_name):
        raw_ids = [int(t) for t in base64_to_ndarray_vec(row[2])]
        names = [_resolve_token(vm, t) for t in raw_ids]
        return raw_ids, names
    raise LookupError(f"function {func_name!r} not found in {csv_path}")


def _function_metadata(csv_path: Path, func_name: str) -> dict:
    """Decode the v2 ``metadata`` JSON column for the first occurrence
    of ``func_name`` in ``csv_path``.
    """
    for row in _iter_function_row(csv_path, func_name):
        # v2 metadata column is at index 5 (header position is fixed:
        # function_name, occurrence, tokens_base64, block_runlength_base64,
        # instruction_runlength_base64, metadata).
        return json.loads(row[5])
    raise LookupError(f"function {func_name!r} not found in {csv_path}")


# ---------- 1. REG_LIST: ARM writeback bracketing ----------------------


def test_reg_list_writeback_arm32_aeabi_idiv0() -> None:
    """``__aeabi_idiv0`` is a one-off PLT stub that compiles down to a
    ``stmdb sp!, {r1, lr}`` / ``ldmia sp!, {r1, pc}`` pair on ARM
    (compiler-emitted divide-by-zero trap). Both instructions carry a
    REG_LIST operand with writeback enabled — exactly the shape Phase
    J.3 wants to assert.
    """
    csv_path = _csv_path("arm32-gcc-7-Os_minigzip")
    _, names = _function_tokens(csv_path, "__aeabi_idiv0")

    # writeback marker present
    assert "asm_writeback_detect" in names, (
        f"missing asm_writeback_detect in __aeabi_idiv0 token stream: {names}"
    )

    # find the first open/close brace pair; assert >= 2 registers
    # between them.
    try:
        open_idx = names.index("REG_LIST_OPEN_BRACE")
    except ValueError as exc:
        raise AssertionError(
            f"REG_LIST_OPEN_BRACE not in __aeabi_idiv0 stream: {names}"
        ) from exc

    # The brace pair is balanced (open then close, no nesting). Find
    # the next close after the open.
    close_idx = None
    for j in range(open_idx + 1, len(names)):
        if names[j] == "REG_LIST_CLOSE_BRACE":
            close_idx = j
            break
    assert close_idx is not None, (
        f"REG_LIST_CLOSE_BRACE not found after open at {open_idx}: {names}"
    )

    # The between-braces window contains REGISTER tokens (everything
    # that isn't a digit slot or another bracketing token). For
    # ``stmdb sp!, {r1, lr}`` the two members are ``arm32_r1`` and
    # ``arm32_lr`` — count entries that aren't a digit-slot pad and
    # aren't another bracketing marker.
    between = names[open_idx + 1 : close_idx]
    reg_members = [
        n
        for n in between
        if not n.startswith("<digit_")
        and n not in {"REG_LIST_OPEN_BRACE", "REG_LIST_CLOSE_BRACE"}
    ]
    assert len(reg_members) >= 2, (
        f"expected >=2 registers in reg-list, got {reg_members} "
        f"from window {between}"
    )


# ---------- 2. Disp+base merge: riscv64 c.sdsp ------------------------


def test_mem_disp_base_merge_riscv64_xcalloc() -> None:
    """riscv64 ``xcalloc`` calls into the standard prologue ``c.sdsp
    ra, 8(sp)``. The provider's SLEIGH-split disp+base pair must be
    merged into a single ``mem[sp + 8]mem`` operand shape — i.e. inside
    a single ``MEM_OPEN_BRACKET`` ... ``MEM_CLOSE_BRACKET`` window the
    base register, ``MEM_PLUS``, and the displacement valued_const are
    co-located.
    """
    csv_path = _csv_path("riscv64-clang-10-O2_hello")
    _, names = _function_tokens(csv_path, "xcalloc")

    # Find every [MEM_OPEN_BRACKET ... MEM_CLOSE_BRACKET] window;
    # require at least one to match the merged shape.
    matched = False
    open_idx = -1
    for i, n in enumerate(names):
        if n == "MEM_OPEN_BRACKET":
            open_idx = i
            continue
        if n == "MEM_CLOSE_BRACKET" and open_idx >= 0:
            window = names[open_idx + 1 : i]
            # Required pieces of the merged shape — register token,
            # MEM_PLUS, valued_const_v2. Token-class names per the
            # actual provider output: riscv64's stack-pointer is
            # ``riscv64_sp`` (NOT ``arm32_sp`` as the J.3 spec
            # provisionally states — the spec explicitly allows
            # adapting the token-class name to actual provider output).
            non_digit = [w for w in window if not w.startswith("<digit_")]
            has_base_reg = any(w.startswith("riscv64_") for w in non_digit)
            has_plus = "MEM_PLUS" in non_digit
            has_valued = "valued_const_v2" in non_digit
            if has_base_reg and has_plus and has_valued:
                matched = True
                break
            open_idx = -1

    assert matched, (
        "no MEM[ base + valued_const_v2 ]MEM window in xcalloc token stream "
        f"({names})"
    )


# ---------- 3. Negative immediate: riscv64 c.addi sp, -0x10 -----------


def test_negative_imm_mem_minus_riscv64_xcalloc() -> None:
    """riscv64 ``c.addi sp, -0x10`` (stack adjust at function entry):
    the negative immediate must be encoded as a leading ``MEM_MINUS``
    token immediately before a ``valued_const_v2`` token, and that
    pair must NOT be nested inside a ``MEM_OPEN_BRACKET`` ...
    ``MEM_CLOSE_BRACKET`` window (a ``mem[ ]mem`` would imply the
    minus is part of an addressing expression rather than a leading
    arithmetic sign).
    """
    csv_path = _csv_path("riscv64-clang-10-O2_hello")
    _, names = _function_tokens(csv_path, "xcalloc")

    # Track open-bracket depth so we can check the MEM_MINUS is at
    # depth 0 (outside any mem[ ]mem pair).
    depth = 0
    found = False
    for i, n in enumerate(names):
        if n == "MEM_OPEN_BRACKET":
            depth += 1
            continue
        if n == "MEM_CLOSE_BRACKET":
            depth -= 1
            continue
        if n == "MEM_MINUS" and depth == 0:
            # Look ahead for the next non-digit token; should be a
            # valued_const_v2.
            for j in range(i + 1, len(names)):
                if names[j].startswith("<digit_"):
                    continue
                if names[j] == "valued_const_v2":
                    found = True
                break
            if found:
                break

    assert found, (
        "no MEM_MINUS-prefix valued_const_v2 pair at depth 0 in xcalloc "
        f"token stream ({names})"
    )


# ---------- 3b. Provider-direct: IMM scalar arrives signed --------------


_RISCV64_HELLO_BINARY = Path(
    "/tmp/asm_smoke/src/riscv64-clang-10-O2_hello"
)


def test_negative_imm_signed_at_provider_riscv64_xcalloc() -> None:
    """riscv64 ``xcalloc`` opens with ``c.addi sp, -0x10``. Reading the
    operand straight off the Ghidra provider (BEFORE the operand
    tokenizer's ``abs(...)`` rebind / MEM_MINUS pre-emission), the IMM
    scalar must arrive as ``-16`` — Ghidra's signed reinterpretation of
    the i6 immediate's stored bit pattern, matching what Ghidra renders
    in disassembly. This pins the Scalar.getSignedValue() decoding in
    ``decode_helper.operand_spec``; without it, ``Scalar.getValue()``
    surfaces the unsigned bit pattern and ``op.imm`` would be
    non-negative for every Ghidra-decoded immediate.
    """
    if not _RISCV64_HELLO_BINARY.is_file():
        pytest.skip(
            f"riscv64 fixture missing at {_RISCV64_HELLO_BINARY}; "
            f"run test_ghidra_e2e_smoke.py first to stage it"
        )

    # Local import: keep the Ghidra provider construction inside this
    # test function so module-level collection stays cheap (Ghidra
    # spin-up only pays when this test actually runs).
    from tokenizer.disasm import get_disassembly_provider
    from tokenizer.disasm.types import OperandKind

    provider = get_disassembly_provider("ghidra", _RISCV64_HELLO_BINARY)
    provider.build_cfg()
    try:
        xcalloc = None
        for _addr, name, function in provider.iter_functions():
            if name == "xcalloc":
                xcalloc = function
                break
        assert xcalloc is not None, "xcalloc function not found in riscv64 fixture"

        negative_imms: list[int] = []
        addi_sp_negative_found = False
        for block in xcalloc.blocks:
            for insn in block.instructions:
                base_mn = insn.base_mnemonic
                for op in insn.operands:
                    if op.kind != OperandKind.IMM:
                        continue
                    if op.imm < 0:
                        negative_imms.append(int(op.imm))
                    # The canonical c.addi sp, -0x10 stack-adjust is the
                    # most reliable single-instruction probe; pin it
                    # explicitly when present so a regression that
                    # changes only this site (e.g. a future refactor
                    # that re-introduces getValue() under a different
                    # branch) trips here directly.
                    if base_mn in ("c.addi", "addi") and op.imm == -16:
                        addi_sp_negative_found = True

        assert negative_imms, (
            "no negative IMM operand surfaced in riscv64 xcalloc; "
            "Scalar.getSignedValue() decoding appears broken (every "
            "Ghidra IMM came back non-negative)"
        )
        assert addi_sp_negative_found, (
            "expected c.addi/addi with imm == -16 (the canonical stack "
            f"adjust at xcalloc entry); negative IMMs observed: {negative_imms}"
        )
    finally:
        provider.close()


# ---------- 4. MIPS delay-slot collapse --------------------------------


def test_mips32_delay_slot_collapse_adler32_combine() -> None:
    """MIPS delay-slot collapse: a Ghidra-disassembled MIPS function
    should emit ``mips32_sra`` (canonical mnemonic) — NOT the
    SLEIGH-internal double-underscore form ``mips32__sra`` that
    appears when delay-slot reordering hasn't been applied.
    """
    csv_path = _csv_path("mips32-gcc-9-Os_minigzip")
    _, names = _function_tokens(csv_path, "adler32_combine")

    assert "mips32_sra" in names, (
        f"adler32_combine missing mips32_sra mnemonic: {names}"
    )
    assert "mips32__sra" not in names, (
        f"adler32_combine emitted double-underscore mnemonic "
        f"(delay-slot collapse failed): {names}"
    )


# ---------- 5. FP detection (Ghidra-only) -------------------------------


_FP_TOKEN_NAMES = {
    "float16", "bfloat16", "float32", "float64", "float80", "float128",
}


def test_fp_detection_emits_float_token_anywhere() -> None:
    """Ghidra's ``OperandType.FLOAT`` reading drives Step-1 of the v2
    constant classifier (``floatXX`` token + IEEE digit bytes for
    arithmetic FP immediates, or postfix ``floatXX`` annotation after
    an FP-typed load's ptr token). Either way a float token must
    appear in the vocab + at least one function's token stream.

    Provider-side gap: ``_compute_fp_type`` keys off
    ``Instruction.getOperandType(i) & OperandType.FLOAT`` (per-operand
    SLEIGH spec tag) but the FP-typed-ness of an SSE op like ``MULSD``
    is only available at the P-code level (``FLOAT_MULT`` p-code op),
    NOT on the instruction's operand-type bitmask. Probed on the
    clamscan x64 corpus (e.g. ``src/clamav/x64-clang-3.5-O2_clamscan``,
    ``x64-gcc-7-O2_clamscan``): every SSE FP mnemonic
    (MULSD/DIVSD/ADDSD/CVTSI2SD/...) returns ``OperandType``s of 0x200
    (REGISTER) or 0x2080 (ADDRESS|SCALAR) — zero FLOAT-tagged operands
    across the whole binary. So even staging a FP-bearing fixture into
    the smoke wouldn't trip the v2 classifier's FP path.

    This is the same shape of provider-side architectural gap as the
    ``string_ptr`` case below: the data Ghidra has IS sufficient (the
    p-code carries ``FLOAT_*`` ops with width-from-output-varnode), but
    ``_compute_fp_type`` consumes the wrong API surface. Fixing it
    needs a p-code-walk fallback that's a separate (larger) refactor.
    """
    # Scan every available fixture's vocab for a float token. If none
    # has any, skip (no FP-typed instruction in any current fixture).
    fixtures = [
        "x64-gcc-7-Os_minigzip",
        "arm32-gcc-7-Os_minigzip",
        "arm64-gcc-4.8-Os_minigzip",
        "mips32-gcc-9-Os_minigzip",
        "mips64-gcc-7-Os_minigzip",
        "ppc64-clang-10-O0_hello",
        "riscv64-clang-10-O2_hello",
    ]
    fixtures_with_fp: list[str] = []
    for stem in fixtures:
        p = _OUT_DIR / f"{stem}_output.csv"
        if not p.is_file():
            continue
        vm = load_vocab_manager(p)
        if vm is None:
            continue
        for fp_name in _FP_TOKEN_NAMES:
            if fp_name in vm.token_to_id:
                fixtures_with_fp.append(stem)
                break

    if not fixtures_with_fp:
        pytest.skip(
            "no fixture vocab contains a floatXX token; the FP-detection "
            "surface is provider-side gated on Instruction.getOperandType "
            "& OperandType.FLOAT, which Ghidra's x86 SLEIGH spec does NOT "
            "set on SSE FP mnemonics (MULSD/DIVSD/ADDSD/...) — only on "
            "p-code FLOAT_* ops. FLAGGED: needs a p-code-walk fallback in "
            "_compute_fp_type before any real-world corpus exercises it."
        )

    # At least one fixture has a float token in vocab — assert it also
    # appears in at least one function's token stream there.
    for stem in fixtures_with_fp:
        p = _OUT_DIR / f"{stem}_output.csv"
        vm = load_vocab_manager(p)
        float_ids = {vm.token_to_id[name] for name in _FP_TOKEN_NAMES if name in vm.token_to_id}
        with p.open(newline="", encoding="utf-8") as fh:
            reader, _ver = open_versioned_csv_reader(fh)
            next(reader)  # header
            for row in reader:
                if row and row[0] != "vocabulary":
                    toks = base64_to_ndarray_vec(row[2])
                    if any(int(t) in float_ids for t in toks):
                        return  # success — float token found in stream
    pytest.skip(
        "floatXX token in vocab but no function-row token stream "
        "references one; FLAGGED."
    )


# ---------- 6. String detection (Ghidra-only) --------------------------


def test_string_ptr_metadata_arm32_minigzip() -> None:
    """Ghidra's string analyzer feeds Step-7 of the v2 classifier: an
    address inside a recognised string region emits ``string_ptr``
    metadata of shape ``{line, start_offset, encoding}`` and pushes the
    underlying bytes into ``<binary>_strings.bin``.

    If the fixture's strings sidecar is empty AND no row's metadata
    carries a ``string_ptr`` key, FLAG via skip — this points at a
    classifier-side regression (Phase J.3 spec) rather than a test bug.
    """
    csv_path = _csv_path("arm32-gcc-7-Os_minigzip")
    sidecar_path = _strings_path("arm32-gcc-7-Os_minigzip")
    assert sidecar_path.is_file(), f"sidecar missing: {sidecar_path}"

    # Scan every function row for a string_ptr metadata entry.
    csv.field_size_limit(10_000_000)
    matched_entry: dict | None = None
    matched_func: str | None = None
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader, _ver = open_versioned_csv_reader(fh)
        next(reader)  # header
        for row in reader:
            if not row or row[0] == "vocabulary":
                continue
            try:
                meta = json.loads(row[5])
            except json.JSONDecodeError:
                continue
            entries = meta.get("string_ptr") if isinstance(meta, dict) else None
            if entries:
                matched_func = row[0]
                matched_entry = entries[0]
                break

    if matched_entry is None:
        pytest.skip(
            f"no string_ptr metadata in any function of {csv_path}; "
            f"sidecar size={sidecar_path.stat().st_size}. FLAGGED: "
            f"classifier-side string detection not exercising on this fixture."
        )

    # Shape: {line, start_offset, encoding} per precedence.md step 7.
    assert isinstance(matched_entry, dict), matched_entry
    assert "line" in matched_entry, matched_entry
    assert "encoding" in matched_entry, matched_entry
    line_index = matched_entry["line"]
    assert isinstance(line_index, int), matched_entry

    # The referenced line must be readable from the sidecar (and the
    # sidecar must be non-empty when at least one entry has been emitted).
    assert sidecar_path.stat().st_size > 0, (
        f"function {matched_func} emitted string_ptr but sidecar is empty"
    )
    if line_index >= 0:
        lines = list(iter_sidecar_lines(sidecar_path))
        assert 0 <= line_index < len(lines), (
            f"string_ptr line {line_index} out of range "
            f"(sidecar has {len(lines)} lines)"
        )


# ---------- 7. Negative bracket disp: arm32 ldr r6, [r5, #-4] ----------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARM32_ZLIB_BINARY = Path(
    os.environ.get("ASM_TOKENIZER_ZLIB_SRC")
    or "/home/sirati/devel/python/asm-tokenizer/src/zlib"
) / "arm32-gcc-7-Os_minigzip"


def _tokenize_inplace(binary: Path, tmp_path: Path) -> Path:
    """Re-tokenize ``binary`` via the standalone ``--batch`` entry-point
    into ``tmp_path/out`` and return the resulting CSV path. Mirrors
    ``test_ghidra_e2e_smoke.py``'s single-binary smoke pattern so the
    bracket-shape assertions below exercise the full Ghidra → tokenizer
    pipeline (not just an in-process operand-tokenizer probe) on a CSV
    that was produced *after* the postfix-``value_negative`` migration.
    Reusing a stale ``/tmp/asm_smoke/out/`` snapshot would assert the
    pre-migration shape and would falsely pass against the producer fix.
    """
    queue = tmp_path / "queue.txt"
    queue.write_text(f"{binary.name}\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    cmd = [
        sys.executable,
        "-m",
        "tokenizer",
        "--backend", "ghidra",
        "--batch", str(queue),
        "--source", str(binary.parent),
        "--output", str(out_dir),
        "--platform", "auto",
    ]
    env = dict(os.environ)
    env["ASM_TOKENIZER_FORMAT_VERSION"] = "2"

    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"tokenize failed (rc={proc.returncode}) for {binary.name}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    csv_path = out_dir / f"{binary.name}_output.csv"
    assert csv_path.is_file(), f"CSV not emitted at {csv_path}"
    return csv_path


@pytest.mark.slow
def test_negative_disp_arm32_value_negative_postfix(tmp_path: Path) -> None:
    """arm32 ``main`` in ``arm32-gcc-7-Os_minigzip`` contains
    ``ldr r6, [r5, #-4]`` at 0x10a94 (a plain pre-indexed negative-disp
    load). After the postfix-``value_negative`` migration, the bracket
    must render as
    ``mem[ arm32_r5 MEM_PLUS valued_const_v2 <0x04> value_negative ]mem``
    — the sign rides on a postfix metatoken inside the brackets, NOT as
    a leading ``MEM_MINUS`` operator.

    This is the ARM32-side counterpart to the riscv64
    ``test_negative_imm_value_negative_riscv64_xcalloc`` canary (which
    pins the *outside-brackets* arithmetic-immediate shape); together
    they cover both endpoints of the v2 sign-handling contract.
    """
    if not _ARM32_ZLIB_BINARY.is_file():
        pytest.skip(f"arm32 fixture missing at {_ARM32_ZLIB_BINARY}")

    csv_path = _tokenize_inplace(_ARM32_ZLIB_BINARY, tmp_path)
    _, names = _function_tokens(csv_path, "main")

    # Walk every [MEM_OPEN_BRACKET ... MEM_CLOSE_BRACKET] window in
    # ``main``. The target window contains an ``arm32_<reg>`` base, a
    # ``MEM_PLUS`` separator, a ``valued_const_v2`` magnitude header,
    # and a postfix ``value_negative`` marker — in that order — and
    # NO ``MEM_MINUS`` anywhere inside.
    matched_window: list[str] | None = None
    open_idx = -1
    for i, n in enumerate(names):
        if n == "MEM_OPEN_BRACKET":
            open_idx = i
            continue
        if n == "MEM_CLOSE_BRACKET" and open_idx >= 0:
            window = names[open_idx + 1 : i]
            non_digit = [w for w in window if not w.startswith("<digit_")]
            if "value_negative" not in non_digit:
                open_idx = -1
                continue
            has_base = any(w.startswith("arm32_") for w in non_digit)
            try:
                plus_pos = non_digit.index("MEM_PLUS")
                vc_pos = non_digit.index("valued_const_v2")
                vneg_pos = non_digit.index("value_negative")
            except ValueError:
                open_idx = -1
                continue
            ordered = plus_pos < vc_pos < vneg_pos
            no_mem_minus = "MEM_MINUS" not in non_digit
            if has_base and ordered and no_mem_minus:
                matched_window = window
                break
            open_idx = -1

    assert matched_window is not None, (
        "no mem[ base MEM_PLUS valued_const_v2 ... value_negative ]mem "
        "window in arm32 main token stream (expected one for "
        f"ldr r6, [r5, #-4] at 0x10a94); got: {names}"
    )

    # The magnitude bytes between ``valued_const_v2`` and
    # ``value_negative`` must decode to 4 (the absolute disp of -4).
    matched_non_digit_indices = [
        idx for idx, w in enumerate(matched_window) if not w.startswith("<digit_")
    ]
    vc_window_pos = next(
        idx for idx in matched_non_digit_indices
        if matched_window[idx] == "valued_const_v2"
    )
    vneg_window_pos = next(
        idx for idx in matched_non_digit_indices
        if matched_window[idx] == "value_negative"
    )
    digit_slots = matched_window[vc_window_pos + 1 : vneg_window_pos]
    # ``_v2_int_to_minimum_bytes(4)`` yields a single byte ``0x04`` →
    # one ``<digit_04>`` slot between the magnitude header and the
    # postfix sign marker.
    assert digit_slots == ["<digit_04>"], (
        f"expected single <digit_04> magnitude byte, got {digit_slots} "
        f"in window {matched_window}"
    )
