"""Phase J.2: end-to-end multi-binary smoke test for the Ghidra backend.

Each parametrized binary is tokenized via the standalone ``--batch``
entry-point (no framework, no socket — ``tokenizer._run_standalone``
loops over the queue and calls ``run_tokenizer`` directly). After the
run we cross-check the CSV's outer shape (``version=2`` prelude, header
row with the v2 metadata column name, trailing ``vocabulary`` row), at
least one function row's metadata column round-trips as JSON, and the
``_strings.bin`` sidecar exists.

Marked ``@pytest.mark.slow`` because the Ghidra path takes several
seconds per binary (load + auto-analysis + tokenize). Pre-tokenized
outputs at ``/tmp/asm_smoke/out/`` are NOT reused — each test creates
a fresh tmp output directory so the run-to-completion assertion has
meaning (skip-existing would short-circuit the whole pipeline).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tokenizer.csv_files import open_versioned_csv_reader
from tokenizer.vocab_unifier.loader import load_vocab_manager


# Fixture roots. The zlib-family minigzip binaries live in the main
# repo (not necessarily under each worktree); we resolve via an env
# override first, then fall back to the canonical absolute path the
# Phase J spec documents. The hello-world fixtures for ppc64 / riscv64
# are pre-extracted under ``/tmp/asm_smoke/src/`` (see the
# ``ghidra_default_provider`` memory entry — riscv64 is Ghidra-default
# in v2; ppc64 also stays on the Ghidra path because angr's CFG
# resolver is fragile across the board).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ZLIB_SRC = Path(
    os.environ.get("ASM_TOKENIZER_ZLIB_SRC")
    or "/home/sirati/devel/python/asm-tokenizer/src/zlib"
)
_HELLO_SRC = Path(
    os.environ.get("ASM_TOKENIZER_HELLO_SRC")
    or "/tmp/asm_smoke/src"
)


# (platform-prefix, source-root, binary-relative-path). The platform
# prefix is what ``tokenizer.run_tokenizer`` auto-detects from the
# filename when ``--platform auto`` is passed. Sources differ per
# fixture: zlib binaries live under the repo's ``src/zlib`` tree, while
# ppc64 / riscv64 ``hello`` binaries are pre-extracted under
# ``/tmp/asm_smoke/src``. Every binary is the input to a separate
# tokenize run with its own tmp output dir so the assertions in
# ``test_tokenize_csv_shape`` are independent.
_BINARIES = [
    ("x64",     _ZLIB_SRC,  "x64-gcc-7-Os_minigzip"),
    ("arm32",   _ZLIB_SRC,  "arm32-gcc-7-Os_minigzip"),
    ("arm64",   _ZLIB_SRC,  "arm64-gcc-4.8-Os_minigzip"),
    ("mips32",  _ZLIB_SRC,  "mips32-gcc-9-Os_minigzip"),
    ("mips64",  _ZLIB_SRC,  "mips64-gcc-7-Os_minigzip"),
    ("ppc64",   _HELLO_SRC, "ppc64-clang-10-O0_hello"),
    ("riscv64", _HELLO_SRC, "riscv64-clang-10-O2_hello"),
]


def _has_fixture(source: Path, name: str) -> bool:
    """Concrete-file check; ``pytest.skip`` fires if the fixture isn't
    laid out (e.g. a developer hasn't extracted the hello tarballs).
    """
    return (source / name).is_file()


@pytest.mark.slow
@pytest.mark.parametrize(
    "platform, source, binary_name",
    _BINARIES,
    ids=[name for _, _, name in _BINARIES],
)
def test_tokenize_csv_shape(
    platform: str,
    source: Path,
    binary_name: str,
    tmp_path: Path,
) -> None:
    """Run the tokenizer end-to-end and assert the CSV's wire shape.

    The run goes through the standalone ``--batch`` entry-point with a
    single-line queue file so ``_run_standalone``'s batch path drives
    the whole pipeline (matches how a single-binary smoke is exercised
    by the dispatcher's Phase 1 secondary). ``--platform auto`` is
    intentional — it exercises the filename-prefix detector AND the
    Ghidra-default routing for ARM/MIPS that ``run_tokenizer`` applies
    when the requested backend is ``angr``.
    """
    if not _has_fixture(source, binary_name):
        pytest.skip(f"fixture missing: {source / binary_name}")

    queue = tmp_path / "queue.txt"
    queue.write_text(f"{binary_name}\n")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Use ``-m tokenizer`` so we hit the package's ``__main__`` (which
    # owns argparse + ``_run_standalone`` dispatch). Plain ``python -m
    # tokenizer`` matches the CLI shape the task spec describes; the
    # subprocess boundary also guarantees the Ghidra JVM lifecycle
    # stays scoped to this test (cf. ``disasm_provider_lifecycle``).
    cmd = [
        sys.executable,
        "-m",
        "tokenizer",
        "--backend", "ghidra",
        "--batch", str(queue),
        "--source", str(source),
        "--output", str(out_dir),
        "--platform", "auto",
    ]
    env = dict(os.environ)
    # Force v2 explicitly so the test asserts the v2 shape regardless
    # of any future default flip on ``ASM_TOKENIZER_FORMAT_VERSION``.
    env["ASM_TOKENIZER_FORMAT_VERSION"] = "2"

    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,  # Ghidra tokenize per binary ~30-120s; 10min ceiling.
    )

    assert proc.returncode == 0, (
        f"tokenize failed (rc={proc.returncode}) for {binary_name}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # The standalone path writes to ``<output>/<source-relative-parent>/``.
    # With ``--source <fixture-root>`` and a queue line that's just the
    # binary's basename, ``relative_path.parent`` resolves to ``.`` so
    # the CSV lands directly in ``out_dir``.
    csv_path = out_dir / f"{binary_name}_output.csv"
    strings_path = out_dir / f"{binary_name}_strings.bin"

    assert csv_path.is_file(), f"CSV not emitted at {csv_path}"
    assert strings_path.is_file(), f"strings sidecar not emitted at {strings_path}"

    # Outer shape: ``version=2`` prelude, then header row, then >=1
    # function row, then the trailing ``vocabulary`` row.
    with csv_path.open(newline="", encoding="utf-8") as fh:
        first_byte = fh.read(64)
        fh.seek(0)
        assert first_byte.startswith("version=2"), (
            f"missing v2 prelude; first 64 bytes: {first_byte!r}"
        )
        reader, format_version = open_versioned_csv_reader(fh)
        assert format_version == 2

        header = next(reader)
        # v2 column names: ``metadata`` (not ``opaque_metadata``).
        assert header[0] == "function_name", f"unexpected header: {header}"
        assert header[1] == "occurrence", f"unexpected header: {header}"
        assert header[2] == "tokens_base64", f"unexpected header: {header}"
        assert "metadata" in header, f"v2 metadata column missing: {header}"

        rows = list(reader)
        assert rows, "no function/vocab rows after header"
        # The last row's first cell is ``vocabulary`` per the
        # ``save_vocabulary`` writer. There may be intermediate
        # ``vocabulary`` checkpoints (saved every 16384 functions); the
        # last is the final one.
        assert rows[-1][0] == "vocabulary", (
            f"last row is not the vocab def; first cell: {rows[-1][0]!r}"
        )

        # Drop the trailing ``vocabulary`` row (and any prior
        # checkpoints) — function rows are the ones with a non-empty
        # ``tokens_base64`` and a ``function_name`` that isn't
        # ``vocabulary``.
        function_rows = [r for r in rows if r[0] != "vocabulary"]
        assert function_rows, f"no function rows in {csv_path}"

        # The metadata column is JSON-encoded under v2. Parse the first
        # function's metadata cell; an empty ``{}`` is valid (a function
        # with no classified constants).
        first_meta_cell = function_rows[0][header.index("metadata")]
        meta_obj = json.loads(first_meta_cell)
        assert isinstance(meta_obj, dict), (
            f"metadata column is not a JSON object: {first_meta_cell!r}"
        )

    # Round-trip the vocab so the trailing ``vocabulary`` row is
    # well-formed. ``load_vocab_manager`` returns None on malformed /
    # incomplete CSVs and a valid ``VocabularyManager`` otherwise.
    vm = load_vocab_manager(csv_path)
    assert vm is not None, f"vocab manager failed to load from {csv_path}"
    assert vm.format_version == 2, (
        f"loaded vocab format_version != 2: {vm.format_version}"
    )
