"""Standalone ``--batch`` runner: a queued binary that produces no
output must make the process exit non-zero, never silently succeed.

The standalone batch path (``tokenizer.__main__._run_standalone``)
loops over a queue and calls ``run_tokenizer`` per binary. A
``run_tokenizer`` failure (e.g. a Ghidra/JVM bring-up error under
load) used to be swallowed at INFO level with ``continue``, so the
batch always exited 0 even when it emitted nothing — a silent
failure that a smoke test only caught via a downstream
``csv_path.is_file()`` assert. These tests pin the contract that the
runner owns: "every queued binary either produced output, was an
explicit skip-existing hit, or the run fails hard."

We monkeypatch ``run_tokenizer`` so the outcome (raise / clean
return / skip-sentinel) is deterministic — the failing-binary race
is reproduced by its *consequence* (a raising ``run_tokenizer``)
without paying for a real disassembly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tokenizer.__main__ as cli


def _make_args(queue: Path, source: Path, output: Path) -> object:
    """Parse a real ``--batch`` invocation so the namespace matches
    production exactly (defaults, types, mutually-exclusive group).
    """
    return cli._build_argparser().parse_args(
        [
            "--batch",
            str(queue),
            "--source",
            str(source),
            "--output",
            str(output),
            "--platform",
            "auto",
            "--backend",
            "ghidra",
        ]
    )


def test_batch_failed_binary_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queued binary whose tokenize raises makes the batch exit 1."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "x64-gcc-7-O0_prog").write_bytes(b"\x7fELF")
    output = tmp_path / "out"
    output.mkdir()
    queue = tmp_path / "queue.txt"
    queue.write_text("x64-gcc-7-O0_prog\n")

    def _boom(*_args: object, **_kwargs: object) -> tuple[int, int]:
        raise RuntimeError("simulated Ghidra bring-up failure")

    monkeypatch.setattr(cli, "run_tokenizer", _boom)

    with pytest.raises(SystemExit) as exc:
        cli._run_standalone(_make_args(queue, source, output))
    assert exc.value.code == 1


def test_batch_all_success_exits_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queue whose binaries all tokenize cleanly returns (no exit)."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "x64-gcc-7-O0_prog").write_bytes(b"\x7fELF")
    output = tmp_path / "out"
    output.mkdir()
    queue = tmp_path / "queue.txt"
    queue.write_text("x64-gcc-7-O0_prog\n")

    monkeypatch.setattr(cli, "run_tokenizer", lambda *a, **k: (0, 0))

    # No SystemExit raised == clean exit.
    cli._run_standalone(_make_args(queue, source, output))


def test_batch_skip_existing_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit skip-existing sentinel ``(-1, -1)`` is a clean,
    logged zero-output outcome — it must NOT be conflated with a
    no-output failure.
    """
    source = tmp_path / "src"
    source.mkdir()
    (source / "x64-gcc-7-O0_prog").write_bytes(b"\x7fELF")
    output = tmp_path / "out"
    output.mkdir()
    queue = tmp_path / "queue.txt"
    queue.write_text("x64-gcc-7-O0_prog\n")

    monkeypatch.setattr(cli, "run_tokenizer", lambda *a, **k: (-1, -1))

    cli._run_standalone(_make_args(queue, source, output))


def test_batch_partial_failure_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One good + one failing binary still fails the whole batch."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "x64-gcc-7-O0_good").write_bytes(b"\x7fELF")
    (source / "x64-gcc-7-O0_bad").write_bytes(b"\x7fELF")
    output = tmp_path / "out"
    output.mkdir()
    queue = tmp_path / "queue.txt"
    queue.write_text("x64-gcc-7-O0_good\nx64-gcc-7-O0_bad\n")

    def _selective(binary_path: Path, **_kwargs: object) -> tuple[int, int]:
        if "bad" in Path(binary_path).name:
            raise RuntimeError("simulated failure on the bad binary")
        return (0, 0)

    monkeypatch.setattr(cli, "run_tokenizer", _selective)

    with pytest.raises(SystemExit) as exc:
        cli._run_standalone(_make_args(queue, source, output))
    assert exc.value.code == 1


def test_batch_queue_path_outside_source_fails_hard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queue line that resolves outside the source root is an
    operator error, not a skip: ``filter_queue`` raises rather than
    silently shrinking the batch to zero work.
    """
    source = tmp_path / "src"
    source.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    # Absolute path under tmp_path but NOT under source/.
    stray = tmp_path / "elsewhere" / "x64-gcc-7-O0_prog"
    stray.parent.mkdir()
    stray.write_bytes(b"\x7fELF")
    queue = tmp_path / "queue.txt"
    queue.write_text(f"{stray}\n")

    # run_tokenizer must never be reached — the malformed path is
    # rejected during filtering.
    monkeypatch.setattr(
        cli, "run_tokenizer", lambda *a, **k: pytest.fail("reached run_tokenizer")
    )

    with pytest.raises(ValueError, match="not under the source root"):
        cli._run_standalone(_make_args(queue, source, output))
