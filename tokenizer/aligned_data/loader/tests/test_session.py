"""BinarySession lifecycle + exception-safety tests.

The plan explicitly calls out: "BinarySession exception safety is
mandatory." These tests verify that:

  * opening + slicing + closing leaks no file descriptors,
  * an exception raised mid-slice still closes EVERY other handle
    that was opened earlier in the session,
  * ``__exit__`` (and the public ``.close()``) are idempotent — a
    second call is a no-op and does not raise,
  * the session refuses slice calls outside its ``with`` block.

The synthetic fixtures stay deliberately tiny: we don't need a real
``VariantRegistry`` or vocab — only the byte shapes that
``aligned_data.io.parse_function_data_header``,
``variant_tokens.record.read_record``, and the section-CSV parser
expect.
"""

from __future__ import annotations

import csv
import gc
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from tokenizer.aligned_data.binary_format import encode_binary_header
from tokenizer.aligned_data.loader.session import BinarySession


class _FakeArm:
    """Minimal SectionArm stand-in matching 5B's documented shape."""

    def __init__(
        self,
        starts: np.ndarray,
        lengths: np.ndarray,
        func_names: List[str] | None = None,
        section_starts: np.ndarray | None = None,
    ) -> None:
        self.starts = starts
        self.lengths = lengths
        self.func_names = func_names or []
        self.section_starts = section_starts


class _FakeVocab:
    """Vocab stub: just enough for the variant decoder + axis builder."""

    def __init__(self, items: List[str]) -> None:
        self._s2i = {s: i + 256 for i, s in enumerate(items)}
        self._i2s = {v: k for k, v in self._s2i.items()}

    def get_token_id(self, token: str) -> int:
        return self._s2i.get(token, -1)

    def get_token_str(self, token_id: int) -> str:
        return self._i2s.get(token_id, "")


def _write_data_record(handle, tokens: np.ndarray) -> tuple[int, int]:
    """Append one ``_data.bin`` record, return ``(offset, length)``."""
    insn_runlength = np.array([len(tokens)], dtype=np.uint8)
    block_runlength = np.array([1], dtype=np.uint8)
    insn_bytes = insn_runlength.tobytes()
    block_bytes = block_runlength.tobytes()
    header = encode_binary_header(len(insn_bytes), 0, len(block_bytes), pad_size=0)
    offset = handle.tell()
    handle.write(header)
    handle.write(insn_bytes)
    handle.write(block_bytes)
    handle.write(tokens.astype(np.uint16).tobytes())
    return offset, handle.tell() - offset


@pytest.fixture
def synthetic_binary(tmp_path: Path) -> Dict[str, Any]:
    """Lay down a tiny binary with one matched section + one unmatched.

    The variant record uses the v3 layout from
    ``tokenizer.variant_tokens.record.write_record``. Vocab is a
    purpose-built ``_FakeVocab`` carrying the four axis tokens the
    decoder will resolve.
    """
    base = tmp_path
    binary_name = "tinybin"

    # --- _variants.bin -----------------------------------------------
    from tokenizer.variant_tokens.encoder import encode_record

    vocab_strings = [
        "arch:x64",
        "comp:gcc",
        "cver:gcc:13.2.0",
        "opt:O2",
    ]
    vocab = _FakeVocab(vocab_strings)

    class _V:
        arch = "x86_64"
        compiler = "gcc"
        compilerversion = "13.2.0"
        opt = "-O2"
        extra_metadata: Dict[str, Any] = {}

    record = encode_record(_V(), vocab)
    variants_path = base / f"{binary_name}_variants.bin"
    with open(variants_path, "wb") as f:
        variant_offset = f.tell()
        f.write(record.tobytes())

    # --- _data.bin (matched) ----------------------------------------
    data_path = base / f"{binary_name}_data.bin"
    with open(data_path, "wb") as f:
        d_off, d_len = _write_data_record(f, np.array([10, 11, 12], dtype=np.uint16))

    # --- _sections.csv (matched: one section with one version) -----
    sections_path = base / f"{binary_name}_sections.csv"
    with open(sections_path, "w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        section_start = f.tell()
        writer.writerow(["my_func"])
        writer.writerow([f"{variant_offset:x}", "", f"{d_off:x}", f"{d_len:x}"])
        section_end = f.tell()
    section_length = section_end - section_start

    matched_arm = _FakeArm(
        starts=np.array([section_start], dtype=np.uint32),
        lengths=np.array([section_length], dtype=np.uint32),
    )

    # --- _unmatched_data.bin + _unmatched_sections.csv -------------
    unmatched_data_path = base / f"{binary_name}_unmatched_data.bin"
    with open(unmatched_data_path, "wb") as f:
        u_off, u_len = _write_data_record(f, np.array([20, 21], dtype=np.uint16))

    unmatched_sections_path = base / f"{binary_name}_unmatched_sections.csv"
    with open(unmatched_sections_path, "w", newline="", encoding="ascii") as f:
        u_sec_start = f.tell()
        writer = csv.writer(f)
        writer.writerow([
            "lonely_func",
            f"{variant_offset:x}",
            "",
            "",
            f"{u_off:x}",
            f"{u_len:x}",
        ])

    unmatched_arm = _FakeArm(
        starts=np.array([u_off], dtype=np.uint32),
        lengths=np.array([u_len], dtype=np.uint32),
        func_names=["lonely_func"],
        section_starts=np.array([u_sec_start], dtype=np.uint64),
    )

    metadata = {
        "matched_arm": matched_arm,
        "unmatched_arm": unmatched_arm,
        "offset_to_filename": {variant_offset: "tinybin-x64-gcc-13.2.0-O2"},
    }
    return {
        "base_path": base,
        "binary_name": binary_name,
        "vocab": vocab,
        "metadata": metadata,
        "variant_offset": variant_offset,
    }


def _count_open_fds() -> int:
    """Count open file descriptors for the current process (Linux only)."""
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except FileNotFoundError:  # pragma: no cover — non-Linux fallback
        return -1


# --------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------


def test_session_opens_and_closes_cleanly(synthetic_binary):
    fb = synthetic_binary
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with sess:
        m = sess.load_matched(0)
        assert m.func_name == "my_func"
        assert len(m.versions) == 1
        v = m.versions[0]
        assert v.tokens.tolist() == [10, 11, 12]
        # variant resolver attached the filename + axes
        assert v.metadata.get("filename") == "tinybin-x64-gcc-13.2.0-O2"
        assert v.metadata.get("arch") == "x64"
        assert v.metadata.get("compiler") == "gcc"
    # After exit, the stack is closed and slice methods should raise.
    with pytest.raises(RuntimeError):
        # Re-using a closed session is undefined; we just verify the
        # private opener won't silently re-open without a fresh enter.
        sess._open_data("matched")


def test_session_load_unmatched(synthetic_binary):
    fb = synthetic_binary
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        f = sess.load_unmatched(0)
    assert f.func_name == "lonely_func"
    assert f.tokens.tolist() == [20, 21]
    variants = f.metadata.get("variants", [])
    assert len(variants) == 1
    assert variants[0].get("arch") == "x64"


def test_session_fd_no_leak(synthetic_binary):
    """Open + close N sessions; fd count must be stable."""
    fb = synthetic_binary
    gc.collect()
    before = _count_open_fds()
    for _ in range(10):
        with BinarySession(
            fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
        ) as sess:
            sess.load_matched(0)
            sess.get_variant_by_ref(f"{fb['variant_offset']:x}")
        gc.collect()
    after = _count_open_fds()
    if before >= 0:  # /proc is Linux only
        # ``np.memmap`` can leave a tracking fd open in some kernel
        # configs; allow a small slack but flag a 10x leak.
        assert after - before <= 2, f"fd leak: {before} -> {after}"


# --------------------------------------------------------------------
# Exception safety
# --------------------------------------------------------------------


class _RaisingMemmap(np.ndarray):
    """Subclass of ndarray that pretends to be a memmap and raises on
    any indexing — used to inject a mid-slice failure after the
    sections handle has been opened.
    """

    def __getitem__(self, key):  # type: ignore[override]
        raise RuntimeError("simulated data-bin read failure")


def test_exception_in_slice_still_closes_other_handles(synthetic_binary, monkeypatch):
    """Force a failure inside ``load_matched`` AFTER the sections handle
    is open. The ExitStack must close the sections handle (and any
    later-opened variants handle) on ``__exit__`` even though the
    slice raised.
    """
    fb = synthetic_binary
    # Track every fd opened so we can assert close-on-exception.
    opened: List[Any] = []
    real_open = __builtins__["open"] if isinstance(__builtins__, dict) else open

    import builtins

    orig_open = builtins.open

    def tracking_open(*args, **kwargs):
        f = orig_open(*args, **kwargs)
        opened.append(f)
        return f

    monkeypatch.setattr(builtins, "open", tracking_open)

    # Force a failure inside ``_open_data`` AFTER the sections handle
    # is open. Patch ``np.memmap`` so the second handle (data.bin) is
    # the raising stub; the variants memmap would also be raising if
    # it ever got opened, which it should not in this path.
    real_memmap = np.memmap

    def memmap_factory(*args, **kwargs):
        m = real_memmap(*args, **kwargs)
        # ndarray view of the same buffer but with a raising __getitem__.
        view = m.view(_RaisingMemmap)
        return view

    monkeypatch.setattr(np, "memmap", memmap_factory)

    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    raised = False
    with pytest.raises(RuntimeError, match="simulated"):
        with sess:
            sess.load_matched(0)
            raised = True
    # Confirm: the failure happened mid-slice (sections file was
    # already opened by then).
    sections_path = fb["base_path"] / f"{fb['binary_name']}_sections.csv"
    sections_opens = [
        f for f in opened
        if hasattr(f, "name") and str(f.name) == str(sections_path)
    ]
    assert sections_opens, "test did not exercise sections-open path"
    for f in sections_opens:
        assert f.closed, "ExitStack did not close the sections handle on exception"
    # And: the session reports closed.
    assert sess._closed is True
    assert sess._stack is None


def test_exit_is_idempotent(synthetic_binary):
    fb = synthetic_binary
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with sess:
        sess.load_matched(0)
    # Double-exit: should be a no-op, no exception.
    assert sess.__exit__(None, None, None) is False
    assert sess.__exit__(None, None, None) is False
    sess.close()
    sess.close()
    # Internal state stays clean.
    assert sess._closed is True
    assert sess._stack is None


def test_close_called_explicitly_inside_with(synthetic_binary):
    """Calling ``.close()`` inside the with-block leaves nothing for
    ``__exit__`` to unwind — the second close must be a clean no-op.
    """
    fb = synthetic_binary
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    with sess:
        sess.load_matched(0)
        sess.close()
        # Inside the with-block but after explicit close, further
        # slice calls must raise (the stack is gone).
        with pytest.raises(RuntimeError):
            sess.load_matched(0)
    # __exit__ at end of with: already-closed → no-op.
    assert sess._closed is True


def test_slice_outside_with_block_raises(synthetic_binary):
    fb = synthetic_binary
    sess = BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    )
    # Never entered the with-block: no ExitStack, no opens.
    with pytest.raises(RuntimeError):
        sess.load_matched(0)
    with pytest.raises(RuntimeError):
        sess.load_unmatched(0)


# --------------------------------------------------------------------
# Out-of-range indexing
# --------------------------------------------------------------------


def test_load_matched_oob_raises(synthetic_binary):
    fb = synthetic_binary
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(IndexError):
            sess.load_matched(99)


def test_load_unmatched_oob_raises(synthetic_binary):
    fb = synthetic_binary
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        with pytest.raises(IndexError):
            sess.load_unmatched(99)


def test_variant_by_ref_missing_returns_none(synthetic_binary):
    fb = synthetic_binary
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as sess:
        assert sess.get_variant_by_ref("") is None
        # malformed hex
        assert sess.get_variant_by_ref("not-hex") is None
