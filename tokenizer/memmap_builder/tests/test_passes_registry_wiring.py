"""Pass-1 walker integration with the FunctionNamesRegistry.

The walkers take the registry as a required positional argument and
add every function name they emit (header function + called-function
references) so pass 2 can resolve the section-CSV cells to 1-indexed
sidecar line numbers.

These tests stub :func:`process_function_binary_data` so they can
focus on the registry-wiring concern without re-creating valid
binary-encoded fixtures — the encoder's correctness is owned by
``test_builder_error_log.py`` and the ``_writers`` test suite.
"""

from __future__ import annotations

import io

import pytest

from tokenizer.memmap_builder import passes as passes_mod
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry
from tokenizer.memmap_builder.helpers import FunctionBinaryData
from tokenizer.memmap_builder.passes import (
    process_matched_function_pass1,
    process_unmatched_function_pass1,
)


class _FakeVKey:
    """Minimal hashable vkey stand-in; identity is enough for the tests."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeVKey) and self.label == other.label

    def __repr__(self) -> str:
        return f"_FakeVKey({self.label!r})"


def _make_row_with_metadata(callees: list[str]) -> dict:
    """Build a v2-style row dict with the minimum fields the walkers read.

    The walkers consult:
      * ``row["block_runlength_base64"]`` via
        ``should_skip_function_for_matched`` — empty string is rejected
        by ``base64_to_ndarray_vec`` and the helper treats that as a
        skip; supply an empty array's base64 instead (``""``-decoded
        gives a zero-length array, sum < 4096 -> not skipped).
      * ``row["block_runlength"]`` via
        ``should_skip_function_for_unmatched`` — when absent the helper
        falls through cleanly.
      * ``row["metadata"]`` via ``get_called_functions_from_row`` (v2
        schema: JSON with a ``local_funcs`` array of ``{"name": ...}``).

    The actual binary encoding is stubbed out so the row need not carry
    valid ``tokens_base64`` / runlength payloads.
    """
    metadata_json = (
        '{"local_funcs": ['
        + ",".join(f'{{"name": "{name}"}}' for name in callees)
        + '], "plt_funcs": [], "ext_funcs": []}'
    )
    # `AAAA` is base64 for `\0\0\0` -> ndarray([0,0,0]); sum=0 < 4096.
    return {
        "block_runlength_base64": "AAAA",
        "metadata": metadata_json,
    }


def _stub_binary_data(offset: int, length: int = 16) -> FunctionBinaryData:
    return FunctionBinaryData(data_offset=offset, data_len=length, token_len=length // 2)


def test_matched_walker_requires_registry_argument():
    """The walker signature now treats ``registry`` as a required
    positional argument; calls that omit it raise ``TypeError``."""
    with pytest.raises(TypeError, match="registry"):
        # Intentionally drop the registry to confirm the signature.
        process_matched_function_pass1(  # type: ignore[call-arg]
            "fn",
            [None],
            [_FakeVKey("a")],
            {},
            io.BytesIO(),
        )


def test_unmatched_walker_requires_registry_argument():
    with pytest.raises(TypeError, match="registry"):
        process_unmatched_function_pass1(  # type: ignore[call-arg]
            "fn",
            [None],
            [_FakeVKey("a")],
            {},
            io.BytesIO(),
        )


def test_matched_walker_records_header_and_callees_on_emit(monkeypatch):
    """A function that survives encoding records its name + every
    called-function name it references into the registry."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    rows = [
        _make_row_with_metadata(["alpha_callee", "beta_callee"]),
        _make_row_with_metadata(["alpha_callee", "gamma_callee"]),
    ]

    # Stub the encoder so each version "writes" to a unique offset (the
    # walker drops the function when all versions share an offset — the
    # dedup-result heuristic).
    offsets = iter([0, 16])
    monkeypatch.setattr(
        passes_mod,
        "process_function_binary_data",
        lambda *a, **kw: _stub_binary_data(next(offsets)),
    )

    registry = FunctionNamesRegistry()
    entry = process_matched_function_pass1(
        "header_fn",
        rows,
        [vkey_a, vkey_b],
        {},
        io.BytesIO(),
        registry,
    )

    assert entry is not None
    registry.finalize()
    expected = {"header_fn", "alpha_callee", "beta_callee", "gamma_callee"}
    # All emitted names landed in the registry; nothing else slipped in.
    assert set(registry._sorted) == expected  # noqa: SLF001 — test-side white-box


def test_matched_walker_skips_registry_when_function_dropped(monkeypatch):
    """All versions reported the same data offset -> the function is
    dropped (the dedup-result heuristic) and no name is recorded.

    Tests the contract that the registry only holds names pass 2 will
    actually write into a section CSV.
    """
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    rows = [
        _make_row_with_metadata(["only_callee"]),
        _make_row_with_metadata(["only_callee"]),
    ]
    monkeypatch.setattr(
        passes_mod,
        "process_function_binary_data",
        lambda *a, **kw: _stub_binary_data(0),  # both versions hit offset 0
    )

    registry = FunctionNamesRegistry()
    entry = process_matched_function_pass1(
        "deduped_fn",
        rows,
        [vkey_a, vkey_b],
        {},
        io.BytesIO(),
        registry,
    )

    assert entry is None
    registry.finalize()
    assert registry._sorted == ()  # noqa: SLF001


def test_matched_walker_skips_registry_when_all_versions_cap_overflow(monkeypatch):
    """All ``process_function_binary_data`` calls returned ``None``
    (cap overflow + error_log) -> walker drops the function and the
    registry stays empty for that name."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    rows = [
        _make_row_with_metadata(["cap_overflow_callee"]),
        _make_row_with_metadata(["cap_overflow_callee"]),
    ]
    monkeypatch.setattr(
        passes_mod,
        "process_function_binary_data",
        lambda *a, **kw: None,
    )

    registry = FunctionNamesRegistry()
    entry = process_matched_function_pass1(
        "overflow_fn",
        rows,
        [vkey_a, vkey_b],
        {},
        io.BytesIO(),
        registry,
    )

    assert entry is None
    registry.finalize()
    assert registry._sorted == ()  # noqa: SLF001


def test_unmatched_walker_records_header_and_per_version_callees(monkeypatch):
    """Each surviving version contributes ``func_name`` + that version's
    callees to the registry. The registry's set semantics dedupe; the
    walker need not preempt that."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    rows = [
        _make_row_with_metadata(["x_callee"]),
        _make_row_with_metadata(["y_callee", "z_callee"]),
    ]

    offsets = iter([0, 32])
    monkeypatch.setattr(
        passes_mod,
        "process_function_binary_data",
        lambda *a, **kw: _stub_binary_data(next(offsets)),
    )

    registry = FunctionNamesRegistry()
    entries = process_unmatched_function_pass1(
        "unmatched_fn",
        rows,
        [vkey_a, vkey_b],
        {},
        io.BytesIO(),
        registry,
    )

    assert len(entries) == 2
    registry.finalize()
    expected = {"unmatched_fn", "x_callee", "y_callee", "z_callee"}
    assert set(registry._sorted) == expected  # noqa: SLF001


def test_unmatched_walker_skips_registry_for_cap_overflow_version(monkeypatch):
    """The version whose encoder returned ``None`` is dropped; its
    callees do NOT enter the registry. Surviving versions still record."""
    vkey_a = _FakeVKey("a")
    vkey_b = _FakeVKey("b")
    rows = [
        _make_row_with_metadata(["surviving_callee"]),
        _make_row_with_metadata(["overflow_only_callee"]),
    ]

    # First call (vkey_a) succeeds; second (vkey_b) overflows.
    results = iter([_stub_binary_data(0), None])
    monkeypatch.setattr(
        passes_mod,
        "process_function_binary_data",
        lambda *a, **kw: next(results),
    )

    registry = FunctionNamesRegistry()
    entries = process_unmatched_function_pass1(
        "partial_fn",
        rows,
        [vkey_a, vkey_b],
        {},
        io.BytesIO(),
        registry,
    )

    assert len(entries) == 1
    registry.finalize()
    assert set(registry._sorted) == {"partial_fn", "surviving_callee"}  # noqa: SLF001


def test_unmatched_walker_propagates_unexpected_exception(monkeypatch):
    """The bare ``except Exception: pass`` is removed. Programmer-error
    or IO failures from the encoder are no longer silently swallowed —
    they surface immediately so an operator sees the actual cause."""
    vkey_a = _FakeVKey("a")
    rows = [_make_row_with_metadata(["any_callee"])]

    class _SyntheticEncoderBug(RuntimeError):
        pass

    def _boom(*_a, **_kw):
        raise _SyntheticEncoderBug("encoder invariant violated")

    monkeypatch.setattr(passes_mod, "process_function_binary_data", _boom)

    registry = FunctionNamesRegistry()
    with pytest.raises(_SyntheticEncoderBug):
        process_unmatched_function_pass1(
            "leaky_fn",
            rows,
            [vkey_a],
            {},
            io.BytesIO(),
            registry,
        )
