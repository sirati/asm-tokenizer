"""Tests for the per-worker Ghidra JVM processor-cap concern.

Two boundaries are covered without booting a real JVM:

1. ``tokenizer.disasm.ghidra_provider.jvm_processor_cap`` — the pure
   cap computation (ceil division, floor-at-1, bad-input rejection) and
   the flag rendering.
2. ``tokenizer.disasm.ghidra_provider.provider._ensure_jvm_started`` —
   that the rendered cap flags ride into the launcher's ``add_vmargs``
   when a cap is installed and are absent when it is cleared. A fake
   ``pyghidra`` module captures the vmargs so no JVM starts.
3. ``tokenizer.disasm.configure_worker_jvm_processor_cap`` — the
   worker-facing orchestrator's fallback-on-unknown (None / undetectable
   cores → no cap, no crash) and its compute+install on a known count.
"""

from __future__ import annotations

import sys
import types

import pytest

from tokenizer.disasm.ghidra_provider.jvm_processor_cap import (
    compute_processor_cap,
    processor_cap_vmargs,
)


# ---------------------------------------------------------------------------
# Pure cap computation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "machine_cores, workers_per_node, expected",
    [
        (32, 1, 32),  # one worker gets the whole machine
        (32, 4, 8),  # exact division
        (32, 5, 7),  # ceil(32/5) = 7, not 6 — aggregate >= machine width
        (32, 7, 5),  # ceil(32/7) = 5
        (32, 32, 1),  # one core each
        (32, 64, 1),  # more workers than cores → floor at 1
        (1, 8, 1),  # single-core box, many workers → floor at 1
        (7, 2, 4),  # ceil(7/2) = 4
    ],
)
def test_compute_processor_cap_ceil_and_floor(
    machine_cores: int, workers_per_node: int, expected: int
) -> None:
    assert compute_processor_cap(machine_cores, workers_per_node) == expected


def test_compute_processor_cap_never_below_one() -> None:
    # Extreme oversubscription still yields a usable per-worker count.
    assert compute_processor_cap(2, 1000) == 1


@pytest.mark.parametrize("bad", [0, -1, -32])
def test_compute_processor_cap_rejects_nonpositive_machine_cores(
    bad: int,
) -> None:
    with pytest.raises(ValueError):
        compute_processor_cap(bad, 4)


@pytest.mark.parametrize("bad", [0, -1, -8])
def test_compute_processor_cap_rejects_nonpositive_workers(bad: int) -> None:
    with pytest.raises(ValueError):
        compute_processor_cap(32, bad)


def test_processor_cap_vmargs_renders_both_knobs() -> None:
    args = processor_cap_vmargs(8)
    assert args == (
        "-XX:ActiveProcessorCount=8",
        "-Dcpu.core.limit=8",
    )


# ---------------------------------------------------------------------------
# Launcher vmargs assembly (no real JVM)
# ---------------------------------------------------------------------------


class _FakeLauncher:
    """Captures the vmargs handed to ``add_vmargs`` so the test can
    assert the cap flags' presence/absence without starting a JVM."""

    def __init__(self) -> None:
        self.vmargs: list[str] = []

    def add_vmargs(self, *args: str) -> None:
        self.vmargs.extend(args)

    def start(self) -> None:  # JVM boot — a no-op in the fake
        pass


@pytest.fixture
def fake_pyghidra(monkeypatch: pytest.MonkeyPatch) -> _FakeLauncher:
    """Install a fake ``pyghidra`` module reporting ``started() is False``
    and handing out one capturing launcher; yields that launcher."""
    launcher = _FakeLauncher()
    module = types.ModuleType("pyghidra")
    module.started = lambda: False  # type: ignore[attr-defined]
    module.HeadlessPyGhidraLauncher = lambda: launcher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyghidra", module)
    return launcher


def test_jvm_args_include_cap_when_set(
    fake_pyghidra: _FakeLauncher, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokenizer.disasm.ghidra_provider import provider

    monkeypatch.setattr(
        provider,
        "_PROCESSOR_CAP_VMARGS",
        processor_cap_vmargs(6),
        raising=True,
    )
    provider._ensure_jvm_started()
    assert "-XX:ActiveProcessorCount=6" in fake_pyghidra.vmargs
    assert "-Dcpu.core.limit=6" in fake_pyghidra.vmargs
    # The static GC tuning still rides alongside the cap.
    assert "-XX:+UseShenandoahGC" in fake_pyghidra.vmargs


def test_jvm_args_omit_cap_when_unset(
    fake_pyghidra: _FakeLauncher, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokenizer.disasm.ghidra_provider import provider

    monkeypatch.setattr(provider, "_PROCESSOR_CAP_VMARGS", (), raising=True)
    provider._ensure_jvm_started()
    assert not any(
        a.startswith("-XX:ActiveProcessorCount") for a in fake_pyghidra.vmargs
    )
    assert not any(
        a.startswith("-Dcpu.core.limit") for a in fake_pyghidra.vmargs
    )
    # Default GC tuning is still present (the cap is the only thing gated).
    assert "-XX:+UseShenandoahGC" in fake_pyghidra.vmargs


def test_set_processor_cap_none_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenizer.disasm.ghidra_provider import provider

    provider.set_processor_cap(5)
    assert provider._PROCESSOR_CAP_VMARGS == processor_cap_vmargs(5)
    provider.set_processor_cap(None)
    assert provider._PROCESSOR_CAP_VMARGS == ()


# ---------------------------------------------------------------------------
# Worker-facing orchestrator (fallback-on-unknown)
# ---------------------------------------------------------------------------


def test_configure_caps_on_known_count(monkeypatch: pytest.MonkeyPatch) -> None:
    import tokenizer.disasm as disasm
    from tokenizer.disasm.ghidra_provider import provider

    monkeypatch.setattr(disasm.os, "cpu_count", lambda: 32)
    disasm.configure_worker_jvm_processor_cap(4)
    assert provider._PROCESSOR_CAP_VMARGS == processor_cap_vmargs(8)
    provider.set_processor_cap(None)  # leave global state clean


def test_configure_no_cap_when_workers_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenizer.disasm as disasm
    from tokenizer.disasm.ghidra_provider import provider

    provider.set_processor_cap(99)  # prove the call clears it
    monkeypatch.setattr(disasm.os, "cpu_count", lambda: 32)
    disasm.configure_worker_jvm_processor_cap(None)
    assert provider._PROCESSOR_CAP_VMARGS == ()


def test_configure_no_cap_when_cores_undetectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenizer.disasm as disasm
    from tokenizer.disasm.ghidra_provider import provider

    provider.set_processor_cap(99)
    monkeypatch.setattr(disasm.os, "cpu_count", lambda: None)
    disasm.configure_worker_jvm_processor_cap(4)
    assert provider._PROCESSOR_CAP_VMARGS == ()
