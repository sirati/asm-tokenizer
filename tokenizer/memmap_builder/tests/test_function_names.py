"""Round-trip + invariants for the function-names sidecar."""

from __future__ import annotations

import random
import string

import pytest

from tokenizer.aligned_data.loader.function_names_loader import (
    load_function_names,
)
from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.memmap_builder.function_names import FunctionNamesRegistry


def _random_name(rng: random.Random, max_len: int = 24) -> str:
    n = rng.randint(1, max_len)
    alphabet = string.ascii_letters + string.digits + "_"
    return "".join(rng.choice(alphabet) for _ in range(n))


def test_round_trip_1000_names_mixed_case_and_duplicates(tmp_path):
    rng = random.Random(0xC0FFEE)
    # 1000 unique-ish names, then a bunch of duplicates mixed in.
    base_names = {_random_name(rng) for _ in range(1500)}
    # pick 1000 of them deterministically
    names_unique = sorted(base_names)[:1000]
    raw_inputs = list(names_unique) + rng.choices(names_unique, k=500)
    rng.shuffle(raw_inputs)

    reg = FunctionNamesRegistry()
    for name in raw_inputs:
        reg.add(name)
    reg.finalize()

    sidecar = reg.write_sidecar(tmp_path, "binary42")
    assert sidecar == tmp_path / "binary42_function_names.txt"
    assert sidecar.exists()

    # First line is the prelude verbatim.
    with open(sidecar, "r", encoding="utf-8") as f:
        first_line = f.readline()
    assert first_line == f"# format={MEMMAP_FORMAT_VERSION}\n"

    name_to_line, line_to_name = load_function_names(sidecar)

    expected_sorted = sorted(set(raw_inputs))
    assert len(name_to_line) == len(expected_sorted)
    assert len(line_to_name) == len(expected_sorted)

    # Alphabetical 1-indexed mapping is consistent both directions.
    for i, name in enumerate(expected_sorted, start=1):
        assert name_to_line[name] == i
        assert line_to_name[i] == name
        assert reg.line_no(name) == i


def test_add_after_finalize_raises():
    reg = FunctionNamesRegistry()
    reg.add("foo")
    reg.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        reg.add("bar")


def test_line_no_unknown_raises_key_error():
    reg = FunctionNamesRegistry()
    reg.add("foo")
    reg.finalize()
    with pytest.raises(KeyError):
        reg.line_no("never_registered")


def test_write_sidecar_before_finalize_raises(tmp_path):
    reg = FunctionNamesRegistry()
    reg.add("foo")
    with pytest.raises(RuntimeError, match="not finalized"):
        reg.write_sidecar(tmp_path, "binary42")


def test_line_no_before_finalize_raises():
    reg = FunctionNamesRegistry()
    reg.add("foo")
    with pytest.raises(RuntimeError, match="not finalized"):
        reg.line_no("foo")


def test_load_sidecar_missing_prelude_raises(tmp_path):
    bad = tmp_path / "bad_function_names.txt"
    bad.write_text("foo\nbar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prelude"):
        load_function_names(bad)


def test_load_sidecar_wrong_version_raises(tmp_path):
    bad = tmp_path / "future_function_names.txt"
    bad.write_text("# format=2\nfoo\nbar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prelude"):
        load_function_names(bad)


def test_finalize_is_idempotent(tmp_path):
    reg = FunctionNamesRegistry()
    reg.add("alpha")
    reg.add("beta")
    reg.finalize()
    snapshot_alpha = reg.line_no("alpha")
    snapshot_beta = reg.line_no("beta")
    reg.finalize()  # no-op
    assert reg.line_no("alpha") == snapshot_alpha
    assert reg.line_no("beta") == snapshot_beta


def test_empty_registry_round_trips(tmp_path):
    reg = FunctionNamesRegistry()
    reg.finalize()
    sidecar = reg.write_sidecar(tmp_path, "emptybin")
    name_to_line, line_to_name = load_function_names(sidecar)
    assert name_to_line == {}
    assert line_to_name == {}
