"""Tests for ``duplicate_function_dump.write_duplicate_function_dump``.

Concern: the collision-detection + pickle-write orchestrator. Pin
(a) the "no collisions -> empty groups list" path, (b) the dup-group
filtering threshold (count >= 2), (c) pickle round-trip of the final
payload, (d) parent-dir auto-creation.

The Java-handle introspector side (``snapshot_function``) is covered
by ``test_function_metadata_snapshot``.

Shared mock scaffolding lives in ``_duplicate_function_dump_mocks``.
"""

from __future__ import annotations

import pickle

from tokenizer.disasm.ghidra_provider.duplicate_function_dump import (
    write_duplicate_function_dump,
)
from tokenizer.tests._duplicate_function_dump_mocks import make_func_mock


def test_dump_skip_when_no_collisions(tmp_path) -> None:
    """No name collisions -> file written, ``duplicate_groups`` empty,
    return value is 0.
    """
    funcs = [
        (0x401000, "foo", make_func_mock("foo", 0x401000)),
        (0x402000, "bar", make_func_mock("bar", 0x402000)),
        (0x403000, "baz", make_func_mock("baz", 0x403000)),
    ]
    dump_path = tmp_path / "dump.pkl"
    count = write_duplicate_function_dump(funcs, "binA", dump_path)
    assert count == 0
    payload = pickle.loads(dump_path.read_bytes())
    assert payload["binary"] == "binA"
    assert payload["duplicate_groups"] == []


def test_dump_collects_collision_groups(tmp_path) -> None:
    """Two functions sharing a name -> one group entry; singletons absent."""
    funcs = [
        (0x401000, "foo", make_func_mock("foo", 0x401000)),
        (0x402000, "foo", make_func_mock("foo", 0x402000)),
        (0x403000, "bar", make_func_mock("bar", 0x403000)),
        (0x404000, "qux", make_func_mock("qux", 0x404000)),
        (0x405000, "qux", make_func_mock("qux", 0x405000)),
        (0x406000, "qux", make_func_mock("qux", 0x406000)),
    ]
    dump_path = tmp_path / "dump.pkl"
    count = write_duplicate_function_dump(funcs, "binB", dump_path)
    assert count == 2  # foo and qux
    payload = pickle.loads(dump_path.read_bytes())
    names = [g["name"] for g in payload["duplicate_groups"]]
    assert names == ["foo", "qux"]
    foo_group = next(g for g in payload["duplicate_groups"] if g["name"] == "foo")
    assert foo_group["count"] == 2
    assert [f["entry"] for f in foo_group["functions"]] == [0x401000, 0x402000]
    qux_group = next(g for g in payload["duplicate_groups"] if g["name"] == "qux")
    assert qux_group["count"] == 3


def test_dump_payload_pickle_round_trips(tmp_path) -> None:
    """The written file is valid pickle and round-trips through pickle.loads
    with structural equality of the in-memory payload.
    """
    funcs = [
        (0x401000, "foo", make_func_mock("foo", 0x401000)),
        (0x402000, "foo", make_func_mock("foo", 0x402000)),
    ]
    dump_path = tmp_path / "dump.pkl"
    write_duplicate_function_dump(funcs, "binC", dump_path)
    payload = pickle.loads(dump_path.read_bytes())
    assert payload["binary"] == "binC"
    # Snapshot for the colliding function must include the Function's
    # class + Name getter result.
    snap = payload["duplicate_groups"][0]["functions"][0]["snapshot"]
    assert snap["_java_class"] == "ghidra.program.model.listing.FunctionDB"
    assert snap["Name"] == "foo"


def test_dump_creates_parent_dirs(tmp_path) -> None:
    """``output_path`` under a non-existent parent dir is auto-created."""
    funcs = [
        (0x401000, "foo", make_func_mock("foo", 0x401000)),
        (0x402000, "foo", make_func_mock("foo", 0x402000)),
    ]
    nested = tmp_path / "deeply" / "nested" / "dump.pkl"
    assert not nested.parent.exists()
    write_duplicate_function_dump(funcs, "binD", nested)
    assert nested.exists()
