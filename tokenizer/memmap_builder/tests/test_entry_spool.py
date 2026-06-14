"""Unit tests for :class:`tokenizer.memmap_builder._entry_spool.EntrySpool`.

Pins the spill contract the byte-identical build relies on: an appended
entry replays ``==`` to itself (including nested
:class:`CallTargetType` enum members and :class:`VersionKey` frozen
dataclasses), append order is preserved, the spool is re-iterable, and
``close`` unlinks the backing temp file.
"""

from __future__ import annotations

import os

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.memmap_builder._entry_spool import EntrySpool
from tokenizer.memmap_builder.builder import VersionKey


def _matched_entry(name: str) -> dict:
    vkey = VersionKey(arch="x64", compiler="gcc", compilerversion="13", opt="O2")
    return {
        "func_name": name,
        "unique_called": [("callee", CallTargetType.LOCAL), ("ext", CallTargetType.EXTERN)],
        "extern_libraries": {"ext": "libc.so"},
        "version_data": [
            {
                "vkey": vkey,
                "called": [("callee", CallTargetType.LOCAL)],
                "data_offset": 0x40,
                "data_len": 32,
                "token_len": 7,
            }
        ],
    }


def test_round_trip_preserves_nested_enum_and_dataclass(tmp_path) -> None:
    spool = EntrySpool(dir=tmp_path)
    try:
        original = _matched_entry("alpha")
        spool.append(original)
        replayed = list(spool)
        assert len(replayed) == 1
        assert replayed[0] == original
        # The nested enum member survives as the same enum, not a bare int.
        vd = replayed[0]["version_data"][0]
        assert vd["called"][0][1] is CallTargetType.LOCAL
        assert isinstance(vd["vkey"], VersionKey)
    finally:
        spool.close()


def test_append_order_preserved_and_reiterable(tmp_path) -> None:
    spool = EntrySpool(dir=tmp_path)
    try:
        names = [f"fn{i}" for i in range(50)]
        for n in names:
            spool.append(_matched_entry(n))
        first = [e["func_name"] for e in spool]
        second = [e["func_name"] for e in spool]
        assert first == names
        assert second == names  # re-iterable, same order
    finally:
        spool.close()


def test_close_unlinks_backing_file(tmp_path) -> None:
    spool = EntrySpool(dir=tmp_path)
    spool.append(_matched_entry("x"))
    path = spool._path
    assert os.path.exists(path)
    spool.close()
    assert not os.path.exists(path)
    # Idempotent.
    spool.close()


def test_empty_spool_iterates_to_nothing(tmp_path) -> None:
    spool = EntrySpool(dir=tmp_path)
    try:
        assert list(spool) == []
    finally:
        spool.close()
