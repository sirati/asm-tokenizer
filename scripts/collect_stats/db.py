"""SQLite schema and typed row writers for the corpus-stats database.

Single concern: persist typed row records into SQLite.  This module
knows the table layout but nothing about how rows are discovered,
parsed, or counted — callers hand it fully-formed dataclass rows.

Schema
------

``binaries`` — one row per discovered phase-1 binary::

    id            INTEGER PRIMARY KEY
    fullname      TEXT NOT NULL
    program       TEXT            -- part of fullname after first '_'
    package       TEXT NOT NULL   -- first path component under out/
    isa_exact     TEXT            -- literal ISA token (x86_64, armv7l-hf, ...)
    isa_family    TEXT            -- x86 | arm | mips | ppc | riscv
    bitness       INTEGER         -- 32 | 64
    comp          TEXT            -- compiler family (gcc, clang, ...)
    comp_version  TEXT            -- may contain dashes
    optim_level   TEXT            -- O0..O3, Os, Oz, Ofast, ...
    raw_size      INTEGER         -- NULL when the raw binary was not found
    vocab_size    INTEGER         -- serialized wire vocab count; NULL on miss

``phase1_files`` — one row per phase-1 artifact present::

    binary_id     INTEGER NOT NULL REFERENCES binaries(id)
    kind          TEXT NOT NULL   -- output_csv | meta_json | strings_bin | ...
    size_bytes    INTEGER NOT NULL

``phase3_files`` — one row per phase-3 build_memmap artifact::

    program       TEXT NOT NULL
    kind          TEXT NOT NULL   -- data_bin | index_bin | sections_bin | ...
    size_bytes    INTEGER NOT NULL

Axis fields are nullable so an unparseable fullname is recorded (with a
warning) rather than dropped.  ``raw_size`` / ``vocab_size`` are nullable
to distinguish "not resolved" from a genuine zero.  ``phase3_files`` is
keyed by ``program`` (not a foreign key) because phase-3 programs are
discovered independently and a binary's ``program`` may or may not match
a build_memmap directory name (e.g. dataset binaries whose build_memmap
dir drops the per-variant hash suffix); the join is left to query time.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE binaries (
    id           INTEGER PRIMARY KEY,
    fullname     TEXT    NOT NULL,
    program      TEXT,
    package      TEXT    NOT NULL,
    isa_exact    TEXT,
    isa_family   TEXT,
    bitness      INTEGER,
    comp         TEXT,
    comp_version TEXT,
    optim_level  TEXT,
    raw_size     INTEGER,
    vocab_size   INTEGER
);

CREATE TABLE phase1_files (
    binary_id  INTEGER NOT NULL REFERENCES binaries(id),
    kind       TEXT    NOT NULL,
    size_bytes INTEGER NOT NULL
);

CREATE TABLE phase3_files (
    program    TEXT    NOT NULL,
    kind       TEXT    NOT NULL,
    size_bytes INTEGER NOT NULL
);

CREATE INDEX idx_binaries_program ON binaries(program);
CREATE INDEX idx_binaries_isa_family ON binaries(isa_family);
CREATE INDEX idx_phase1_binary ON phase1_files(binary_id);
CREATE INDEX idx_phase3_program ON phase3_files(program);
"""


@dataclass(frozen=True, slots=True)
class BinaryRow:
    """A row for the ``binaries`` table.  All axis/size fields nullable."""

    fullname: str
    program: str | None
    package: str
    isa_exact: str | None
    isa_family: str | None
    bitness: int | None
    comp: str | None
    comp_version: str | None
    optim_level: str | None
    raw_size: int | None
    vocab_size: int | None


@dataclass(frozen=True, slots=True)
class Phase1Row:
    """A row for the ``phase1_files`` table."""

    binary_id: int
    kind: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Phase3Row:
    """A row for the ``phase3_files`` table."""

    program: str
    kind: str
    size_bytes: int


class StatsDB:
    """Thin typed writer around an SQLite connection.

    The schema is created once via :meth:`create_schema`; callers then
    stream typed rows in.  ``insert_binary`` returns the new row id so
    the caller can key its phase-1 file rows to it.
    """

    def __init__(self, path: Path) -> None:
        self._conn = sqlite3.connect(path)

    def create_schema(self) -> None:
        self._conn.executescript(_SCHEMA)

    def insert_binary(self, row: BinaryRow) -> int:
        cur = self._conn.execute(
            "INSERT INTO binaries (fullname, program, package, isa_exact, "
            "isa_family, bitness, comp, comp_version, optim_level, raw_size, "
            "vocab_size) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                row.fullname,
                row.program,
                row.package,
                row.isa_exact,
                row.isa_family,
                row.bitness,
                row.comp,
                row.comp_version,
                row.optim_level,
                row.raw_size,
                row.vocab_size,
            ),
        )
        return cur.lastrowid

    def insert_phase1_files(self, rows: list[Phase1Row]) -> None:
        self._conn.executemany(
            "INSERT INTO phase1_files (binary_id, kind, size_bytes) VALUES (?,?,?)",
            [(r.binary_id, r.kind, r.size_bytes) for r in rows],
        )

    def insert_phase3_files(self, rows: list[Phase3Row]) -> None:
        self._conn.executemany(
            "INSERT INTO phase3_files (program, kind, size_bytes) VALUES (?,?,?)",
            [(r.program, r.kind, r.size_bytes) for r in rows],
        )

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> StatsDB:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
