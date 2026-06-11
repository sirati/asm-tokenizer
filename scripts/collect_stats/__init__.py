"""Corpus-statistics collection for the asm-tokenizer ``out/`` tree.

Builds an SQLite database of per-binary and per-program size statistics
sliced along the compilation axes (ISA, compiler, compiler version,
optimization level) so size-ratio questions can be answered with plain
SQL.

Each submodule owns a single concern and crosses exactly one boundary:

* :mod:`scripts.collect_stats.axes` — parse a fullname into compilation
  axes (no filesystem, no DB).
* :mod:`scripts.collect_stats.discovery` — walk the ``out/`` tree and
  yield the phase-1 binaries and phase-3 programs it finds (no axis
  parsing, no DB).
* :mod:`scripts.collect_stats.vocab` — count the serialized vocabulary
  entries in a per-binary ``_output.csv`` (no DB).
* :mod:`scripts.collect_stats.raw_index` — resolve a raw binary file by
  exact-filename match across user-supplied roots (lazy walk, no DB).
* :mod:`scripts.collect_stats.db` — the SQLite schema and typed row
  writers (knows nothing about discovery/axes internals).
* :mod:`scripts.collect_stats.queries` — the example ratio queries.
* :mod:`scripts.collect_stats.__main__` — CLI orchestration; the only
  place that wires the modules together.
"""
