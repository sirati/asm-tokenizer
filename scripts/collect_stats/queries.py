"""Example ratio queries over the corpus-stats database.

Single concern: the canonical example SQL and a helper to run them
against a :class:`StatsDB`-built database.  No discovery, no writing.

The examples mirror the questions the database was built to answer:

1. Average ``strings_bin / raw_size`` ratio, grouped by ISA family and
   (separately) by exact ISA — how much of a binary is string data,
   per architecture.
2. Per program, ``sum(raw_size)`` vs ``sum(phase-3 sizes)`` and their
   ratio — how the memmap representation's footprint compares to the
   raw input across all variants of a program.  The detailed variant
   (:data:`PROGRAM_RAW_VS_PHASE3_DETAILED`) additionally reports the raw
   coverage (``n_variants_with_raw`` / ``n_variants``) so the ratio's
   bias under partial raw resolution is visible, and splits the phase-3
   total into ``data_bin`` vs the rest (``z3``'s data.bin alone is
   ~10GB and would otherwise swamp the per-program ratio).
3. Average ``vocab_size`` grouped by ``comp`` × ``optim_level`` — how
   per-binary vocabulary growth varies with compiler and optimisation.

Ratios that would divide by NULL/zero (missing raw size, etc.) are
filtered out in SQL so the averages are over the resolvable population.
"""

from __future__ import annotations

import sqlite3

# (1a) avg(strings_bin / raw_size) by ISA family.
STRINGS_RATIO_BY_FAMILY = """
SELECT b.isa_family,
       COUNT(*)                              AS n,
       AVG(CAST(p.size_bytes AS REAL) / b.raw_size) AS avg_strings_ratio
FROM binaries b
JOIN phase1_files p
  ON p.binary_id = b.id AND p.kind = 'strings_bin'
WHERE b.raw_size IS NOT NULL AND b.raw_size > 0
GROUP BY b.isa_family
ORDER BY b.isa_family
"""

# (1b) avg(strings_bin / raw_size) by exact ISA.
STRINGS_RATIO_BY_ISA = """
SELECT b.isa_exact,
       COUNT(*)                              AS n,
       AVG(CAST(p.size_bytes AS REAL) / b.raw_size) AS avg_strings_ratio
FROM binaries b
JOIN phase1_files p
  ON p.binary_id = b.id AND p.kind = 'strings_bin'
WHERE b.raw_size IS NOT NULL AND b.raw_size > 0
GROUP BY b.isa_exact
ORDER BY b.isa_exact
"""

# (2) per program: sum(raw_size) over its binaries vs sum(phase-3 sizes)
# over its build_memmap artifacts, plus the ratio.  Joined on program
# name (see db.py for why this is a query-time join, not an FK).
PROGRAM_RAW_VS_PHASE3 = """
WITH raw_per_program AS (
    SELECT program, SUM(raw_size) AS raw_total
    FROM binaries
    WHERE raw_size IS NOT NULL
    GROUP BY program
),
phase3_per_program AS (
    SELECT program, SUM(size_bytes) AS phase3_total
    FROM phase3_files
    GROUP BY program
)
SELECT COALESCE(r.program, p.program)         AS program,
       r.raw_total,
       p.phase3_total,
       CASE WHEN r.raw_total > 0
            THEN CAST(p.phase3_total AS REAL) / r.raw_total
            ELSE NULL END                      AS phase3_over_raw
FROM raw_per_program r
FULL OUTER JOIN phase3_per_program p ON p.program = r.program
ORDER BY program
"""

# (2b) the detailed per-program raw↔phase3 report the analysis wants:
# for each program (memmap group) the sum of ALL raw binary variant
# sizes vs the sum of that program's phase-3 file sizes, as a ratio,
# with raw coverage (n_variants_with_raw / n_variants) so the ratio's
# bias under partial raw resolution is visible, and the phase-3 total
# split into ``data_bin`` vs the rest (z3's data.bin alone is ~10GB and
# would otherwise swamp the ratio).  Same query-time program join as
# (2); ``n_variants`` counts ALL binaries of the program (raw or not).
PROGRAM_RAW_VS_PHASE3_DETAILED = """
WITH binaries_per_program AS (
    SELECT program,
           COUNT(*)                                       AS n_variants,
           SUM(raw_size IS NOT NULL)                       AS n_variants_with_raw,
           SUM(raw_size)                                   AS raw_total
    FROM binaries
    GROUP BY program
),
phase3_per_program AS (
    SELECT program,
           SUM(CASE WHEN kind = 'data_bin' THEN size_bytes ELSE 0 END) AS phase3_data_bin,
           SUM(CASE WHEN kind = 'data_bin' THEN 0 ELSE size_bytes END) AS phase3_other,
           SUM(size_bytes)                                             AS phase3_total
    FROM phase3_files
    GROUP BY program
)
SELECT COALESCE(b.program, p.program)              AS program,
       b.n_variants_with_raw,
       b.n_variants,
       b.raw_total,
       p.phase3_data_bin,
       p.phase3_other,
       p.phase3_total,
       CASE WHEN b.raw_total > 0
            THEN CAST(p.phase3_total AS REAL) / b.raw_total
            ELSE NULL END                           AS phase3_over_raw
FROM binaries_per_program b
FULL OUTER JOIN phase3_per_program p ON p.program = b.program
ORDER BY program
"""

# (3) avg(vocab_size) by compiler x optimization level.
VOCAB_BY_COMP_OPTIM = """
SELECT comp,
       optim_level,
       COUNT(*)        AS n,
       AVG(vocab_size) AS avg_vocab_size
FROM binaries
WHERE vocab_size IS NOT NULL
GROUP BY comp, optim_level
ORDER BY comp, optim_level
"""

EXAMPLE_QUERIES: tuple[tuple[str, str], ...] = (
    ("avg(strings_bin/raw_size) by ISA family", STRINGS_RATIO_BY_FAMILY),
    ("avg(strings_bin/raw_size) by exact ISA", STRINGS_RATIO_BY_ISA),
    ("per-program sum(raw) vs sum(phase3) ratio", PROGRAM_RAW_VS_PHASE3),
    (
        "per-program raw↔phase3 ratio (detailed: coverage + data_bin split)",
        PROGRAM_RAW_VS_PHASE3_DETAILED,
    ),
    ("avg(vocab_size) by comp x optim_level", VOCAB_BY_COMP_OPTIM),
)


def run_examples(conn: sqlite3.Connection) -> None:
    """Print each example query's SQL and its result rows."""
    for title, sql in EXAMPLE_QUERIES:
        print(f"\n=== {title} ===")
        print(sql.strip())
        try:
            cursor = conn.execute(sql)
        except sqlite3.OperationalError as exc:
            # FULL OUTER JOIN needs SQLite >= 3.39; degrade loudly rather
            # than crash the whole --examples run on an old runtime.
            print(f"  [skipped: {exc}]")
            continue
        columns = [d[0] for d in cursor.description]
        print("  " + " | ".join(columns))
        for record in cursor.fetchall():
            print("  " + " | ".join("" if v is None else str(v) for v in record))
