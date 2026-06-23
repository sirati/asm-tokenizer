// Thesis section: size characterisation of the tokeniser pipeline output.
//
// Self-contained and compilable on its own (`typst compile dataset_size_stats.typ`),
// but written as a section so it can be dropped into a larger thesis with
// `#include "dataset_size_stats.typ"`. All figures are produced by the
// reporters in `scripts/size_stats/` applied to the LMU `gen-binary-dataset`
// corpus and its `tokenizer-gen-binary-dataset` outputs (run 2026-06-23);
// the raw per-binary and per-program data live alongside this file in
// `phase1_report.csv` and `memmap_report.csv`.

#set par(justify: true)
#set table(stroke: 0.5pt + luma(180))
#show table.cell.where(y: 0): strong

= Representation size of the tokenised corpus <sec:size-stats>

This section quantifies how the on-disk size of the binary corpus changes as it
passes through the two size-relevant stages of the pipeline: the per-binary
tokenisation (phase 1), which emits a textual CSV token stream per binary, and
the corpus-level memory-map build (phase 3), which packs every tokenised binary
into the binary artefacts the data loader consumes. Two questions motivate the
measurement: how much the textual token representation inflates a binary, and
whether the final packed representation is competitive in size with the raw
inputs it was derived from.

== Corpus

The measured corpus is the `gen-binary-dataset` build: #strong[167,248] raw ELF
files (executables and shared objects) produced by compiling 32 source packages
across a matrix of nine target architectures (`ppc64`, `x86_64`, `armv7l-hf`,
`mipsel`, `i686`, `mips64el`, `aarch64`, `ppc32`, `riscv64`), two compiler
families (LLVM/Clang and GCC, spanning Clang 7–22 and GCC 4.4–15), and the
optimisation levels `O0`–`O3`, `Os`, `Ofast` and `Oz`. Of these,
#strong[83,971] binaries (52,029 executables and 31,942 shared objects) carry a
phase-1 tokeniser output and thus enter the size analysis; the remainder are
present on disk but outside the tokenised set. @tab:corpus-arch summarises the
architecture distribution of the tokenised binaries.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, left, right, left) + (right,) * 0,
    table.header([Architecture], [Binaries], [Architecture], [Binaries], []),
    [`ppc64`],    [17,569], [`mips64el`], [8,128], [],
    [`x86_64`],   [12,560], [`aarch64`],  [7,261], [],
    [`armv7l-hf`], [9,133], [`ppc32`],    [7,120], [],
    [`mipsel`],    [8,677], [`riscv64`],  [5,118], [],
    [`i686`],      [8,405], [#emph[total]], [#emph[83,971]], [],
  ),
  caption: [Architecture distribution of the 83,971 tokenised binaries.],
) <tab:corpus-arch>

== Method

Sizes are collected by two standalone reporters (`scripts/size_stats/`). Both
take the raw-binary root and the tokeniser output root as arguments and walk the
trees with a single `find` pass each, so the measurement is reproducible and
free of any pipeline state.

The phase-1 reporter pairs each binary with its tokeniser output by directory
identity: a binary's phase-1 artefacts live at
`<pkg>/<variant>/<binname>/` under the output root, a path identical to the raw
binary's path under the source root. Only the canonical `_output.csv` token
stream is counted as "the phase-1 output"; the metadata sidecars
(`_meta.json`, `_strings.bin`, `_function_ranges.txt`) are excluded.

The phase-3 reporter sums every artefact under `build_memmap/` #emph[except] the
human-readable `*_sections.csv` mirrors (both the matched and the unmatched
section CSVs), which are verbose textual duplicates of the packed
`*_sections.bin` catalogues and are not part of the consumed representation.
Everything else — the `*.bin` payloads and indices, the sorted `*.idx` files,
the variant and vocabulary tables, and the name/log sidecars — is counted.
Because a `build_memmap/<program>/` directory is keyed by binary name and merges
every variant of that binary (name collisions are hash-disambiguated, e.g.
`<hash>_pzstd`), the phase-3 figure is reported as a single corpus-level ratio
rather than per binary.

== Phase 1: textual token stream vs. raw binary

The tokeniser output is consistently larger than the binary it describes.
Across the tokenised corpus the aggregate token-stream size is #strong[2.07#sym.times]
the aggregate raw size, and the per-binary inflation is tightly concentrated
(median #strong[2.27#sym.times], interquartile range 1.56#sym.times–2.69#sym.times).
The distribution has a long but thin low tail — the minimum 0.0006#sym.times
corresponds to binaries whose disassembly yields a near-empty token stream —
while the upper tail is bounded below 5#sym.times. @tab:phase1-size reports the
aggregate figures and @tab:phase1-dist the per-binary distribution.

#figure(
  table(
    columns: (auto, auto),
    align: (left, right),
    table.header([Metric], [Value]),
    [Tokenised binaries],            [83,971],
    [Raw size (sum)],                [53.57 GiB],
    [Token-stream size (sum)],       [110.71 GiB],
    [Aggregate token-stream / raw],  [2.07#sym.times],
  ),
  caption: [Phase-1 aggregate size: `_output.csv` token streams vs. raw ELF binaries.],
) <tab:phase1-size>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left,) + (right,) * 6,
    table.header([], [p05], [p25], [median], [p75], [p95], [max]),
    [token-stream / raw], [0.75#sym.times], [1.56#sym.times], [2.27#sym.times],
      [2.69#sym.times], [3.29#sym.times], [4.88#sym.times],
  ),
  caption: [Per-binary phase-1 size ratio distribution (mean 2.14#sym.times).],
) <tab:phase1-dist>

== Phase 3: packed memory map vs. raw corpus

The packed representation reverses the inflation seen in phase 1. The counted
`build_memmap` artefacts total #strong[80.07 GiB] across 133 program groups —
#strong[0.98#sym.times] the size of the entire 167,248-file raw corpus
(81.95 GiB), and #strong[1.49#sym.times] the size of the 53.57 GiB tokenised
subset it is actually derived from. Excluding the `*_sections.csv` textual
mirrors removes a further 3.32 GiB of redundant data that would otherwise be
carried on disk. @tab:memmap-size gives the aggregate figures and
@tab:memmap-top the eight largest program groups, which are dominated by the
SQLite, libxml2 and Duktape shared objects and their drivers.

#figure(
  table(
    columns: (auto, auto),
    align: (left, right),
    table.header([Metric], [Value]),
    [Program groups],                       [133],
    [Memory-map artefacts (counted)],       [80.07 GiB],
    [`*_sections.csv` mirrors (excluded)],  [3.32 GiB],
    [Raw size, full corpus],                [81.95 GiB],
    [Raw size, tokenised subset],           [53.57 GiB],
    [Memory map / full raw corpus],         [0.98#sym.times],
    [Memory map / tokenised raw],           [1.49#sym.times],
  ),
  caption: [Phase-3 aggregate size: `build_memmap` artefacts vs. raw binaries.],
) <tab:memmap-size>

#figure(
  table(
    columns: (auto, auto),
    align: (left, right),
    table.header([Program group], [Memory-map size]),
    [`libxml2.so.16.1.1`],        [11.86 GiB],
    [`sqlite3`],                  [9.25 GiB],
    [`libsqlite3.51.2.so`],       [7.93 GiB],
    [`libsqlite3.so.3.51.2`],     [7.71 GiB],
    [`m4`],                       [3.26 GiB],
    [`lz4`],                      [3.15 GiB],
    [`libduktaped.so.207.20700`], [3.15 GiB],
    [`libduktape.so.207.20700`],  [3.11 GiB],
  ),
  caption: [Eight largest `build_memmap` program groups by counted artefact size.],
) <tab:memmap-top>

== Discussion

The two stages trade representational convenience against size in opposite
directions. Phase 1 expands each binary roughly two-fold into a token stream
that is human-inspectable and stable across architectures, which is acceptable
because the per-binary outputs are intermediate. Phase 3 then packs the
tokenised corpus back down to within a few percent of the raw input size while
discarding the textual mirrors, so the representation the model ultimately
trains on is no larger than the binaries themselves (0.98#sym.times the full raw
corpus) despite carrying the full vocabulary, section and index structure. The
1.49#sym.times figure against the tokenised subset is the more faithful
overhead measure, since the memory map is built only from the binaries that were
tokenised; the sub-unity ratio against the full corpus reflects that roughly
half of the on-disk binaries lie outside the tokenised set.
