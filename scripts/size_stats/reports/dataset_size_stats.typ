// Thesis section: size characterisation of the tokeniser pipeline output,
// across all three tokenised corpora.
//
// Self-contained and compilable on its own (`typst compile dataset_size_stats.typ`),
// but written as a section so it can be dropped into a larger thesis with
// `#include "dataset_size_stats.typ"`. All figures are produced by the
// reporters in `scripts/size_stats/` applied on the LMU gateway (run
// 2026-06-23); the raw per-binary and per-program data live alongside this file
// in `*_phase1_report.csv` and `*_memmap_report.csv`.

#set par(justify: true)
#set table(stroke: 0.5pt + luma(180))
#show table.cell.where(y: 0): strong

= Representation size of the tokenised corpora <sec:size-stats>

This section quantifies how the on-disk size of a binary corpus changes as it
passes through the two size-relevant stages of the pipeline: per-binary
tokenisation (phase 1), which emits a textual CSV token stream per binary, and
the corpus-level memory-map build (phase 3), which packs every tokenised binary
into the binary artefacts the data loader consumes. Two questions motivate the
measurement: how much the textual token representation inflates a binary, and
whether the final packed representation is competitive in size with the raw
inputs it was derived from. The analysis is run over the three tokenised
corpora — the synthetic compiler-matrix `gen-binary-dataset`, the legacy
multi-architecture `Dataset-1`, and the large single-architecture `BinaryCorp`.

== Corpora

@tab:corpora summarises the three corpora. They span very different shapes:
`gen-binary-dataset` is a deep compiler/architecture matrix (nine target
architectures, Clang and GCC, seven optimisation levels) with roughly half its
on-disk binaries lying outside the tokenised set; `Dataset-1` is a smaller
eight-package multi-architecture corpus; `BinaryCorp` is a large flat collection
of x86-64 binaries with essentially complete tokenisation coverage.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, left),
    table.header([Corpus], [Raw binaries], [Tokenised], [Raw size], [Composition]),
    [`gen-binary-dataset`], [167,248], [83,971], [81.95 GiB], [32 pkgs, 9 arch, Clang/GCC],
    [`Dataset-1`],          [6,085],   [5,643],  [12.11 GiB], [8 pkgs, multi-arch],
    [`BinaryCorp` (train+test)], [46,513], [46,511], [24.16 GiB], [x86-64, GCC 11],
  ),
  caption: [The three tokenised corpora. "Tokenised" counts binaries with a
    phase-1 output; "Raw size" is the sum of all raw binary bytes.],
) <tab:corpora>

== Method

Sizes are collected by two standalone reporters (`scripts/size_stats/`). Both
take the raw-binary root and the tokeniser output root as arguments and walk the
trees with a single `find` pass each, so the measurement is reproducible and
free of any pipeline state.

The phase-1 reporter reports an aggregate ratio — the sum of all `_output.csv`
token-stream bytes over the sum of all raw binary bytes — which needs no pairing
and is exact whenever coverage is complete. It additionally pairs each output
with its raw binary, where the layout permits, to recover the per-binary
inflation distribution and a coverage-robust paired ratio. Pairing tries the
output anchor's parent directory and then its suffix-stripped path against the
raw-binary set, which covers both the per-binary-subdir layout
(`gen-binary-dataset`, `Dataset-1` nested) and the flat layout (`Dataset-1`
flat). `BinaryCorp` mangles the binary name between source and output
(truncated build hash, separator change), so it has no structural pairing and is
reported by the aggregate ratio only — exact there because its coverage is
#sym.tilde 100%. Only the canonical `_output.csv` is counted; the metadata
sidecars (`_meta.json`, `_strings.bin`, `_function_ranges.txt`) are excluded.

The phase-3 reporter sums every artefact under `build_memmap/` #emph[except] the
human-readable `*_sections.csv` mirrors (the matched and unmatched section
CSVs), which are verbose textual duplicates of the packed `*_sections.bin`
catalogues and are not part of the consumed representation. Everything else —
the `*.bin` payloads and indices, the sorted `*.idx` files, the variant and
vocabulary tables, and the name/log sidecars — is counted. A
`build_memmap/<program>/` directory is keyed by binary name and merges every
variant of that binary, so the phase-3 figure is a single corpus-level ratio. For
`BinaryCorp`, the non-tokenised subset splits (`small_train`, `small_test`) are
excluded from the raw denominator.

== Phase 1: textual token stream vs. raw binary

The token stream is consistently larger than the binary it describes.
@tab:phase1 reports, per corpus, the inflation factor — the token-stream bytes
over the raw bytes of the binaries that were tokenised. It ranges from
1.39#sym.times (`BinaryCorp`) to 2.46#sym.times (`Dataset-1`). For
`gen-binary-dataset` and `Dataset-1` the per-binary distribution is available:
both are tightly concentrated (medians 2.27#sym.times and 1.98#sym.times), the
byte-weighted aggregate sitting above the median because larger binaries inflate
more.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.header([Corpus], [Raw (tok.)], [Token stream], [Inflation], [Per-binary median]),
    [`gen-binary-dataset`], [53.57 GiB], [110.71 GiB], [2.07#sym.times], [2.27#sym.times],
    [`Dataset-1`],          [12.08 GiB], [29.66 GiB],  [2.46#sym.times], [1.98#sym.times],
    [`BinaryCorp`],         [24.16 GiB], [33.52 GiB],  [1.39#sym.times], [#sym.dash.em],
  ),
  caption: [Phase-1 size: `_output.csv` token streams vs. the raw binaries that
    were tokenised. "Inflation" is the byte-aggregate token-stream / raw ratio;
    `BinaryCorp` has no per-binary pairing (mangled names) but #sym.tilde 100%
    coverage, so its aggregate is exact.],
) <tab:phase1>

== Phase 3: packed memory map vs. raw corpus

The packed representation heavily damps the phase-1 inflation. @tab:memmap
reports the counted `build_memmap` artefact bytes against the raw bytes of the
binaries that are actually present in those memory maps. The ratio spans
1.08#sym.times (`BinaryCorp`) to 1.49#sym.times (`gen-binary-dataset`): the
fully-structured packed corpus — vocabulary, sections, indices and all — is at
most about half again the size of the raw inputs, even though it carries the
complete token-level structure. The excluded `*_sections.csv` mirrors (0.7–3.3
GiB per corpus) would otherwise inflate the on-disk footprint substantially.

A memory-map group is keyed by binary name and merges every optimisation/build
variant of that binary into one set of artefacts, so the group count is far
below the binary count (e.g. `BinaryCorp`'s 9,684 distinct programs from 46,513
variant binaries). The denominator is scoped to exactly the binaries whose
program has a built memory map: `gen-binary-dataset` and `Dataset-1` are fully
built (133/133 and 28/28 programs), whereas `BinaryCorp`'s phase-3 build is
partial — only 821 of 9,684 programs (8.5%) — so its row is the ratio over the
built subset, not the whole corpus.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    table.header([Corpus], [Programs (built)], [Memory map], [`sections.csv` excl.], [Raw (in maps)], [Map / raw]),
    [`gen-binary-dataset`], [133 / 133], [80.07 GiB], [3.32 GiB],   [53.57 GiB], [1.49#sym.times],
    [`Dataset-1`],          [28 / 28],   [17.14 GiB], [1.14 GiB],   [12.11 GiB], [1.42#sym.times],
    [`BinaryCorp`],         [821 / 9,684], [13.24 GiB], [702.15 MiB], [12.25 GiB], [1.08#sym.times],
  ),
  caption: [Phase-3 size: counted `build_memmap` artefacts vs. the raw binaries
    present in those maps. "Programs (built)" is built / total per-binary
    memory-map groups; the raw denominator and ratio cover only the built
    programs. `BinaryCorp`'s build is partial (8.5%).],
) <tab:memmap>

== Discussion

Across all three corpora the two stages trade representational convenience
against size in opposite directions. Phase 1 expands each binary by roughly
1.4–2.5#sym.times into a token stream that is human-inspectable and stable across
architectures, which is acceptable because the per-binary outputs are
intermediate. Phase 3 then packs the tokenised binaries down to 1.08–1.49#sym.times
the size of the raw inputs while discarding the textual mirrors, so the
representation the model ultimately trains on is only modestly larger than the
binaries themselves despite carrying the full vocabulary, section and index
structure. `BinaryCorp` packs the tightest (1.08#sym.times), consistent with its
heavy per-binary variant merging — five optimisation levels collapsed into one
memory map per program. Two caveats on comparability: the phase-3 ratio is
measured against only the binaries actually present in the built memory maps, so
non-tokenised binaries (notably `gen-binary-dataset`'s large untokenised
remainder) do not distort it; and `BinaryCorp`'s phase-3 build is only 8.5%
complete, so its 1.08#sym.times is the figure for the built subset and may shift
as the remaining programs are built.
