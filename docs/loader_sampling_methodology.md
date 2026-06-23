# Loader & sampling methodology (asm-tokenizer)

Verified against `origin/main` @ commit `852501d` on 2026-06-23.

**Scope note (read first).** The numeric thresholds a methodology chapter wants
to cite — bucket sizes, the section-eligibility counts, the length percentile,
the eligibility band, the per-section variant count — are **training
configuration supplied by the consumer (ml-project), NOT constants in this
repo.** This note documents the *mechanisms* these parameters drive and cites
the canonical source for each; the authoritative *train-time values* must be
read from ml-project's training config. Where a value is genuinely fixed in
code it is stated; where it is a caller parameter that is stated too.

## Section eligibility — `VariantGate`

`tokenizer/aligned_data/sorted_index/_gating.py` — `VariantGate(min_variants,
min_variants_unique)`. A matched-function section passes only if it has at
least `min_variants` total variants **and** at least `min_variants_unique`
*distinct tokenisations*. Both are caller flags (`--min-variants` /
`--min-variants-unique`); the defaults are `0` (no gate). The pair `(8, 6)`
is a configured value, not a code constant.

## Length binning — percentile of token length

`tokenizer/aligned_data/sorted_index/_types.py` — `LengthReduction(kind=
PERCENTILE, percentile=N)`. Sections are binned by the `N`-th percentile of
their token length; `N` is validated to `[1, 99]`. `N=75` ("p75") is a
configured value, not a code constant.

## Cross-(binary × spec) draw and the eligibility band

- `tokenizer/aligned_data/sorted_index/_sampler/_sample.py` —
  `CrossSpecSortedIndexSampler.sample_section_pointers(target_length, count,
  rng, band)`. A hypergeometric urn over the eligible `(binary × length-spec)`
  cells (cells ordered binary-major, spec-minor). `band=(lo, hi)` is a
  **caller-supplied absolute length window with no default**; the draw is over
  sections whose binned length falls in `band`. One draw mixes multiple
  binaries.
- `tokenizer/aligned_data/sorted_index/_collection/_collection.py` —
  `IndexedMemmapCollection.load_batch_cross_depth(...)` is the single primitive
  that runs the urn draw + per-binary decode + alphabetical concat and returns
  a `MultiBinaryBatchDecodeResult` carrying per-row binary identity
  (`binary_id_per_row`, `binary_names`, `depth_per_row`).

A band expressed as fractions of a target length `L` (e.g. `[lo·L, hi·L]`) is
computed *by the caller* before the call; no fractional band factors are fixed
in this repo.

## Variant sampling — without replacement

`tokenizer/aligned_data/loader/_session_helpers.py` —
`generator.choice(n_variants, size=selection_size, replace=False)`. Up to
`num_variants_per_section` variants are drawn per section **without
replacement** (so a section with fewer available variants yields all of them).
`num_variants_per_section` is a caller parameter.

## Crude inlining heuristic — call-target splicing

- `tokenizer/aligned_data/splice_inclusion/_state.py` — at each splice depth a
  columnwise **ALL** over the variant axis flags any callee reached by *every*
  sampled variant; such a callee is treated as inlined-equivalent and **pruned**
  (not spliced, never expanded deeper).
- A callee reached by *some but not all* variants is discriminative and **is**
  included (once-only per root).
- `tokenizer/aligned_data/loader/vector_batch/_inclusion/_compute.py` — a
  **single-variant** section/root splices in **no** call-targets.

## In-process threaded dataloader — `ready_pool`

`tokenizer/aligned_data/loader/ready_pool/` — an in-process, multithreaded
(no multiprocessing) keep-N-ready batch pool with pipelined GPU H2D overlap
over the above. Cross-binary batches come from `make_cross_binary_produce`,
which wraps `load_batch_cross_depth` verbatim (byte-identical to the production
draw, gate-tested).

## One-paragraph summary

Matched-function sections group compiler/optimisation variants of the same
source function. A section is eligible only if it contains at least
`min_variants` total variants and at least `min_variants_unique` distinct
tokenisations. Eligible sections are binned by a percentile (p75 in the
reference config) of their token length; for a target length `L` a batch draws
sections whose binned length falls within a configured band `[lo·L, hi·L]`. The
draw is a hypergeometric urn over the eligible `(binary × length-spec)` cells,
so one batch mixes multiple binaries and each row carries its source-binary
identity. Within each drawn section, up to `num_variants_per_section` variants
are sampled without replacement. Call targets are spliced in under a crude
inlining heuristic: at each depth a callee reached by every sampled variant is
treated as inlined-equivalent and pruned, whereas a callee reached by only some
variants is discriminative and retained; a single-variant section splices in no
call targets.
