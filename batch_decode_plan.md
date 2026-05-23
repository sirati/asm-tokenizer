# Plan — Batch-vectorized v2 dataloader pipeline (4-stage)

## Context

### Current state (what's shipped on `worktree-dataloader` as of this plan)

The v2 dataloader has already been refactored from per-source Python loops to vectorized numpy via the `InlineDecodeState` pre-compute (commits up to `fffb8cb`). Today's pipeline:

- `BinarySession.splice_with_callees(idx, arm, max_depth, max_variants=1, rng=...)` is the per-function entry point.
- It walks the inline-call tree DFS up to `max_depth`, calling `_decode_to_staging(raw_tokens, …, format_version=1)` for the root variant and each inlined callee variant.
- `_decode_to_staging` builds an `InlineDecodeState` once per stream (fields below), uses it for the identity arm + the number arm + the postfix-invariant check, then strips/shifts the post-promotion `working_tokens`.
- The post-strip output is a `DecodedFunction` with `real_tokens: u16` (every original id ≥ 257 shifted down by 256, so the smallest token id is now 1; ids 0 + 256 are stripped — see "Null-content contract" below).
- The splice walker concatenates per-callee `DecodedFunction`s and runs per-Category compaction at the top to produce a single `DecodedFunction` per top-level call.

### What this plan changes

This plan **replaces `splice_with_callees` entirely** with a 4-stage batch pipeline. The model-facing output is a 2D `(batch_size, context_len)` u16 token tensor plus flat-with-offsets sidecars for identities + number chunks. The per-function `DecodedFunction` dataclass is GONE.

Two motivations:

1. **Batch-level vectorization.** When a training batch needs `B` variants from `S` sections, today's path pays `B × per-variant Python overhead` for chunk-decode + identity-decode. Concatenating ALL inline-byte payloads across the batch into ONE shared u8 array, then decoding each TokenType via a single `np.view('>uXX').reshape(-1)` call, collapses that to O(batch) numpy work.

2. **Fix a long-standing identity-decode semantic bug.** Today's `_resolve_fid_payload` maps caller-local function ids directly to `function_name_ptr` (FID). The correct semantics is **per-Category variant-global counter ids**, with FID equality used as the dedup key for the function categories (LOCAL_FUNC, PLT_FUNC, EXT_FUNC) — other Categories renumber by offset across functions. The current behavior would silently surface FIDs (large u32 values) to the model where it expects small counters; this rewrite removes that path entirely.

### Vocab + wire format reference

Constants on `tokenizer.token_manager.VocabularyManager` (the unified vocab at `format_version=1`):

| Constant | Value | Meaning |
|---|---|---|
| `_V2_RESERVED_DIGIT_COUNT` | 256 | Wire-stream digit-vs-metatoken boundary. Inline-byte payload values live in `[0, 256)`. |
| `_V2_VALUE_NEGATIVE_TOKEN_ID` | 256 | Postfix sign marker. Stripped from the model-facing stream. |
| `_V2_RESERVED_TOKEN_COUNT` | 257 | First real-token id after inline digits + sign marker. |
| `_V2_NUMBER_BLOCK_START` | 257 | First NUMBER carrier (VC2). |
| `_V2_NUMBER_BLOCK_COUNT` | 7 | VC2 + F16 + BF16 + F32 + F64 + F80 + F128 in source-declaration order. |
| `_V2_IDENTITY_BLOCK_START` | 264 | First IDENTITY carrier (BLOCK_V2). |
| `_V2_IDENTITY_BLOCK_COUNT` | 8 | BLOCK_V2 + LOCAL_FUNC + PLT_FUNC + EXT_FUNC + STRING_PTR + JUMP_TABLE + RO_DATA_PTR + RW_DATA_PTR (user-canonical, then alphabetical). |
| `_V2_EAGER_BLOCK_END` | 272 | First slot past NUMBER + IDENTITY blocks; instruction-rep ids start here. |

Per-binary v2 vocabs lay these out lazily and the contiguous-range invariants don't hold; the unified vocab guarantees them. **This entire plan assumes `format_version=1` (unified vocab) — every consumer asserts that constraint.**

### Null-content contract (already shipped — official API)

The `_decode_to_staging` output today applies:

- `keep_mask = working_tokens > 256` (strict; drops both inline-byte runs AND `value_negative`).
- `real_tokens = (working_tokens[keep_mask] - 256).astype(np.uint16)` — the model-facing vocab shifts so the smallest produced id is 1 (originally id 257 = `valued_const_v2`); slot 0 is the reserved `[null-content]` id.

The batch pipeline preserves this contract:
- `tokens` row-tails (where a row's content is shorter than `context_len`) are id 0 by construction.
- `tokens == 0` is an unambiguous null-content marker — never emitted from real source content.
- Identity sidecar counter ids do NOT inherit this reservation: counter id 0 in any per-Category space is a legitimate first entry (the sidecar is sized exactly via row offsets, so 0 entries past `offset[i+1]` are never read).

### Inputs available from existing modules

What stage 1 reads from existing code (re-using; don't reimplement):

- `BinarySession` (`tokenizer/aligned_data/loader/_session_splice.py`) — handle resolution + per-arm load:
  - `_load_matched_for_splice(idx, variant_index) -> (FunctionData, Section, section_offset)`
  - `_load_unmatched_for_splice(idx) -> (FunctionData, Section, section_offset)`
  - `_load_matched_section_and_variants(idx) -> (Section, section_offset, MatchedFunction)` — used for per-section variant enumeration.
- `_select_variant_indices(variants, max_variants, rng) -> list[int]` — RNG-sampled subset OR all variants if `max_variants >= len(variants)`. Stays.
- `_build_fids_per_category(section: Section) -> dict[Category, np.ndarray]` — per-category arrays of `function_name_ptr` indexed by caller-local id. Stays (but its OUTPUT is consumed by the new counter-renumbering path, NOT by FID resolution).
- `Section` (`tokenizer/aligned_data/matched_sections_bin.py`):
  - `Section.variants: list[VariantHeader]` — each variant has `variant_ref_offset` (vkey) + `data_offset_shifted` (data.bin offset).
  - `Section.call_targets: list[CallTargetRow]` — encounter-ordered, grouped by `CallTargetType` (LOCAL → PLT → EXT). Each row has `function_name_ptr: u32`, `function_section_ptr: u32` (callee section offset; 0 = unresolved), `type: CallTargetType`, `flags: u8`.
- `FunctionData` (`tokenizer/aligned_data/loader/function_data.py`):
  - `tokens: u16[N]` — raw wire stream (variant_tokens prefix + body).
  - `insn_runlength: u8 or u16[K_insn]` — per-instruction token runlengths.
  - `block_runlength: u8 or u16[K_blocks]` — per-block instruction counts.
  - `variant_tokens: u16[K_var]` — prefix tokens from `_variants.bin` (re-extracted by stage 1 if needed).
  - `metadata: dict[str, Any]` — bag (matched_fn_name, encoder counters per Category, etc.).
- `InlineDecodeState` (`tokenizer/aligned_data/loader/decoded/_inline_decode_state.py`) — `build_inline_decode_state(raw_tokens, *, format_version)` returns:
  - `raw_tokens: u16[N]` — the input stream (NOT a copy).
  - `real_mask: bool[N]` = `raw_tokens > 256` (strict).
  - `number_mask: bool[N]` = `raw_tokens < 256` (inline-digit band).
  - `runlen_number: u16[N]` — `run_lengths(number_mask)`; run-start carries length, others 0.
  - `runlen_value: u16[N]` — `run_lengths(~real_mask)` = numbers + sign.
  - `carries_inline_mask: bool[N]` = `real_mask & (raw_tokens < 272)`.
  - `is_negative_per_position: bool[N]` — True at carriers whose immediate postfix is `value_negative`. Computed via `runlen_value` vs `runlen_number` diff.

The stage-1 walk MUST pass `format_version=1` when building each `InlineDecodeState`.

### Amendment (Phase 0 audit)

Stage1Section + SectionPointerSpec carry `arm: SectionKind` (not `str`); Stage1Section reads `section_offset` from `section.section_offset` rather than carrying a mirror field.

## Locked-in decisions

**D1.** Replace `splice_with_callees` entirely. The new batch entry is the only production path. Synthetic per-function tests rewrite against the batch entry with a 1-section, 1-variant input.

**D2.** Stage-4 truncation is **token-level at exactly `context_len`**, mid-multi-chunk-source allowed AND the side-arrays are truncated identically. If a multi-chunk source (3-chunk VC2 or 2-chunk F128) starts at expanded position `p` and the row's cut falls at `p + j` with `j < chunk_count`:
- Visible tokens for this source = `j` (positions `p` through `p + j − 1`).
- Side-array entries for this source = `j` (one per visible chunk, in stream order).
- The dropped trailing chunks are NOT in the side-array. Information loss is acknowledged — the per-chunk exponent baked into each surviving chunk's `sign_exponent` (`chunk_index_within_source × 64`) tells the model the bit-position of every visible chunk so the model can interpret the partial number without ambiguity.

Side-array entries per row therefore equal the count of number-token positions in the row's `[0, context_len)` post-promotion slice. `number_row_offsets[row+1] − number_row_offsets[row]` is the EXACT per-row chunk count.

**D3.** Self-prepend at function start: every function in the splice tree (root + every inlined callee) gets a synthetic prepended token of the calling category — LOCAL_FUNC for the root and for LOCAL-inlined callees; PLT_FUNC for PLT-inlined callees. The prepended token's identity slot holds the variant-global counter id (0 for the root in its LOCAL_FUNC space; the deduped counter for callees). EXT_FUNC is NOT inlined (no body) — externs only appear as in-stream tokens.

**D4.** Identity values are **variant-global counter ids per Category**, NOT FIDs. The existing `_resolve_fid_payload` path is DELETED.

- **Per-Category counter spaces are fully INDEPENDENT.** A function "foo" appearing as both a LOCAL_FUNC call and a PLT_FUNC call gets DIFFERENT counter ids — one in each space. LOCAL_FUNC's dedup never overlaps with PLT_FUNC's.
- **FUNCTION categories** (LOCAL_FUNC, PLT_FUNC, EXT_FUNC): two caller-local ids whose `call_targets[K].function_name_ptr` match within the SAME Category get the SAME variant-global counter (dedup keyed on `function_name_ptr` within Category).
- **COUNTER categories** (BLOCK_V2, STRING_PTR, RO_DATA_PTR, RW_DATA_PTR, JUMP_TABLE): no dedup — each function's caller-local ids are renumbered by `offset = sum(prev_functions.unique_count_of_category_for(category))`.
- **Counter id 0 in each FUNCTION category is reserved for the root function's self-prepend** (LOCAL_FUNC space only; root is always a LOCAL entity). PLT/EXT_FUNC counter 0 is consumed by encounter order. COUNTER categories don't reserve id 0 — it's just the first counter.
- **Remap-table holes are forbidden.** Variant-global counter ids are dense, hole-free. The hole-free pattern (per ALG-3 below): `mask_remapped` covers already-deduped entries; the unmapped entries get fresh ids via `np.arange(n_fresh) + next_fresh_id`.

**D5.** Output shape. `tokens` id 0 = `[null-content]` is an official, load-bearing API contract:

- `tokens: u16[batch_size, context_len]` — `id == 0` is the null-content padding. `batch_size` depends on the variant-padding policy (D6).
- `identities: u16[total_surviving_identity_tokens]` + `identity_row_offsets: u32[batch_size + 1]` — flat 1D, one entry per identity-token position in row-major iteration. ONE shared array across all identity Categories (Category is recovered from `tokens[row, col]` at the same position). u16 fits per-Category per-variant unique-count comfortably (<65k).
- `numbers_significant: u64[total_surviving_number_chunks]`, `numbers_sign_exponent: u32[total_surviving_number_chunks]`, `number_row_offsets: u32[batch_size + 1]` — shared across ALL number TokenTypes (VC2 + FLOAT16/BF16/32/64/80/128); chunks in stream-position order within each row.
- Optional `fid_sidecar: u32[total_function_counters]` + `fid_row_offsets: u32[batch_size + 1]` — controlled by `include_fid_sidecar` parameter (default `False`). Maps `(LOCAL/PLT/EXT_FUNC, counter_id=K)` → original `function_name_ptr`. Only function categories contribute.

All sidecars are sized EXACTLY to surviving content (D8); `offset[i+1] − offset[i]` gives the exact per-row segment length.

**D6.** Variant-padding policy is a runtime enum:

```python
class VariantPadding(Enum):
    PAD_NULL = "pad_null"               # short sections pad with all-null-content rows (recommended default)
    RESAMPLE_WITHIN_SECTION = "resample" # oversample within section to fill rows
    RAGGED = "ragged"                    # batch_size = total_real_variants; no padding rows
    REDISTRIBUTE = "redistribute"        # take extra samples from sections that had MORE variants than requested
```

**D7.** `max_depth` survives as a STAGE-1 cutoff: at stage 1 the recursion depth is the only thing knowable without decoding token lengths, so it gates which functions get loaded at all. `context_len` is the stage-2 (sidecar pre-size) + stage-4 (cut) cutoff.

**D8.** Stage 1 loads every reachable function (gated by `max_depth`). **Stage 2 is cutoff-aware** — predicts per-function full lengths AND per-function surviving counts (identity tokens + number chunks) AFTER applying the row's `context_len` cutoff. Surviving counts drive sidecar sizing in stage 3. Stage 3 builds idx_2d arrays SIZED to surviving content. Stage 4 is pure assembly + per-Category remap + variant-padding — no cutoff logic remaining.

Stage 2's per-row cutoff walk:
- Walk functions in encounter order (root + each callee from stage 1).
- Accumulate `cumsum_tokens` over each function's predicted-full length.
- Identify the **cut function** as the first whose `cumsum_tokens ≥ context_len`. Functions before: fully included. Cut function: partially included up to `partial_cut_length = context_len − cumsum_tokens_before_it`. Functions after: dropped.
- For each surviving function, count identity-token + number-token positions in the POST-PROMOTION stream within `[0, partial_cut_length)`. Each token = exactly 1 sidecar entry (promotion already expanded multi-chunk sources to consecutive tokens — D2's rule).

**D9.** Each stage produces a **4-level hierarchical dataclass tree**, with the same shape at every stage. The next stage wraps the prior stage's same-level class via composition — so data added in stage N lives at the level it naturally belongs (data shared across the batch lives at level 1; per-call-target data lives at level 4), and a level-N+1 entry at any level reaches into the prior stage via a single `.stageN` back-pointer (no co-indexing of parallel lists).

The 4 levels:

1. **Request** (outermost) — the whole batch. Owns batch-shared arrays (the flat u8 inline buffer, batch-wide row_offsets, per-TokenType `(significand, sign_exp)` arrays).
2. **Section** — one per section pointer in the request. Owns section identity + reference to the `Section` BIN object.
3. **Variant** — one per sampled variant (level-2 children). Owns variant identity + the `batch_idx` this variant occupies in the output.
4. **CallTarget** — one per function in the variant's splice tree. **The root function's body is level-4 entry index 0** (same shape as inlined callees — no special-case dataclass for the root). Owns per-function state (FunctionData, InlineDecodeState, expanded_token_ids, per-call-target slices into level-1 arrays).

Within a Variant's `call_targets` list, the ordering is DFS encounter-order from stage 1's walk: index 0 = root body, indices 1+ = inlined callees in the order stage 1 discovered them.

### Stage 1 — load

```python
@dataclass(frozen=True)
class Stage1CallTarget:
    """Level 4. Per-function loaded state."""
    function_data: FunctionData             # tokens + insn_runlength + block_runlength + variant_tokens + metadata
    state: InlineDecodeState                # masks + runlengths + carries_inline + is_negative
    call_targets_section: list[CallTargetRow]  # THIS function's section.call_targets — used at stage 4 for FID resolution
    encounter_category: Category            # LOCAL_FUNC for root + LOCAL-inlined; PLT_FUNC for PLT-inlined
    parent_call_target_index: Optional[int] # index in the parent's call_targets_section that pointed here; None for root
    function_name_ptr: int                  # this function's global FID

@dataclass(frozen=True)
class Stage1Variant:
    """Level 3. Per-variant load context."""
    variant_idx: int                        # index within the section's variants list
    variant_ref_offset: int                 # vkey for variant identity lookup
    batch_idx: Optional[int]                # row in the output tensor; None if this variant slot is padding-out
    call_targets: list[Stage1CallTarget]    # [root, callee_1, callee_2, ...]; index 0 is the variant body

@dataclass(frozen=True)
class Stage1Section:
    """Level 2. Per-section-pointer context."""
    arm: SectionKind                        # SectionKind.MATCHED or SectionKind.UNMATCHED
    idx: int                                # per-arm function/section idx
    section: Section                        # the BIN's parsed Section; `section.section_offset` is the authoritative BIN-side offset
    variants: list[Stage1Variant]           # one per sampled variant

@dataclass(frozen=True)
class Stage1Batch:
    """Level 1. The whole batch."""
    sections: list[Stage1Section]
    batch_idx_to_section_variant: np.ndarray  # u32[batch_size, 2]; columns (section_idx, variant_idx); padding rows (UINT32_MAX, UINT32_MAX)
    batch_size: int                            # cached len(batch_idx_to_section_variant)
```

### Stage 2 — predict lengths + cutoff

Each level wraps the prior stage's same-level class. Per-call-target data added at level 4; per-variant cutoff result at level 3; per-batch row_offsets at level 1.

```python
@dataclass(frozen=True)
class Stage2CallTarget:
    stage1: Stage1CallTarget
    expanded_token_ids: np.ndarray         # u16[predicted_full_length] — post-promotion, post-strip, post-shift
    extra_value_v2_mask: np.ndarray        # bool[predicted_full_length]
    extra_f128_mask: np.ndarray            # bool[predicted_full_length]
    predicted_full_length: int
    surviving_token_count: int             # ≤ predicted_full_length; equals it when fully included
    surviving_identity_count: int          # in expanded_token_ids[:surviving_token_count]
    surviving_number_chunk_count: int      # in expanded_token_ids[:surviving_token_count]
    is_cut: bool
    partial_cut_length: int                # = surviving_token_count

@dataclass(frozen=True)
class Stage2Variant:
    stage1: Stage1Variant
    call_targets: list[Stage2CallTarget]   # parallel to stage1.call_targets; same indices
    cut_call_target_index: int             # which level-4 entry was cut; len(call_targets) if no cut needed
    total_surviving_token_count: int       # over all call_targets in the variant
    total_surviving_identity_count: int
    total_surviving_number_chunk_count: int

@dataclass(frozen=True)
class Stage2Section:
    stage1: Stage1Section
    variants: list[Stage2Variant]

@dataclass(frozen=True)
class Stage2Batch:
    stage1: Stage1Batch
    sections: list[Stage2Section]
    identity_row_offsets: np.ndarray       # u32[batch_size + 1] — cumsum over batch rows
    number_row_offsets: np.ndarray         # u32[batch_size + 1]
```

### Stage 3 — bulk byte buffer + FP normalization → f96 sidecar

Batch-shared arrays at level 1 (inline_bytes, per-TokenType significand+sign_exp arrays, the identity caller-local array). Each level-4 call_target carries per-call-target SLICES into these level-1 arrays so consumers never compute offsets externally.

```python
@dataclass(frozen=True)
class Stage3CallTarget:
    stage2: Stage2CallTarget
    inline_byte_slice: slice               # this call_target's range in stage3_batch.inline_bytes
    identity_slice: slice                  # range in stage3_batch.identities_flat_caller_local
                                           # — INCLUDES the prepend slot at index identity_slice.start;
                                           # stage 4 writes that slot directly per ALG-9
    # Per-TokenType number chunk slices — keyed by TokenType so stage 4 can pull
    # the right (significand, sign_exp) chunks for assembling the per-row number arrays.
    number_chunk_slices: dict[TokenType, slice]   # per-TokenType ranges in stage3_batch.numbers_<TokenType>_*

@dataclass(frozen=True)
class Stage3Variant:
    stage2: Stage2Variant
    call_targets: list[Stage3CallTarget]   # parallel to stage2.call_targets

@dataclass(frozen=True)
class Stage3Section:
    stage2: Stage2Section
    variants: list[Stage3Variant]

@dataclass(frozen=True)
class Stage3Batch:
    """Level 1. Owns the batch-shared arrays."""
    stage2: Stage2Batch
    sections: list[Stage3Section]
    inline_bytes: np.ndarray                       # u8[total_bytes + 1]; index 0 is the leading zero pad
    identities_flat_caller_local: np.ndarray       # u16[total_surviving_identity_tokens] — pre-remap; stage 4 mutates IN PLACE
    # Per-TokenType normalized number arrays (ALG-7 output):
    numbers_per_TokenType: dict[TokenType, tuple[np.ndarray, np.ndarray]]
        # key: TokenType (VC2, F16, BF16, F32, F64, F80, F128)
        # value: (significand: u64[n_chunks_of_type], sign_exp: u32[n_chunks_of_type])
    # Intermediate diagnostic arrays — useful for tests + the optional intermediate output:
    identity_idx_2d: np.ndarray                    # u32[total_surviving_identity_tokens, 2]
    number_idx_2d_per_TokenType: dict[TokenType, np.ndarray]
    vc2_chunk_exponent_sidecar: np.ndarray         # u32[total_vc2_chunks]; per-chunk exponent base index
```

### Stage 4 — final tensor + remap + variant padding

Stage 4 is the only stage that produces a non-hierarchical output (the model-facing flat tensors). The hierarchical Stage3Batch may optionally be carried alongside via `keep_intermediate=True`.

```python
@dataclass(frozen=True)
class BatchDecodeResult:
    """Level 1 output. The user-facing flat tensors + sidecar offsets."""
    tokens: np.ndarray                              # u16[batch_size, context_len]
    identities: np.ndarray                          # u16[total_surviving_identity_tokens] — POST per-Category remap
    identity_row_offsets: np.ndarray                # u32[batch_size + 1]
    numbers_significant: np.ndarray                 # u64[total_surviving_number_chunks]
    numbers_sign_exponent: np.ndarray               # u32[total_surviving_number_chunks]
    number_row_offsets: np.ndarray                  # u32[batch_size + 1]
    batch_idx_to_section_variant: np.ndarray        # u32[batch_size, 2]
    fid_sidecar: Optional[np.ndarray]               # u32; only when include_fid_sidecar=True
    fid_row_offsets: Optional[np.ndarray]           # u32[batch_size + 1]; only when include_fid_sidecar=True
    intermediate: Optional[Stage3Batch]             # only when keep_intermediate=True (default False)
```

### Consumer navigation

To enumerate every (section, variant, call_target) in stage N:

```python
for section in stageN_batch.sections:
    for variant in section.variants:
        if variant.stage1.batch_idx is None:  # walk stage1 for the level-3 identifier
            continue  # padding row — skip
        for call_target_idx, call_target in enumerate(variant.call_targets):
            is_root = (call_target_idx == 0)
            # Reach prior-stage data via the back-pointer chain:
            #   stage2 data: call_target.stage2 (for a Stage3CallTarget) or call_target (for a Stage2CallTarget)
            #   stage1 data: call_target.stage2.stage1 (for Stage3) or call_target.stage1 (for Stage2)
            #   level-3 (variant) data: navigate up via the outer for-loop's `variant`.
            ...
```

NO parallel-list co-indexing: each level-N CallTarget chain navigates entirely through composition; the outer level-2/level-3 loops provide section/variant identity.

**D10.** Batch ordering is **linear by `batch_idx`**, with `batch_idx → (section_idx, variant_idx)` determined by the variant-padding policy. NOT generally `section_idx * num_variants + variant_idx`:

- `PAD_NULL`: `batch_idx = section_idx * num_variants + variant_idx`; padding rows fill missing slots, mapped to sentinel `(UINT32_MAX, UINT32_MAX)`.
- `RESAMPLE_WITHIN_SECTION`: same linear shape; missing slots oversampled from this section.
- `RAGGED`: dense over actual variants only; `batch_size = total_real_variants`.
- `REDISTRIBUTE`: sections with MORE variants donate extras to sections with FEWER.

The mapping is computed ONCE at stage 1's top (post-sampling, pre-loading) and threaded through every subsequent stage. Within each batch member, function order is `function_idx_in_splice` (root first, then callees in stage-1 encounter order).

## Algorithms

Numbered numpy specs filling holes left by the conversational design. Each is callable as-is.

### ALG-1. Inline-byte concat via narrowing assignment

The user-noted insight: assigning a u16 numpy array into a u8 destination automatically truncates the high byte. For inline-number values the high byte is 0 (wire ids 0..255), so the truncation is lossless. Boolean indexing on `raw_tokens` returns a fresh copy already, so the source isn't writing into the backing memmap.

```python
# Per function, lifting SURVIVING inline bytes into the shared u8 array:
inline_bytes[function_byte_offset : function_byte_offset + K] = (
    function.loaded.state.raw_tokens[function.loaded.state.number_mask]
)
# K = number of surviving inline bytes for this function (predicted in stage 2).
# For the cut function, slice to the surviving prefix BEFORE assignment.
# Numpy 1.x narrowing is implicit. Numpy 2.x may require an explicit .astype(np.uint8)
# on the source side (it's still a cheap copy; the bool-index already copied once).
```

### ALG-2. F128 finite-vs-NaN/Inf detection at stage 2

Vectorized across all F128 sources within a function. IEEE-754 binary128 exponent occupies bits 112..126 (15 bits, big-endian). The detection requires the FULL HIGH U16 of the payload — not just the high byte — because the all-ones exponent pattern spans 15 bits = byte 0's low 7 bits (after the sign) PLUS all 8 bits of byte 1. Masking the high u16 with `0x7FFF` strips the sign bit and leaves exactly the 15-bit exponent for the all-ones comparison.

```python
raw_tokens = function.loaded.state.raw_tokens
f128_carrier_mask = (
    function.loaded.state.real_mask & (raw_tokens == FLOAT128_VOCAB_ID)
)
f128_source_positions = np.nonzero(f128_carrier_mask)[0]
if f128_source_positions.size > 0:
    # Codec precondition: every F128 carrier has 16 bytes of payload at p+1..p+16.
    # Bounds-check the tail (raise on stream corruption):
    if (f128_source_positions + 2 >= raw_tokens.shape[0]).any():
        raise AssertionError(
            "F128 carrier within 1 position of stream tail — malformed v2 stream"
        )
    # Two u16 inline bytes ARE the big-endian byte pair (high byte = 0 in each u16):
    high_bytes = raw_tokens[f128_source_positions + 1].astype(np.uint16) << np.uint16(8)
    low_bytes  = raw_tokens[f128_source_positions + 2].astype(np.uint16)
    high_u16 = high_bytes | low_bytes
    is_nan_or_inf = (high_u16 & np.uint16(0x7FFF)) == np.uint16(0x7FFF)
    f128_chunk_counts = np.where(is_nan_or_inf, np.uint8(1), np.uint8(2))
else:
    f128_chunk_counts = np.empty(0, dtype=np.uint8)
```

### ALG-3. Hole-free remap (FUNCTION categories — LOCAL_FUNC / PLT_FUNC / EXT_FUNC)

Per row, per FUNCTION Category. **Backed by a new Rust hashmap `HashMapU32U16` in `dedup_hashmap/src/lib.rs`, paralleling the existing `HashMapU64U32` + `HashMapU32U32`.** What's NEW (and missing in the existing siblings) is a **numpy-batched lookup API**: a single FFI call that takes `ndarray[u32]` keys and returns `ndarray[u16]` values with `0xFFFF` as the "not found" sentinel. This avoids the Python-dict + per-element loop that would otherwise bottleneck the dedup at training-step scale (millions of lookups per step).

**Required Rust changes — generic macro spanning the full hashmap product:**

Instead of adding `HashMapU32U16` ad-hoc, replace today's two hand-rolled hashmaps (`HashMapU64U32`, `HashMapU32U32`) with a single macro-generated set spanning the full Cartesian product:

- **Key types**: `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64` (9 variants).
- **Value types**: same 9 integer types **plus** `f32` + `f64` (11 variants).
- Class names follow the existing pattern: `HashMap<K><V>` e.g. `HashMapU32U16`, `HashMapI64F64`, etc.

The macro generates each PyO3 class with the same surface:

- `__init__(capacity: int)` — preallocate buckets (replaces the current zero-arg constructor; we want to avoid rehashing during a hot dedup walk).
- `insert(key, value)` — single-entry insert.
- `lookup(key) -> Optional[value]` — single-entry lookup (returns None or a sentinel; see below).
- `lookup_ndarray(keys: NDArray[K]) -> NDArray[V]` — vectorized lookup. Returns the type-appropriate "not found" sentinel (`0xFFFF...` of the value width for unsigned ints; `<dtype>.min` for signed; NaN for floats; False for bool). Document the sentinel per dtype.
- `insert_ndarray(keys: NDArray[K], values: NDArray[V])` — batch insert; later calls overwrite for duplicate keys.
- `clean()` — reset all entries WITHOUT deallocating the backing storage. Critical for the dedup walk: each row needs a fresh hashmap, but reallocating on every row would dominate the cost. `clean()` keeps the capacity allocation; only the entry map is reset.
- `__len__()` — entry count.
- `__contains__(key)` — single-entry membership.

The macro lives in a new `dedup_hashmap/src/hashmap_macro.rs` (or extends the existing `lib.rs`). Each generated class registers itself in the `#[pymodule]` block.

Migration: the call sites that use `HashMapU64U32` / `HashMapU32U32` today (the SectionWriter refactor) continue to work — the macro generates classes with the same names. The plan adds `HashMapU32U16` for stage 4's dedup walk (one per FUNCTION Category × one per batch row, reused via `clean()` across rows).

```python
from dedup_hashmap import HashMapU32U16

NOT_FOUND = np.uint16(0xFFFF)

# Per row, per FUNCTION Category — initial state:
#   dedup_map: HashMapU32U16 (one per Category)
#   next_fresh_id: int
#
# LOCAL_FUNC: dedup_map.insert(root.function_name_ptr, 0) BEFORE walking the root body;
#             next_fresh_id = 1. (Root's self-prepend reserves counter 0; self-recursion
#             in the root body dedupes via the same map below.)
# PLT_FUNC, EXT_FUNC: dedup_map empty; next_fresh_id = 0.

# Per call_target in encounter order, per FUNCTION Category:
# Build the caller-local-id → function_name_ptr array directly from the section header —
# NO Python dict, NO co-iteration with another structure.
fn_name_ptrs = call_target.stage2.stage1.call_targets_section_for_category(category)
# fn_name_ptrs: np.ndarray[np.uint32], shape (K,)
K = int(fn_name_ptrs.size)
if K == 0:
    continue   # no in-stream tokens of this Category in this call_target

# Single FFI call: ndarray-batched lookup; misses come back as 0xFFFF.
remap_lookup = dedup_map.lookup_ndarray(fn_name_ptrs)   # u16[K]
mask_remapped = (remap_lookup != NOT_FOUND)

# Fresh ids for the misses, dense from next_fresh_id (hole-free per D4):
n_fresh = int((~mask_remapped).sum())
if n_fresh > 0:
    fresh_ids = np.arange(n_fresh, dtype=np.uint16) + np.uint16(next_fresh_id)
    remap_lookup[~mask_remapped] = fresh_ids
    # Single FFI call: batch-insert so subsequent call_targets in this row see the new entries.
    dedup_map.insert_ndarray(fn_name_ptrs[~mask_remapped], fresh_ids)
    next_fresh_id += n_fresh

# Apply remap_lookup to this call_target's slice of identities_flat_caller_local,
# masked to slots whose token id matches this Category's id (post-shift = original − 256).
sl = call_target.identity_slice   # Stage3CallTarget.identity_slice
in_stream_sl = slice(sl.start + 1, sl.stop)   # skip the prepend slot (written by ALG-9)
# The token ids parallel to in_stream_sl come from the surviving in-stream identity-token
# positions in expanded_token_ids[:partial_cut_length]:
surviving = call_target.stage2.expanded_token_ids[: call_target.stage2.partial_cut_length]
in_stream_identity_mask = (surviving >= 8) & (surviving < 16)
in_stream_token_ids = surviving[in_stream_identity_mask][1:]   # drop the prepend's row
cat_mask = (in_stream_token_ids == category_token_id_shifted)
if cat_mask.any():
    target_slice = stage3_batch.identities_flat_caller_local[in_stream_sl]
    target_slice[cat_mask] = remap_lookup[target_slice[cat_mask]]
```

`call_targets_section_for_category(category)` is a thin accessor on the section header: `call_targets` rows are already grouped LOCAL → PLT → EXT in the BIN; the accessor returns the `function_name_ptr` column for rows whose `.type` matches `category`. Same source-of-truth as today's `_build_fids_per_category` — no new BIN parsing required.

### ALG-4. COUNTER-category renumbering — pure offset, no remap_lookup

For BLOCK_V2 / STRING_PTR / RO_DATA_PTR / RW_DATA_PTR / JUMP_TABLE. Caller-local ids start at 0 within each function; bump by the running per-Category offset.

```python
# Per row (= per Stage3Variant), per COUNTER Category:
offset = 0  # counter id 0 is a legitimate first entry; no reservation

for call_target in stage3_variant.call_targets:
    if call_target.stage2.surviving_token_count == 0:
        continue   # dropped due to context_len cut
    sl = call_target.identity_slice
    in_stream_sl = slice(sl.start + 1, sl.stop)  # skip prepend slot
    # Token ids parallel to in_stream_sl come from the SURVIVING in-stream identity tokens
    # in call_target.stage2.expanded_token_ids[:partial_cut_length]:
    surviving_expanded = call_target.stage2.expanded_token_ids[
        : call_target.stage2.partial_cut_length
    ]
    in_stream_identity_mask = (
        (surviving_expanded >= 8) & (surviving_expanded < 16)
    )
    in_stream_token_ids = surviving_expanded[in_stream_identity_mask][1:]  # drop prepend's row
    cat_mask = (in_stream_token_ids == category_token_id_shifted)
    if cat_mask.any():
        stage3_batch.identities_flat_caller_local[in_stream_sl][cat_mask] += np.uint16(offset)
    # Per-call_target unique count for this COUNTER Category, sourced from FunctionData.metadata
    # (the encoder tracks per-function counter maxima during emission and writes them to the BIN).
    offset += call_target.stage2.stage1.function_data.metadata["category_counts"][category]
```

### ALG-5. Identity decode — single u16 idx_2d for ALL identity Categories

```python
# inline_bytes: u8[total_bytes + 1]; index 0 is the leading zero pad.
# identity_idx_2d: u32[total_surviving_in_stream_identity_tokens, 2]
#   Each row holds 2 byte offsets into inline_bytes (big-endian u16 caller-local id).
#   - 2-byte payloads:   [hi_offset, lo_offset]
#   - 1-byte payloads:   [0, lo_offset]   (leading-zero pad supplies the high byte)
#   - 0-byte payloads:   [0, 0]           (read as u16 0; encoder reserves caller-local 0 for this)
# PREPEND SLOTS ARE NOT IN identity_idx_2d (stage 4 writes them directly — see ALG-9).

gathered = inline_bytes[identity_idx_2d]                          # u8[N, 2] — fresh contiguous copy
identities_flat_caller_local = gathered.view('>u2').reshape(-1)   # u16[N]
# The view-cast produces caller-local ids in stream order.
# Stage 4 applies per-Category remap_lookup IN PLACE (no second copy):
#   identities_flat_caller_local[per_category_slice_mask] = remap_lookup[...]
# This same array is returned to the consumer as BatchDecodeResult.identities after
# stage 4 fills the prepend slots and applies all per-Category remaps.
```

### ALG-6. Per-Category mask within the flat identity output

```python
# expanded_token_ids: post-promotion, post-shift token ids for one function.
# tokens_mask_all_with_identity: which positions hold an identity token.
tokens_mask_all_with_identity = (
    (expanded_token_ids >= IDENTITY_BLOCK_START_SHIFTED)
    & (expanded_token_ids < IDENTITY_BLOCK_END_SHIFTED)
)
# IDENTITY_BLOCK_START_SHIFTED = 264 - 256 = 8   (BLOCK_V2 in shifted vocab)
# IDENTITY_BLOCK_END_SHIFTED   = 272 - 256 = 16

# Per-Category sub-mask over identity_token_ids (NOT over the full token tensor):
identity_token_ids = expanded_token_ids[tokens_mask_all_with_identity]
per_category_mask = (identity_token_ids == specific_category_token_id_shifted)
# specific_category_token_id_shifted:
#   BLOCK_V2 = 8, LOCAL_FUNC = 9, PLT_FUNC = 10, EXT_FUNC = 11,
#   STRING_PTR = 12, JUMP_TABLE = 13, RO_DATA_PTR = 14, RW_DATA_PTR = 15
# (User-canonical, then alphabetical order matches the unified vocab's IDENTITY block.)
```

### ALG-7. Number-arm decode — view-cast THEN vectorized FP normalization to the "f96" sidecar shape

Every number TokenType — VC2, F16, BF16, F32, F64, F80, F128 — decodes through this same pipeline:
1. Gather payload bytes via `idx_2d`.
2. View-cast to the type's bit-pattern dtype (u16/u32/u64).
3. **Vectorized per-TokenType normalization**, producing `(u64 significand, u32 sign_exponent)` per chunk — the canonical "f96" sidecar shape used uniformly across ALL number TokenTypes (existing `pack_sign_exp` convention on `custom_float.py`: sign bit at MSB bit 31; biased exponent in lower bits; the significand normalizes the leading 1 to bit 63, "without integer bit").

**Denormal renormalization is part of the per-TokenType normalization step.** IEEE-754 denormals (biased exponent == 0 AND mantissa != 0) DO appear in the source bit patterns. The f96 sidecar shape is wider than any source format (u64 mantissa + 15-ish exponent bits) so denormals can ALWAYS be re-expressed as a true normalized value (leading 1 at bit 63 of the u64, with the unbiased exponent adjusted downward). The vectorized normalization branch detects denormals via `(biased_exp == 0) & (mantissa != 0)`, renormalizes via `np.clz`-equivalent leading-zero count, and emits the same shape as normal values — the model never sees denormal-form numbers.

This is a VECTORIZED rewrite of `custom_float.py`'s per-source `from_float16` / `from_bfloat16` / `from_float32` / `from_float64` / `from_float80` / `from_float128` / `from_int` functions — same formulae (including the denormal branch + NaN/Inf branch already present per-source), batch-applied via numpy. The existing per-source functions in `custom_float.py` are kept as the Python-loop oracle for byte-equivalence tests, but stage 3 NEVER calls them per-source.

Per TokenType, stage 3 produces `(per_type_significand, per_type_sign_exponent)` arrays of shape `(n_chunks_of_type,)`. Stage 4 concatenates them in row-order into the global `numbers_significant` + `numbers_sign_exponent`.

```python
# ============================================================
# F16 / BF16 — 2-byte payloads → u16 bit pattern → normalize.
# ============================================================
f16_bits  = inline_bytes[f16_idx_2d].view('>u2').reshape(-1)   # u16[n_f16]
bf16_bits = inline_bytes[bf16_idx_2d].view('>u2').reshape(-1)  # u16[n_bf16]
# Per-type vectorized renormalize: extract sign+exponent+mantissa bit fields, handle
# subnormal + NaN/Inf branches via np.where, output (sig: u64, sign_exp: u32).
# Formulae: same as from_float16 / from_bfloat16 in custom_float.py, vectorized.

# ============================================================
# F32 — 4-byte payloads → u32 → normalize.
# ============================================================
f32_bits = inline_bytes[f32_idx_2d].view('>u4').reshape(-1)    # u32[n_f32]
# Same shape as f16 batch path.

# ============================================================
# F64 — 8-byte payloads → u64 → normalize.
# ============================================================
f64_bits = inline_bytes[f64_idx_2d].view('>u8').reshape(-1)    # u64[n_f64]

# ============================================================
# F80 — 10-byte payloads → 5 u16 limbs (big-endian).
# Reassembled via shift+OR into (sign_bit: u8, exponent_bits: u16, mantissa_u64: u64).
# F80 has an EXPLICIT integer leading bit in the mantissa — strip it during normalization.
# ============================================================
f80_limbs = inline_bytes[f80_idx_2d].view('>u2')               # u16[n_f80, 5]
# limb 0 = bytes 0..1 = sign + exponent (15 bits)
# limbs 1..4 = bytes 2..9 = 64-bit mantissa (big-endian)
# Reassemble mantissa as u64:
f80_sign_exp_word = f80_limbs[:, 0].astype(np.uint16)
f80_mantissa_u64 = (
    (f80_limbs[:, 1].astype(np.uint64) << np.uint64(48))
    | (f80_limbs[:, 2].astype(np.uint64) << np.uint64(32))
    | (f80_limbs[:, 3].astype(np.uint64) << np.uint64(16))
    |  f80_limbs[:, 4].astype(np.uint64)
)
# Strip the explicit integer bit (bit 63 of mantissa per x87 convention) + normalize
# the remaining bits so the new leading 1 sits at bit 63. NaN/Inf branch when
# (f80_sign_exp_word & 0x7FFF) == 0x7FFF. Vectorized via np.where.

# ============================================================
# F128 — 16-byte payloads → 2 u64 limbs (big-endian).
# Finite: emit 2 chunks (one per u64 limb, with chunk-index exponent base).
# NaN/Inf: emit 1 chunk (encoded via _encode_infnan formula).
# ============================================================
f128_limbs = inline_bytes[f128_idx_2d_per_source].view('>u8')  # u64[n_f128_sources, 2]
# is_nan_or_inf already computed in stage 2 via ALG-2.
# Finite branch: for each source, emit 2 chunks with chunk_index = 0 (low) + 1 (high);
# normalize each chunk's u64 to the "leading-1-at-bit-63" form; pack into the global
# f128 chunk arrays.
# NaN/Inf branch: emit 1 chunk via _encode_infnan(sign, mantissa_is_zero).

# ============================================================
# VC2 — variable-length payloads → K u64 chunks per source (ALG-8 byte layout).
# Each chunk is an INTEGER magnitude limb (not an FP bit pattern); normalize so the
# leading 1 sits at bit 63 of the chunk's u64, capture the shift into the chunk's
# exponent_unbiased = chunk_index * 64 + leading_one_bit_offset.
# Zero chunks: pack_sign_exp(sign, 0) — canonical signed-zero shape.
# ============================================================
vc2_chunk_u64 = inline_bytes[vc2_idx_2d].view('>u8').reshape(-1)  # u64[total_vc2_chunks]
# Normalize per chunk: leading-1 position via 63 - np.uint64(np.floor(np.log2(vc2_chunk_u64)))
# (with branch for chunk == 0 to emit signed zero). Vectorized via np.where +
# np.left_shift, or via the per-chunk loop if numpy intrinsics aren't enough.
```

Outputs after ALG-7 (stage 3 product):
```python
# Per TokenType, both arrays parallel + in stream order:
f16_significand:  np.ndarray   # u64[n_f16]
f16_sign_exp:     np.ndarray   # u32[n_f16]
bf16_significand: ...
bf16_sign_exp:    ...
# ... (same shape for f32, f64, f80, f128, vc2)
```

Edge cases:
- **Short final VC2 chunk** (17-byte payload → chunks 0+1 fill 16 bytes; chunk 2 has 1 byte): chunk-2's idx_2d row left-pads with 7 references to `inline_bytes[0]` then 1 to the actual byte. View-cast → u64 with low 8 bits set, high 56 bits zero. Normalization sees a small magnitude with leading 1 in the low bits, produces (sig with leading-1-at-bit-63, exponent = chunk_index * 64 + small_offset).
- **F80 / F128 stride**: fancy-index gather produces C-contiguous u8; `.view('>u2')` / `.view('>u8')` is safe because the row stride equals the row size.
- **F128 NaN/Inf vs finite source-row layout**: the per-source chunk count from ALG-2 drives the per-source contribution to `f128_idx_2d` (1 row for NaN/Inf, 2 for finite). A separate parallel array `f128_is_nan_or_inf: bool[n_f128_sources]` is carried into stage 3 so the normalization step branches correctly per source.

### ALG-8. VC2 multi-chunk byte layout

`_split_to_chunks` (existing in `decoded/custom_float.py`) emits chunks low-to-high: chunk 0 = LSB = trailing bytes of the big-endian payload; chunk K−1 = MSB = leading bytes (possibly short).

```python
# Per VC2 source at carrier position p_carrier (raw_tokens position):
#   p_carrier_byte: source's first inline-byte offset in inline_bytes (after the leading pad).
#   L            : inline_len = state.runlen_number[p_carrier + 1].
#   K            : max(1, ceil(L / 8))  # chunk count.

# Per chunk c (0 = LSB chunk, K-1 = MSB chunk):
#   Payload bytes: [p_carrier_byte + L - 8*(c+1), p_carrier_byte + L - 8*c)
#     intersected with [p_carrier_byte, p_carrier_byte + L).
#   The MSB chunk may have fewer than 8 payload bytes when L % 8 != 0.

# Concrete example for L = 17 (so K = 3):
#   chunk 0 (LSB): inline_bytes[p+9 .. p+17)   — 8 bytes, no pad
#   chunk 1:        inline_bytes[p+1 ..  p+9)   — 8 bytes, no pad
#   chunk 2 (MSB): inline_bytes[p   ..  p+1)   — 1 byte, idx_2d row = [0, 0, 0, 0, 0, 0, 0, p]

# Per-chunk exponent index sidecar (one entry per VC2 chunk in global stream order):
vc2_chunk_exponent_sidecar[chunk_global_idx] = chunk_c_within_source   # 0..K-1

# Stage 4 reconstructs numbers_sign_exponent per chunk:
#   exponent_base = chunk_index_within_source * 64
#   exponent_unbiased = exponent_base + leading_one_position_within_u64(chunk_value)
#   sign_bit = state.is_negative_per_position[source_position]  # same sign for all chunks
#   numbers_sign_exponent[chunk_global_idx] = pack_sign_exp(sign_bit, exponent_unbiased)
#
# All-zero chunk optimization: when chunk_value == 0, the exponent_base doesn't affect the
# numeric value, so emit pack_sign_exp(sign, 0) — the canonical signed-zero shape.

# Per-source per-chunk byte-offset arithmetic (small numpy op):
chunk_byte_starts = (
    p_carrier_byte + L - 8 * np.arange(K, 0, -1, dtype=np.int64)
)  # length K; chunk_byte_starts[c] is the source-relative start for chunk c.
# Negative entries indicate left-pad; the MSB chunk has at most (8 - L % 8) leading pads.
```

### ALG-9. Prepend slot handling — stage 4 writes directly

Synthetic-bytes-in-idx_2d would tangle semantics (the prepend's caller-local id lives in the parent's call_targets space, not the current function's, so applying the current function's remap_lookup would be wrong). Cleanest split: prepend slots live in `function_identity_offsets`'s per-function range (always the FIRST slot) but are NOT in `identity_idx_2d`. Stage 4 writes them during the per-function dedup walk.

```python
# Stage 4, per function F in encounter order:
calling_category = F.loaded.encounter_category   # LOCAL_FUNC or PLT_FUNC
# self_counter is exactly what dedup_dict_per_category[calling_category] holds for F's FID.
# - Root: LOCAL_FUNC, self_counter = 0 (seeded at row start).
# - Inlined LOCAL callee: self_counter = dedup_dict[LOCAL_FUNC][F.loaded.function_name_ptr]
#   (set during the parent's ALG-3 dedup walk when F's FID was first encountered).
# - Inlined PLT callee: self_counter = dedup_dict[PLT_FUNC][F.loaded.function_name_ptr].
self_counter = dedup_dict_per_category[calling_category][F.loaded.function_name_ptr]

# Write to the prepend slot at the start of F's identity slice:
prepend_slot_offset = decoded_batch.function_identity_offsets[global_function_idx]
decoded_batch.identities_flat_caller_local[prepend_slot_offset] = np.uint16(self_counter)

# Write the prepend's token id to the row's tokens tensor (shifted = original − 256):
# LOCAL_FUNC_SHIFTED = 265 - 256 = 9; PLT_FUNC_SHIFTED = 266 - 256 = 10
tokens[row_idx, prepend_token_position] = np.uint16(
    LOCAL_FUNC_SHIFTED if calling_category is LOCAL_FUNC else PLT_FUNC_SHIFTED
)
```

Stage 3 still allocates the slot in `function_identity_offsets` (each function's range starts with 1 prepend slot + the count of surviving in-stream identity tokens) — it just doesn't populate it via view-cast.

### ALG-10. batch_idx ordering — policy-aware mapping

```python
# Computed ONCE at the top of stage 1, AFTER per-section variant sampling, BEFORE loading.
# Output: batch_idx_to_section_variant: u32[batch_size, 2] columns (section_idx, variant_idx).
# Sentinel: padding rows hold (UINT32_MAX, UINT32_MAX).
# Stages 1..4 iterate in batch_idx order; per-row work indexes through this mapping.

# Per policy:
#   PAD_NULL:        batch_size = num_sections * num_variants;
#                    mapping[s * nv + v] = (s, v) when s has at least v+1 variants;
#                                          (UINT32_MAX, UINT32_MAX) otherwise.
#   RESAMPLE_WITHIN_SECTION: batch_size = num_sections * num_variants;
#                    missing slots filled by random sampling with replacement from
#                    this section's available variants.
#   RAGGED:          batch_size = total_real_variants;
#                    mapping is dense over the sampled variants only.
#   REDISTRIBUTE:    batch_size = num_sections * num_variants;
#                    sections with MORE variants donate extras to sections with FEWER,
#                    so the final mapping has no padding rows (but section_id_per_row
#                    is non-uniform across the linear layout).

# DO NOT assume batch_idx == section_idx * num_variants + variant_idx.
```

## Stages — algorithm sketch

### Stage 1: section walk + raw-data load

Input: a list of `(arm, idx)` section pointers + `num_variants_per_section`, `max_depth`, `inlined_equivalent_call_targets_only`, `rng`, `variant_padding` policy.

1. **Top-level mapping**: build `batch_idx_to_section_variant: u32[batch_size, 2]` per ALG-10's policy table. This determines which `(section_idx, variant_idx)` pairs (if any) populate each row of the final output.
2. **Per section pointer (level 2)**:
   - Resolve via `BinarySession._load_matched_section_and_variants` (matched) or `_load_unmatched_record_and_section` (unmatched). Produces `Section` + `section_offset`.
   - Sample variant indices via `_select_variant_indices(section.variants, max_variants=num_variants_per_section, rng=rng)`.
3. **Per sampled variant (level 3)**:
   - Compute `batch_idx` from the reverse mapping built in step 1 (None if this variant slot was not selected into the linear layout, e.g. under RAGGED some sections contribute fewer rows).
   - Load the root function's `FunctionData` (the variant body); build `InlineDecodeState(format_version=1)`.
   - The root becomes level-4 entry index 0 in this variant's `call_targets` list.
4. **Per recursive call target (level 4)**:
   - Recurse DFS up to `max_depth`:
     - For each `call_target` in the parent's `call_targets_section` (encounter order): if `function_section_ptr` resolves AND not in active visited set, load FunctionData + build InlineDecodeState + append as a new `Stage1CallTarget` to this variant's level-4 list.
     - Visited set on `(arm, section.section_offset)`; remove on backtrack (DAG semantics).
     - Apply `inlined_equivalent_call_targets_only` filter when true: skip callees where ALL or NONE of the parent's variants called this target (only inline when SOME but not ALL did).
   - Each level-4 entry records `parent_call_target_index` (the parent's call_targets_section row index that pointed here; `None` for the root) and `encounter_category` (`LOCAL_FUNC` for root + LOCAL-inlined; `PLT_FUNC` for PLT-inlined; EXT_FUNC NOT inlined per D3).

Output: `Stage1Batch` (D9). The 4-level hierarchy is fully populated; no flat per-function lists at the batch level.

### Stage 2: length + sidecar-size prediction + cutoff walk

Operates over `Stage1Batch`. Produces `Stage2Batch` (D9) with the 4-level mirror.

For each level-4 call target (across the whole batch — sections → variants → call_targets, in DFS order):

1. Build `expanded_token_ids` by promoting multi-chunk sources in a working copy of `stage1_call_target.state.raw_tokens` then stripping + shifting:
   - Identify VC2 sources: positions with `raw_tokens[p] == VC2_VOCAB_ID & state.real_mask[p]`. For each, `chunk_count = max(1, ceil(state.runlen_number[p+1] / 8))`; paint `working_tokens[p+1 : p+chunk_count] = VC2_VOCAB_ID`.
   - Identify F128 sources: via ALG-2; for each finite source, `chunk_count = 2`; paint `working_tokens[p+1] = FLOAT128_VOCAB_ID`. NaN/Inf sources have chunk_count = 1, no painting.
   - Strip + shift: `expanded_real = (working_tokens[working_tokens > 256] - 256).astype(np.uint16)`.
   - Prepend self-token: `expanded_token_ids = np.concatenate([[calling_category_token_id_shifted], expanded_real])` where the calling category comes from `stage1_call_target.encounter_category`.
2. Build `extra_value_v2_mask` + `extra_f128_mask` (bool arrays of shape `predicted_full_length`) marking promoted-slot positions in `expanded_token_ids`.
3. `predicted_full_length = expanded_token_ids.shape[0]`.

Per level-3 variant (in section-then-variant iteration order):

4. Cumsum `predicted_full_length` over the variant's `call_targets` (root + each inlined callee, in stage-1 encounter order). Find the cut call target index — first whose cumsum ≥ `context_len`. `partial_cut_length` for cut entry = `context_len − cumsum_before`; entries after are dropped (counts = 0); entries before are fully included.
5. For each surviving call target compute `surviving_identity_count` and `surviving_number_chunk_count` via masks on `expanded_token_ids[:partial_cut_length]` (or the full length for fully-included call targets):
   ```python
   identity_mask = (expanded_token_ids >= 8) & (expanded_token_ids < 16)   # IDENTITY block shifted
   number_mask   = (expanded_token_ids >= 1) & (expanded_token_ids < 8)    # NUMBER block shifted (VC2=1, F16=2, ..., F128=7)
   surviving_identity_count = int(identity_mask[:partial_cut_length].sum())
   surviving_number_chunk_count = int(number_mask[:partial_cut_length].sum())
   ```
6. Aggregate per-variant totals into `Stage2Variant.total_surviving_*` fields. Compute per-batch row totals (per-variant, indexed by `stage1_variant.batch_idx`) → cumsum into `identity_row_offsets`, `number_row_offsets` at level 1 (Stage2Batch).

Output: `Stage2Batch` (D9).

### Stage 3: bulk u8 buffer + 2D indexers + vectorized FP normalization to f96 sidecar shape

Operates over `Stage2Batch`. Produces `Stage3Batch` (D9). The batch-shared arrays (inline_bytes, per-TokenType significand+sign_exp, identities_flat_caller_local) live at level 1; per-level-4 slices into them are recorded on each `Stage3CallTarget`.

Iteration order: sections → variants → call_targets in DFS encounter order, equivalently the same linearization stage 1 used. The linear traversal is what determines each call_target's offset slices into level-1 arrays.

1. Compute `inline_bytes` total size: `1 + sum_over_level4(surviving.inline_byte_count)`. Leading slot 0 is the zero pad referenced by short-payload indexers. Allocate.
2. For each level-4 call_target in linear order: extract surviving inline bytes via ALG-1; write into `inline_bytes` at the call_target's allocated offset. Record `Stage3CallTarget.inline_byte_slice`.
3. Build `identity_idx_2d` per ALG-5 (skipping prepend slots — written by stage 4 in ALG-9). View-cast to `identities_flat_caller_local: u16[N]`. Per-level-4 `identity_slice` includes the leading prepend slot (length = 1 + surviving_in_stream_identity_count for that call_target).
4. Build per-TokenType number `idx_2d` arrays per ALG-7:
   - Gather source positions across all level-4 call_targets in linear order, in stream-position order within each call_target.
   - For VC2 + F128: per ALG-8, K rows per source (K from stage 2's chunk-count predictions).
   - For F16/BF16/F32/F64: 1 row per source.
   - For F80: 1 row per source (10 bytes; reassembled via 5 u16 limbs in ALG-7).
5. **Vectorized per-TokenType FP normalization to (u64 significand, u32 sign_exponent)** per ALG-7. Bit-field extraction → renormalize the mantissa so the leading 1 sits at bit 63 ("without integer bit") → pack sign + biased exponent via the existing `pack_sign_exp` convention. Stage 3 produces per-TokenType `(significand: u64, sign_exp: u32)` array pairs, stored at level 1 in `numbers_per_TokenType`.
6. **Multi-chunk per-source bookkeeping**:
   - VC2: per-chunk `exponent_base = chunk_index_within_source × 64`. The chunk's `u64` bit-position-of-leading-1 adds the residual offset. Zero chunks emit canonical signed-zero (exponent = 0).
   - F128: per-source `f128_chunk_counts` from ALG-2 (1 for NaN/Inf, 2 for finite). Finite emits two chunks with `chunk_index ∈ {0, 1}`; NaN/Inf emits one via `_encode_infnan` formulae.
7. **Per-level-4 number_chunk_slices**: each `Stage3CallTarget` records its per-TokenType slice into the level-1 `numbers_per_TokenType[T]` arrays. Stage 4 uses these to interleave per-row number arrays in stream order.

Output: `Stage3Batch` (D9). Stage 4 only concatenates the per-call-target slices into the global `numbers_significant` + `numbers_sign_exponent` in row-order; no normalization remaining at stage 4.

### Stage 4: assemble flat output + counter remap + variant padding

Operates over `Stage3Batch`. Produces `BatchDecodeResult` (D9). Iterate the 4-level tree: sections → variants → call_targets.

For each level-3 variant whose `stage1.batch_idx` is not None (= every real, non-padding row in the final batch):

1. Initialize per-Category dedup state per ALG-3 (LOCAL_FUNC seeded with `{root.function_name_ptr: 0}` — taken from `variant.call_targets[0].stage2.stage1.function_name_ptr`; next_fresh_id = 1. PLT/EXT_FUNC empty; next_fresh_id = 0).
2. For each level-4 call_target in `variant.call_targets` (root first, then callees in encounter order) WHOSE `surviving_token_count > 0`:
   - **Write prepend** per ALG-9: token id to `tokens[row, prepend_pos]`; counter id to `identities_flat_caller_local[call_target.identity_slice.start]` (the self_counter — the value dedup_dict_per_category[encounter_category] holds for this call_target's function_name_ptr).
   - For each FUNCTION Category that has rows in this call_target's `call_targets_section` filter: run ALG-3 dedup → build `remap_lookup` → apply to `identities_flat_caller_local[call_target.identity_slice]` skipping the prepend slot (the first entry).
   - For each COUNTER Category present in this call_target: run ALG-4 offset bump → apply to the call_target's identity slice (skipping the prepend slot).
3. Write `call_target.expanded_token_ids[:partial_cut_length]` for each surviving call_target into `tokens[row]` at the running column offset. Concatenate root + each callee; truncate at exactly `context_len`; trailing positions stay at id 0 (null-content) from the zero-allocation.
4. Concatenate per-TokenType `(significand, sign_exp)` pairs into the global `numbers_significant` + `numbers_sign_exponent`:
   - For each level-4 call_target in encounter order, walk `call_target.expanded_token_ids[:partial_cut_length]`. At each number-token position, pull the next chunk from `stage3_batch.numbers_per_TokenType[that TokenType]` via `call_target.number_chunk_slices[T]` + a per-call_target progress counter (or use `np.take` with a precomputed index gather array).
   - Write into the row's slice of `numbers_significant` + `numbers_sign_exponent` at offsets given by `identity_row_offsets[batch_idx]` (analogous for numbers).
   - NO renormalization or per-source dispatch — already done in stage 3.
5. If `include_fid_sidecar=True`: build `fid_sidecar` from each per-Category dedup_dict (reverse mapping counter_id → function_name_ptr), with per-row offsets.

Variant-padding policy enforcement happens at level 1 BEFORE this loop, via stage 1's `batch_idx_to_section_variant`:
- `PAD_NULL`: padding rows (sentinel `(UINT32_MAX, UINT32_MAX)`) are skipped at step 1 — their tokens stay id 0 (null-content); their sidecar offsets stay equal to the prior offset (zero-length slice).
- `RESAMPLE_WITHIN_SECTION`: stage 1 already filled the missing slots by resampling, so every batch_idx maps to a real variant; no special-case at stage 4.
- `RAGGED`: `batch_size == total_real_variants`; no padding rows.
- `REDISTRIBUTE`: stage 1 already redistributed; every row is real.

Output: `BatchDecodeResult` (D9), possibly carrying `intermediate=stage3_batch` if `keep_intermediate=True`.

## Module layout

| Path | Concern | Status |
|---|---|---|
| `tokenizer/aligned_data/loader/batch_decode/__init__.py` | Public re-exports: `BatchDecodeResult`, `batch_decode`, `VariantPadding`, `SectionPointerSpec`. | NEW |
| `tokenizer/aligned_data/loader/batch_decode/_types.py` | The 4 staged dataclasses (D9) + `VariantPadding` enum + `SectionPointerSpec`. | NEW |
| `tokenizer/aligned_data/loader/batch_decode/_section_walk.py` | Stage 1 — section pointer resolution + RNG variant sampling + recursive DFS callee discovery + per-variant raw-data load. Uses `BinarySession._load_matched_section_and_variants`, `_select_variant_indices`, `build_inline_decode_state`. | NEW |
| `tokenizer/aligned_data/loader/batch_decode/_length_predict.py` | Stage 2 — `expanded_token_ids` construction + extra-masks + per-row cutoff walk + surviving-count predictions. ALG-2 inline. | NEW |
| `tokenizer/aligned_data/loader/batch_decode/_bulk_bytes.py` | Stage 3 — `inline_bytes` concat + per-TokenType idx_2d construction + view-cast decode. ALG-1 + ALG-5 + ALG-7 + ALG-8. | NEW |
| `tokenizer/aligned_data/loader/batch_decode/_assemble.py` | Stage 4 — per-row dedup walk (ALG-3 + ALG-4) + prepend writes (ALG-9) + token + sidecar assembly + variant-padding policy. | NEW |
| `tokenizer/aligned_data/loader/_session_splice.py` | DELETE `splice_with_callees`, `_load_root_variants`, `_decode_then_splice`, `_build_fids_per_category`. KEEP per-arm load helpers + `_select_variant_indices` for stage 1's reuse. | EDIT |
| `tokenizer/aligned_data/loader/decoded/splice.py` | DELETE entirely. | DELETE |
| `tokenizer/aligned_data/loader/decoded/extract/` | DELETE the whole package (`__init__.py`, `_identity_arm.py`, `_number_arm.py`, `_occurrence_iter.py`, `_orchestrator.py`, `_staging.py`). The per-function `_decode_to_staging` + `decode_raw_tokens` path is gone. | DELETE |
| `tokenizer/aligned_data/loader/decoded/decoded_function.py` | DELETE. `DecodedFunction` is replaced by the 4 staged dataclasses. | DELETE |
| `tokenizer/aligned_data/loader/decoded/category_tokens.py` | KEEP `resolve_category_token_ids`, `resolve_number_token_ids`, `resolve_value_negative_token_id` (used at stage 4 for the per-Category id constants); DELETE the `FID_KEYED_CATEGORIES` constant (no longer the dispatch axis under D4). | EDIT |
| `tokenizer/aligned_data/loader/decoded/__init__.py` | Drop the deleted exports; add re-exports for the batch entry. | EDIT |
| `tokenizer/aligned_data/loader/__init__.py` | Same. | EDIT |
| `tokenizer/aligned_data/loader/decoded/_inline_decode_state.py` | KEEP — reused by stage 1 per function. | UNCHANGED |
| `tokenizer/aligned_data/loader/decoded/custom_float.py` | KEEP `pack_sign_exp`, `_FP_ENCODERS`, `from_int`, `from_float*` — referenced by stage 3 (ALG-7) as the per-TokenType formulae for the VECTORIZED rewrite. The per-source functions stay callable from the test suite as Python-loop oracles for byte-equivalence tests, but stage 3 implements the batch numpy versions inline. | UNCHANGED |
| `tokenizer/aligned_data/loader/decoded/run_lengths.py` | KEEP. | UNCHANGED |
| `dedup_hashmap/src/lib.rs` + `dedup_hashmap/src/hashmap_macro.rs` (NEW) | REPLACE the hand-rolled `HashMapU64U32` + `HashMapU32U32` with a macro-generated full product: 9 key types (bool / i8..i64 / u8..u64) × 11 value types (same 9 + f32 + f64). Every class supports `capacity` constructor + `clean()` + numpy-batched `lookup_ndarray` / `insert_ndarray`. Used by stage 4's ALG-3 dedup (`HashMapU32U16` × 3 Categories, reused across rows via `clean()`). | EDIT (rewrite) |
| `dedup_hashmap/tests/test_dedup_hashmap_module.py` | ADD tests for a representative cross-section of the generated classes: `HashMapU64U32` (existing); `HashMapU32U32` (existing); `HashMapU32U16` (new, used by stage 4); `HashMapU8U8` (edge: smallest dtypes); `HashMapI32F64` (mixed-signed-key + float-value, exercises sentinel rules). Plus: capacity-respect smoke; `clean()` reuse; ndarray round-trips with miss-sentinels. | EDIT (extend) |

## Verification

1. **Unit tests** (under `tokenizer/aligned_data/loader/batch_decode/tests/`):
   - `test_section_walk.py` — stage 1: cycle handling (DAG semantics), RNG reproducibility, `max_depth` cutoff, `inlined_equivalent_call_targets_only` filter, `batch_idx_to_section_variant` mapping for all 4 variant-padding policies.
   - `test_length_predict.py` — stage 2: VC2 chunk-count formula (boundary cases: 0/1/8/9/16/17 byte payloads), F128 NaN/Inf detection on synthetic bit patterns (high u16 = 0x7FFF for inf; high u16 = 0x7FFF + non-zero mantissa for NaN; high u16 < 0x7FFF for finite), `expanded_token_ids` correctness vs a Python reference, per-row cutoff walk with cut function selection, surviving-count predictions.
   - `test_bulk_bytes.py` — stage 3: `inline_bytes` concat byte-equivalence against Python reference; identity idx_2d view-cast on synthetic short-payload fixtures (0/1/2 byte); per-TokenType number idx_2d + vectorized FP normalization byte-equivalence against the Python-loop `custom_float.from_float16` / `from_float32` / `from_float64` / `from_float80` / `from_float128` / `from_int` (oracle); VC2 multi-chunk byte layout (17-byte fixture validates short-MSB-chunk); F128 NaN/Inf emits 1 chunk via `_encode_infnan` vs finite emits 2 chunks; F80 mantissa explicit-integer-bit stripping; VC2 zero-chunk → canonical signed-zero shape.
   - `test_assemble.py` — stage 4: FUNCTION-category dedup (intra-row dedup against repeated function_name_ptrs); COUNTER-category offset renumbering; self-prepend (root's LOCAL_FUNC counter = 0, callee's freshly-issued counter); truncation at `context_len`; mid-multi-chunk truncation drops trailing chunks; null-content padding tail; all 4 `VariantPadding` policies.
   - `test_batch_decode_end_to_end.py` — full pipeline: build a synthetic fixture, run `batch_decode`, byte-equivalence against a Python reference impl that walks the splice tree per-function (separately implemented in the test).
2. **Migration audit**: `git grep -n 'splice_with_callees\|_decode_to_staging\|decode_raw_tokens\|_resolve_fid_payload\|FID_KEYED_CATEGORIES\|DecodedFunction\b'` returns zero hits outside the `batch_decode/` package or the audit test itself.
3. **Full suite**: `nix develop --command pytest tokenizer/ -q` green. Today's count is 1110; the migration deletes ~30 tests from the old `decoded/extract/tests/` path and adds ~50 new tests under `batch_decode/tests/`. Final count ≈ 1130, all green.
4. **Smoke**: build a real memmap dir (e.g. `/tmp/sb_test/out` from a small corpus); call `batch_decode(session, section_pointers=[(arm, idx)] * 4, num_variants_per_section=3, context_len=512, max_depth=2, variant_padding=VariantPadding.PAD_NULL)`. Spot-check:
   - `result.tokens.shape == (12, 512)` (4 sections × 3 variants).
   - `result.identity_row_offsets[12] == result.identities.shape[0]`.
   - `result.number_row_offsets[12] == result.numbers_significant.shape[0]`.
   - `result.tokens == 0` only at trailing-pad positions (use `np.diff(np.where(result.tokens == 0, 1, 0))` to find the transitions per row).
   - Sanity-check a known LOCAL_FUNC chain: the root prepend's identity == 0; a callee's prepend identity == the deduped LOCAL counter for the callee's FID.

## Execution discipline

### Per-task lifecycle (apply to every code-changing task in the plan)

1. Run validation + tests (`nix develop --command pytest tokenizer/ -q` + any task-specific validators). Iterate until green.
2. Merge local `master` into the working branch. If anything changed (master advanced), re-run validation + tests until green.
3. Merge the working branch into local `master`. If `master` advanced again during the previous step, repeat steps 2–3.
4. If local `master` has new commits beyond `origin/master`, push to `origin/master`.
5. `master` MUST NEVER carry uncommitted code. On entry to any task, audit working-tree state: if dirty, decide whether the work is complete + error-free; if so, commit it so nothing gets messed up.

### Commit message style (VERBATIM — must be quoted to every subagent prompt)

> in commit messages: do not include co-authered-by etc, and do not refer to task planning internal details in the commit message. e.g. no 1A, F3 or B2.1. Commit messages must be brief and not include extensive text. they should be written by default lazy commit message style that most pros use. that means a commit message should not have more then the title line and one sentence of extra info + signoff at most

Operationally: every commit uses `git commit --signoff`. NO `Co-Authored-By` (any variant). NO internal task identifiers in the message body. Title ≤70 chars + at most ONE sentence body + signoff trailer.

Concrete examples (to bake into subagent prompts):
- OK: `batch_decode: vectorize stage 3 FP normalization\n\nReplace per-source from_floatXX with batch numpy view-casts.\n\nSigned-off-by: …`
- NOT OK: paragraphs of body; "Phase 2.A.1 implementation (per plan §D9)"; bullet lists; `Co-Authored-By:` trailer.

### Subagent spawn protocol (every spawn prompt MUST include verbatim)

- The subagent's FIRST action, before reading the task description:
  ```
  git fetch /home/sirati/devel/python/asm-tokenizer/.claude/worktrees/dataloader worktree-dataloader
  git reset --hard FETCH_HEAD
  ```
  Then verify `git rev-parse HEAD` matches the parent's commit hash supplied in the spawn prompt. STOP and report if not.
- Spawn flags: `isolation: "worktree"`, `model: "opus"` (Opus 4.7 1M context), `run_in_background: true`.
- **Worktree discipline (HAMMER THIS HOME in every spawn prompt)**:
  - Stay on the assigned worktree branch (the one `isolation: "worktree"` provisions). Never `git checkout main`/`master`. Never commit to main/master directly.
  - `git status && git rev-parse --abbrev-ref HEAD` before any commit to verify the branch.
  - DO NOT `git push` anywhere. NEVER create a remote branch. Local commits only — the parent pulls from the worktree branch on completion.
  - DO NOT `git rebase -i`, `git commit --no-verify`, `git push --force`, or any other destructive op.
- The verbatim commit-rules block (above) is quoted into the prompt as a literal block.

### Encourage further splitting

Subagents are encouraged to further split their assigned subtask into self-contained sub-subtasks and spawn their OWN subagents (also `model: "opus"`, also `isolation: "worktree"`, also `run_in_background: true`). If a piece of work has a clear seam, the plan below already pre-identifies the split.

### Per-phase audit subagents (mandatory)

After every phase (1, 2, 3, 4) completes and merges, spawn TWO parallel audit subagents:

- **Audit-Regression**: walks the git history of the phase, inspects every changed file for regression, sloppy work, lazy work, incomplete work, or unmaintainable code. Reports a punch list.
- **Audit-PlanDrift**: walks the commit history vs. the corresponding plan section, checks whether the phase was implemented IN FULL or if there was drift / unacceptable shortcuts. The plan is authoritative; this audit enforces adherence.

Both audits are BLOCKING: their findings must be addressed (fix, defer with user agreement, or document) before the next phase starts.

### Plan acceptance handshake

When the user accepts this plan:

1. Parent immediately updates the task list in detail with all subtasks (one TaskCreate per sub-bullet from the parallelization map below; per-subtask state via TaskUpdate).
2. Parent yields and waits for the user to compact the conversation. The user may send additional messages before compaction that should be folded into the plan.
3. When the parent detects compaction has occurred (post-compaction summary message), it begins execution by spawning the first batch of parallel subagents per the parallelization map.

## Parallelization

The 4 stages have a SEQUENTIAL DATA DAG (stage N reads stage N-1's output), but the IMPLEMENTATION splits into phases that can run partially in parallel:

### Phase map

```
Phase 0 — Foundations (must complete before stages can be implemented)
   ├── 0a. Rust hashmap macro (full Cartesian product + capacity + clean() + ndarray APIs)
   ├── 0b. _types.py — the 4-level dataclasses for all stages (D9)
   └── 0c. Stage-1 + stage-2 + stage-3 + stage-4 stub modules with TODO bodies
        so phase-1..4 subagents can import skeletons and write tests against them.

Phase 1 — Stage 1: section walk + raw-data load (depends on 0b + existing helpers)
   ├── 1a. Section pointer resolution + RNG variant sampling (reuses _select_variant_indices)
   ├── 1b. DFS callee recursion + cycle detection
   ├── 1c. batch_idx_to_section_variant per variant_padding policy (ALG-10)
   └── 1d. Stage 1 tests (parallel-safe; synthetic Section + FunctionData fixtures)

Phase 2 — Stage 2: length predict + cutoff (depends on 0b + 0c stage-1 stub for tests)
   ├── 2a. expanded_token_ids construction + extra_*_mask building (ALG-2 inline)
   ├── 2b. Per-variant cutoff walk (cumsum + cut function identification)
   ├── 2c. Surviving identity + number-chunk count prediction
   └── 2d. Stage 2 tests (synthetic Stage1Batch fixtures)

Phase 3 — Stage 3: bulk byte buffer + FP normalization (depends on 0a + 0b)
   ├── 3a. Inline-byte concat + leading-zero pad (ALG-1)
   ├── 3b. Identity idx_2d construction + view-cast (ALG-5)
   ├── 3c. Number idx_2d per TokenType (ALG-7 byte layouts + ALG-8 VC2 chunking)
   ├── 3d. Vectorized FP normalization to f96 sidecar (ALG-7) — denormal + NaN/Inf branches
   └── 3e. Stage 3 tests (synthetic Stage2Batch fixtures + byte-equivalence vs custom_float.py oracles)

Phase 4 — Stage 4: assemble + remap + variant padding (depends on 0a + 0b + 0c)
   ├── 4a. Per-row dedup walk (ALG-3 + ALG-4) using the Rust hashmap from phase 0a
   ├── 4b. Prepend slot writes (ALG-9)
   ├── 4c. Token tensor assembly + truncation at context_len
   ├── 4d. Sidecar concat per row (significand / sign_exp / fid_sidecar)
   ├── 4e. variant_padding policy enforcement at level 1
   └── 4f. Stage 4 tests (synthetic Stage3Batch fixtures + all 4 VariantPadding policies)

Phase 5 — Migration + deletes (sequential after Phases 1–4 merge)
   ├── 5a. Delete old splice + decoded/extract package + decoded_function.py
   ├── 5b. Delete _session_splice.splice_with_callees + downstream consumers
   ├── 5c. Rewrite consuming tests against the batch entry
   └── 5d. End-to-end smoke against a real memmap

After EACH phase merges:
   Audit-Regression + Audit-PlanDrift (parallel, BLOCKING per Execution discipline)
```

### Parallel-safe subtasks within phases

- Within Phase 0: 0a (Rust), 0b (Python dataclasses), 0c (Python stubs) are independent. Spawn 3 subagents in parallel.
- Within Phase 1: 1a + 1b + 1c can run in parallel (different concerns); 1d follows. → up to 3 subagents parallel.
- Within Phase 2: 2a + 2b + 2c can run in parallel against shared stage-1 stub fixtures; 2d follows. → 3 parallel.
- Within Phase 3: 3a + 3b + 3c + 3d are mostly independent (different file regions); 3e follows. → up to 4 parallel.
- Within Phase 4: 4a + 4b + 4c + 4d + 4e are independent; 4f follows. → up to 5 parallel.
- Within Phase 5: 5a + 5b are independent (different deletes); 5c follows; 5d follows. → up to 2 parallel + 2 sequential.

### Subagent encouragement to further split

Every spawned subagent receives a closing block in its prompt:

> If your assigned subtask has internal seams (multiple independent files, multiple independent algorithm pieces), feel free to further split into sub-subtasks and spawn your OWN subagents (also `model: "opus"`, `isolation: "worktree"`, `run_in_background: true`). After your sub-batch merges, review their work + integrate before reporting back.

### Subagent worktree discipline (HAMMER home in every spawn prompt)

The fetch-parent protocol + the worktree-only constraint + the no-push rule are repeated verbatim per the Execution-discipline section above. Subagents OFTEN accidentally touch the main branch — every prompt must drill this in explicitly, not by reference.

## Out of scope (follow-ups)

- **EXT_FUNC body inlining** — explicitly out of scope (no body to inline).
- **Beyond-batch reuse of `DecodedBatch.intermediate`** — currently returned only when `keep_intermediate=True`; future consumers may want to share decoded batches across multiple BatchDecodeResults for streaming-style training, but that's outside this plan.
