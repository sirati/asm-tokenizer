# v2 constant classifier — precedence

This document describes how an address-or-value flowing through `ConstantHandler` is
classified into exactly one v2 token category. The rules are encoded as an explicit
ordered list (`PRECEDENCE = [...]` in `ConstantHandler`, literal — not implicit
`if/elif`); **first match wins**. Trust the disassembler's operand type wherever it
provides one — FP detection in particular is provider-authoritative.

Provider asymmetry (Ghidra-default; angr best-effort) is called out per step. The
exhaustive angr-side catalog lives in [`angr_limitations.md`](angr_limitations.md).

## The 11 precedence steps

### 1. Disassembler reports FP operand type → `floatXX`

- **Rule.** Operand FP-typed by the disassembler → emit `floatXX` (width per the
  operand's reported FP precision) followed by `W` digit bytes for the IEEE bit
  pattern.
- **Example.** x86-64 `movss xmm0, 0x3f800000` with Ghidra `OperandType.FLOAT`
  reported on the immediate → `float32` token + 4 digit bytes `[0x3F, 0x80, 0x00, 0x00]`.
- **Provider notes.** Ghidra: `OperandType.FLOAT` on the operand. angr: Capstone FP
  operand kind when available; FP detection is missing for several arch/insn
  combinations (see [`angr_limitations.md`](angr_limitations.md)) — those operands
  fall through to step 11 (`valued_const`) on the angr path.
- **Why top-precedence.** An FP-typed operand is by construction not an address;
  the disassembler's verdict is final.

### 2. Address ∈ `.plt` of any loaded object → `plt_func`

- **Rule.** Any PLT-stub address in any loaded object (main binary or shared
  library) classifies as `plt_func`.
- **Example.** Address `0x401040` in the main binary's `.plt` section (e.g. the
  PLT stub for `printf@plt`) → `plt_func` with metadata
  `{name: "printf", library: "libc.so.6"}`.
- **Provider notes.** Ghidra: section attributes + symbol kind agree. angr: today's
  provider tags main-binary PLT entries `local_function` rather than `plt_func` —
  the v2 metadata-lookup rewrite fixes this so both providers route to step 2.

### 3. Address = function entry in main object → `local_func`

- **Rule.** Address equal to the entry point of a function in the main object →
  `local_func`.
- **Example.** Address `0x4012a0` is `bar`'s entry point in the main binary →
  `local_func` with metadata `{name: "bar", addr: "0x4012a0"}`.
- **Provider notes.** Both providers support function-entry lookup.

### 4. Address inside a function in main object → `block`

- **Rule.** Address falls strictly inside a function body in the main object
  (i.e. it's a branch / call / jump target into already-known code, not the
  function entry itself) → `block`.
- **Example.** Address `0x4012c8` is inside `bar` (which starts at `0x4012a0`),
  targeted by `jne 0x4012c8` → `block` with the per-function block identity
  counter.
- **Provider notes.** Both providers can enumerate per-function block boundaries
  from CFG output.

### 5. Address = real function entry in another loaded object → `ext_func` (`synthetic=false`)

- **Rule.** Address is a *real* function entry in another loaded object (a
  resolved import target, not a CLE synthetic stub) → `ext_func` with
  `synthetic=false`.
- **Example.** Address `0x7ffff7e3a4f0` is `__libc_start_main` in `libc.so.6`
  loaded at runtime address → `ext_func` `{name: "__libc_start_main",
  library: "libc.so.6", synthetic: false}`.
- **Provider notes.** Both providers see loaded objects; Ghidra resolves these
  through its program tree, angr through CLE's loaded backend objects.

### 6. Address = synthetic extern-object slot (CLE) → `ext_func` (`synthetic=true`)

- **Rule.** Address falls inside CLE's synthetic extern object (a placeholder
  slot for an unresolved or stub import) → `ext_func` with `synthetic=true`.
- **Example.** Address inside CLE's `cle##externs` object pointing at the
  placeholder for `dlopen` → `ext_func` `{name: "dlopen", library: "libc.so.6",
  synthetic: true}`.
- **Provider notes.** This is an angr/CLE-specific concept; Ghidra typically
  routes the same imports through step 5 (real entry in loaded object) or step 2
  (PLT stub). Both code paths preserve the `synthetic` flag so downstream
  consumers can distinguish.

### 7. Address inside a provider-confirmed string → `string_ptr`

- **Rule.** Address is inside a region the provider's string analyzer has
  confirmed as a string (any encoding) → `string_ptr`. The metadata entry
  references `{line, start_offset, encoding}` in the per-binary `_strings.bin`
  sidecar; `start_offset > 0` for substring access.
- **Example.** Address `0x402008` points into the C-string `"hello %s\n"`
  starting at `0x402000` → `string_ptr` with `{line: 17, start_offset: 8,
  encoding: "ascii"}`.
- **Provider notes.** Ghidra reads its string analyzer (`Listing.getDataAt(addr)`
  for `StringDataType`, `TerminatedUnicodeDataType`, etc. — authoritative). angr
  has no built-in string analyzer; it falls back to a section + heuristic check
  (per [`angr_limitations.md`](angr_limitations.md)), so some real strings on
  the angr path fall through to step 9 (`ro_data_ptr`).

### 8. Address in a code-pointer array slot → resolved target + `vtable` / `code_ptr_table` modifier

- **Rule.** Address is a slot in a code-pointer array. Emit the modifier
  (`vtable` for RTTI-confirmed C++ vtables, `code_ptr_table` for other
  function-pointer arrays such as `.init_array`, `.fini_array`, `.dtors`,
  dispatch tables) followed by the resolved-target token. If the *specific*
  slot doesn't resolve to a known target, decompose into
  `[ro_data_ptr + valued_const]` keeping the same modifier prefix.
- **Example.** Address `0x405010` is slot 2 of a C++ vtable for class `Foo`
  whose entry points at `Foo::bar` (a local function) → `vtable` modifier +
  `local_func` for `Foo::bar`. If slot 5 doesn't resolve → `vtable` modifier +
  `[ro_data_ptr + valued_const]` decomposition.
- **Provider notes.** Ghidra: RTTI analyzer output (Data tagged `Vftable`) for
  `vtable`; section / symbol heuristics for `code_ptr_table`. angr: no RTTI
  analyzer, so `vtable` detection is effectively absent — those slots fall
  through to step 9 (`ro_data_ptr`) on the angr path. See
  [`angr_limitations.md`](angr_limitations.md).

#### 8a. Jump-table-slot routing

If the address lands inside a known switch jump table
(`kind == JUMP_TABLE_SLOT`), the classifier emits
`[jump_table(id), valued_const(offset)]` instead of decomposing the slot
via the `code_ptr_table` / `vtable` modifier. The `id` is shared with the
function-level footer's `Jump_Table` token (same `Category.JUMP_TABLE`
identity), so a constant-side reference and the footer-side declaration
resolve to the same referent.

#### 8b. Resolved-target branch (`meta.slot_target`)

When `meta.slot_target` is non-None (Ghidra-only — angr always leaves it
None per [`angr_limitations.md`](angr_limitations.md)), the modifier-slot
path recursively classifies the target itself rather than decomposing to
`ro_data_ptr + valued_const(offset)`. The emitted shape becomes
`[modifier, *target_classified_tokens]` — e.g. a vtable slot pointing at
a local function emits `[vtable, local_func(id)]`. Recursion is bounded
by the provider (`slot_target.kind` is guaranteed never to itself be a
slot kind), so the recursion terminates after one level.

### 9. Address in a rodata section → `ro_data_ptr`

- **Rule.** Address falls inside a read-only data section (`.rodata`,
  `.data.rel.ro`, equivalent) → `ro_data_ptr`.
- **Example.** Address `0x402080` points into `.rodata` (not a string per
  step 7) → `ro_data_ptr` with `{section: ".rodata", addr: "0x402080",
  name: "...", size: 24}`.
- **Provider notes.** Both providers expose section attributes. FP-ness of the
  pointed-to value is **not** classified here; see "Postfix FP annotation rule"
  below.

### 10. Address in data / bss / tdata / tbss → `rw_data_ptr` (TLS gets `thread_local` prefix)

- **Rule.** Address falls inside a writable section (`.data`, `.bss`) or a TLS
  section (`.tdata`, `.tbss`) → `rw_data_ptr`. For TLS sections, prepend a
  `thread_local` modifier.
- **Example.** Address `0x404020` in `.data` → `rw_data_ptr` with the
  appropriate metadata. Address `0x405000` in `.tdata` → `thread_local` modifier
  + `rw_data_ptr`.
- **Provider notes.** Ghidra: section attributes are authoritative. angr: same,
  via CLE's section table; `STT_TLS` symbol type provides the TLS hint when the
  section flags alone are ambiguous.

### 11. Unresolved → `valued_const`

- **Rule.** Anything that matched none of steps 1–10 → `valued_const`, encoding
  the raw value with variable byte width (1 byte through AVX-512's 64 bytes).
  The legacy `0x00–0xFF` heuristic is removed; the only width restriction is
  what the value itself requires.
- **Example.** Operand immediate `0xDEADBEEF` that isn't a known address →
  `valued_const` + 4 digit bytes `[0xDE, 0xAD, 0xBE, 0xEF]`.
- **Provider notes.** Provider-independent — this is the fallthrough catch-all.

## Postfix FP annotation rule

Separate from the precedence list. Applies *after* a step 7–10 ptr token has
been emitted: if the *load instruction* targeting that address is FP-typed
(e.g. `movss` / `movsd` / `vmovss` / `vld1.32` / `fld` / `fldd`), annotate the
load's FP type after the ptr token.

A `floatXX` token **ALWAYS carries its `W` inline IEEE digit bytes** — the
value-less bare-`floatXX` form is **FORBIDDEN** (a `floatXX` is a value-token
and the valued-token contract is inviolable). The annotation therefore takes
one of two shapes:

- **Dereference succeeded** → emit a **valued `floatXX`**: the caller reads the
  `W` bytes (`W = floatXX.width_bytes`) at the resolved load address from the
  loaded image and emits `floatXX` + those `W` digit bytes as the IEEE bit
  pattern (big-endian). This captures the *actual* FP constant the pointer
  loads, inline in the stream.
- **Dereference unobtainable** → emit the value-less `float_annotation`
  modifier token (single id, no payload). This is a pure type marker meaning
  "the previous ptr token loads an FP value whose bytes could not be read".

**Dereference policy** (which read attempts are made, by ptr kind):

| ptr kind | policy |
|---|---|
| `ro_data_ptr` | read (read-only data is image-backed and stable) |
| `rw_data_ptr` | attempt read; `.bss` / TLS-bss / unreadable → `float_annotation` |
| `string_ptr` / `jump_table` / slot | attempt read; else `float_annotation` |

In all cases an unreadable target (`read_bytes` returns `None`) falls back to
`float_annotation`.

Example (success): `ro_data_ptr <id> float32 <b0> <b1> <b2> <b3>` — the four
digit bytes are the IEEE-754 single-precision constant read from the address
the pointer references.

Example (fallback): `rw_data_ptr <id> float_annotation` — the load is FP-typed
but its bytes live in `.bss` (no image backing), so only the type marker is
emitted.

## `is_arithmetic` short-circuit

When the caller flags the operand as arithmetic context, **steps 2–10 are
skipped** and the value is routed directly to step 11 (`valued_const`). Step 1
(disassembler-reported FP type) still applies as top-precedence — an
arithmetic FP immediate is a value, but it is a `floatXX` value, not an
integer `valued_const`.

**Rationale.** An arithmetic immediate (e.g. the `0x14` in `add rax, 0x14`)
is a value being arithmetically combined, not an address dereference candidate.
Allowing it to spuriously match a `.plt` / `.rodata` / etc. range would
mislabel ordinary constants as pointer-like.

## Per-function identity counters

Each category maintains a **per-function** identity counter. Identity is
assigned on first stream-occurrence within the current function and resets
at every function boundary. There is no usage-frequency sort — today's
`ConstantHandler.create_opaque_mapping` is dropped in v2; identity is just
a monotonic counter incrementing in stream order.

Categories with identity (per the v2 token table): `block`, `local_func`,
`plt_func`, `ext_func`, `ro_data_ptr`, `rw_data_ptr`, `string_ptr`,
`jump_table`. Modifier tokens (`thread_local`, `vtable`, `code_ptr_table`),
`valued_const`, and all `floatXX` tokens carry no identity — they are
either intrinsic-valued (digit bytes inline) or pure-prefix modifiers.

## Function-level jump-table footer

The jump-table footer pass at function finalization ALWAYS runs. It emits
a `[Block_Def, Jump_Table(id), Block_V2(target0), ...]` footer per known
table, regardless of whether any constant pointing into the table was
seen during instruction tokenization. If a slot registered an identity
(via the Step-8a routing above) but the dispatch instruction wasn't
recovered by `iter_switch_tables`, the footer still emits a target-less
`[Block_Def, Jump_Table(id)]` declaration so the constant-side
`jump_table(id)` token has a resolvable referent.
