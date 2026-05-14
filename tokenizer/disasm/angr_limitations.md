# angr provider — v2 limitations and Ghidra equivalents

This file catalogs v2 features that `tokenizer/disasm/angr_provider.py`
+ `tokenizer/address_meta_data_lookup.py` (the angr-side `MetadataLookup`)
cannot reliably produce, and what the Ghidra-side provider
(`tokenizer/disasm/ghidra_provider.py`) emits instead. Ghidra is the v2
default provider (see project memory `ghidra_default_provider.md`); the
angr backend is retained for architectures and use-cases where its
limits are tolerable.

For every gap below, the angr metadata-lookup falls back to a
conservative value and the classifier in `tokenizer/constant_handler.py`
demotes to a lower-precedence step of the rule list documented in
`precedence.md`. No silent misclassification: each fallback emits an
unambiguous token that downstream tools can recognize.

## 1. FP-immediate detection

**Symptom (angr-side).** FP literal immediates (`vmov.f32 d0, #1.5`,
`fld dword ptr [rip+...]`, FP-to-GP-register moves through immediates,
SIMD lane immediates) are reported as integer-typed operands. The
classifier therefore falls through to step 11 of the precedence list and
emits `valued_const <bytes>` with the bit pattern as a raw integer. The
postfix `floatXX` annotation on a `ro_data_ptr` loaded by an FP
instruction is also absent: angr cannot say "this load is FP-typed" at
the operand level.

**What angr does instead.** Capstone exposes per-architecture
FP-operand kinds in only a subset of instructions, and angr/VEX never
re-tags an integer-decoded immediate as FP. In practice the immediate
arrives at `tokenizer/arch/operands_base.py` as Capstone `OP_IMM` with
no FP precision hint. The `AngrMetadataLookup.lookup` path has no
operand-context input at all — it sees an address, not an instruction
operand — so it cannot supply an FP-width signal from this side either.
The fallback path emits `valued_const` and the FP-ness is lost.

A best-effort uplift is possible per-arch (e.g., recognize `vmov.f32` /
`vmov.f64` / `movss` / `movsd` mnemonics in the architecture provider)
but is not authoritative: many instructions use the same encoding for
integer and FP context, and Capstone's `cs_arm_op.fp` / `cs_x86_op`
fields are populated inconsistently across releases.

**What Ghidra produces.** `OperandType.FLOAT` from
`ghidra.program.model.lang` is a per-operand bitmask set by the
Ghidra processor specification (SLEIGH) at decode time. When
`ghidra_insn.getOperandType(i) & OperandType.FLOAT` is non-zero the
operand carries an FP value, with the precision derived from the
operand's size. This is authoritative — it comes from the same
SLEIGH definition that decoded the instruction — and applies uniformly
across every architecture Ghidra supports. The v2 classifier consumes
this signal at precedence step 1, so Ghidra emits `float16` /
`bfloat16` / `float32` / `float64` / `float80` / `float128` followed by
the W inline digit bytes for the IEEE bit pattern.

For postfix annotation (an FP load of a `ro_data_ptr`), Ghidra likewise
flags the load instruction via `OperandType.FLOAT` on the destination
register operand, letting the classifier append `floatXX` after the
emitted ptr token without inline digits.

## 2. Vtable detection (RTTI)

**Symptom (angr-side).** Slots of C++ virtual function tables in
`.rodata` / `.data.rel.ro` are emitted as `ro_data_ptr <id>` /
`rw_data_ptr <id>` with no `vtable` modifier. The model trainer cannot
distinguish a vtable slot from any other code-pointer-array slot or
from a plain data pointer.

**What angr does instead.** angr has no GCC/MSVC RTTI analyzer.
`AngrMetadataLookup` builds three indices (`exact_lookup` for symbols
and function entries, `range_lookup` for sections and bss segments,
and the `library_ranges` mapping addresses to loaded objects). None of
these record C++ class hierarchy or `Vftable`-tagged data. The classifier
therefore cannot distinguish vtable slots from other rodata pointers and
emits the bare ptr token at precedence step 9 / 10.

A best-effort heuristic could be layered on top (e.g., "a `.rodata`
region containing a contiguous run of in-`.text` pointers, preceded by
a typeinfo pointer at offset −1") but no such pass is implemented and
this would still miss MSVC `.?AV` and complete-object-locator
indirections.

**What Ghidra produces.** Ghidra's `GccRttiAnalyzer` and
`WindowsResourceReference` / `RTTIAnalyzer` pass tag the vtable's
`Data` object in the Listing with the symbol name `Vftable` (the
Itanium ABI form is `_ZTV<mangled>`, the analyzer renames the
referenced data accordingly). When the classifier reaches step 8 of
the precedence list, the metadata-lookup returns
`code_ptr_table_kind="vtable"` and the classifier emits the
`vtable` modifier before the resolved-slot token.

## 3. Switch-table recovery

**Symptom (angr-side).** Compiler-lowered `switch` statements that use
a dispatch-table-in-memory (the common dense / jump-table form, often
`jmp [base + idx*8]` on x86-64, `ldr pc, [pc, idx, lsl #2]` on ARM,
or `jr $t9` after a `lw` from the table on MIPS) are not consistently
recovered. The constant pool that backs the table either appears as
plain `ro_data_ptr` slots with no `jump_table` token in the function's
footer, or the indirect jump dissolves into an
`UnresolvableJumpTarget` placeholder function that
`AngrDisassemblyProvider.iter_functions` filters out
(`angr_provider.py:78`).

**What angr does instead.** angr's `CFGFast` populates
`cfg.kb.indirect_jumps` with whatever its jump-target-resolver
plugin chain manages to resolve. Coverage is uneven: dense table
lookups on x86-64 (`-fjump-tables`) are usually recovered; sparse /
binary-search lowerings are not; tables behind a register-spilled
base pointer or behind a function-call boundary are not; ARM tables
encoded inline in `.text` between BB ends and the next BB start
(common with `tbb`/`tbh`) are partially recovered. There is no
`block_def jump_table <id>` footer emission path on the angr side
for the unresolved cases.

**What Ghidra produces.** Ghidra combines static-analysis lookup
tables (`SwitchAnalyzer`, processor-specific jump-table recognizers)
with the `Listing`-backed flow analysis. Recovered tables appear as
`AddressTable` data with cross-references from the dispatch
instruction. The v2 jump-table analysis pass walks these in
function finalization, appending `block_def jump_table <id>` plus
slot contents (block tokens, in slot order) to the function's stream.

## 4. String detection

**Symptom (angr-side).** A pointer into a string literal is emitted as
`ro_data_ptr <id>` (or `rw_data_ptr <id>` for `.data`-resident strings)
with no `string_ptr` token. The string sidecar
(`<binary>_strings.bin`) ends up missing every string that the
heuristic below misses; conversely, ASCII byte runs in non-string
rodata regions are falsely emitted as `string_ptr`.

**What angr does instead.** angr has no `Listing`-equivalent that
records the semantic data-type of each address. The closest substitute
is the heuristic in `tokenizer/disasm/angr_provider.py:42` —
`re.finditer(b"[\x20-\x7e]{4,}\x00", data)` over the raw bytes of each
configured section (default `.rodata`). This recognizes ASCII-only,
length-≥4, null-terminated strings. It misses:

- UTF-8 strings containing any byte ≥ `0x80` (e.g., any non-ASCII
  literal in a localized binary).
- UTF-16LE / UTF-16BE wide strings (every other byte is `0x00`, the
  ASCII run filter rejects them).
- Pascal-style strings (length prefix instead of null terminator).
- Strings shorter than four characters.
- Strings in any section other than the ones explicitly listed (no
  `.data.rel.ro`, no `.rdata` on PE, no `__cstring` on Mach-O).

It also false-positives on ASCII-byte sequences inside non-string
rodata (e.g., short ASCII tags inside a packed struct, opcode bytes
in an embedded interpreter's bytecode table).

The classifier therefore falls through from precedence step 7 to
step 9 / 10 for missed strings, and false-positively assigns
`string_ptr` to non-string rodata addresses that happen to match
the regex.

**What Ghidra produces.** Ghidra's `ASCIIStringAnalyzer` and
`UnicodeStringAnalyzer` walk the program's data regions during
auto-analysis and create typed `Data` objects covering each string.
`Listing.getDataAt(addr).getDataType()` then returns one of
`StringDataType`, `TerminatedStringDataType`, `UnicodeDataType`,
`TerminatedUnicodeDataType`, `PascalStringDataType`,
`PascalUnicodeDataType`, etc., with the encoding encoded in the
type's name. The v2 string sidecar consumes these directly:
the `Data.getBytes()` payload is the (unescaped) string content,
the data type identifies the encoding for the `encoding` field of
the `string_ptr` metadata triplet, and pointer arithmetic against
`Data.getMinAddress()` yields the `start_offset` for substring
references.

## 5. RISC-V

**Symptom (angr-side).** A RISC-V binary cannot be tokenized at all
through the angr provider — `angr.Project(path)` fails (or accepts the
load but produces no decoded instructions) because RISC-V is not
implemented in VEX, angr's intermediate representation. There is no
partial coverage to fall back to; the entire pipeline aborts for this
ISA.

**What angr does instead.** Nothing. The user-facing message depends
on the angr release: older releases raise `angr.errors.SimEngineError`
("Unsupported architecture"), newer releases tolerate the load but
the CFG is empty. Either way no tokens are produced.

**What Ghidra produces.** Ghidra has a full RISC-V SLEIGH
specification covering RV32 and RV64 with the standard extensions
(I, M, A, F, D, C). Operand decoding, FP-operand tagging
(`OperandType.FLOAT` on F/D extension instructions), section
detection, switch tables, and string analysis all work as on any
other supported ISA. RISC-V binaries flow through the v2 pipeline
unchanged; the CSV schema (column layout, `version=2` prelude,
JSON metadata shape, `format_version=2` vocab.csv, reserved IDs
0–255, sidecar file naming) is identical to other architectures.

## 6. PPC64 ELFv1 function descriptors

**Symptom (angr-side).** On PPC64 ELFv1 (big-endian SYSV ABI, as
used by AIX and older Linux distributions), a function call goes
through a function descriptor: a 24-byte record in `.opd` containing
`{entry_addr, toc_addr, env_addr}`. The compiler emits indirect calls
through the descriptor's first slot. The angr classifier sometimes
sees the descriptor address (as a `.opd` pointer) and sometimes sees
the entry address (the descriptor's first word), depending on whether
the calling convention is fully reconstructed at the call site.
The same logical callee is therefore emitted with two different
identity counters, or split between `ro_data_ptr` (descriptor) and
`local_func` / `ext_func` (entry).

**What angr does instead.** CLE (angr's loader) implements
`.opd` parsing in `cle.backends.elf.elf.ELF._load_opd_entries`. When
parsing succeeds, function symbols are rebased to point at the
descriptor's entry address. The resolution path is fragile, however:
mixed code paths in `AngrMetadataLookup._build_indices` register
function entries through `cfg.kb.functions` (which usually has the
entry address) while sections are registered through
`loader.main_object.sections` (which has the `.opd` descriptor
address). At an indirect call through a register-loaded descriptor
pointer, the constant flowing through the operand is the `.opd`
address, not the entry; the lookup classifies it as a data pointer
and the call's callee identity is lost.

**What Ghidra produces.** Ghidra's PPC64 ELFv1 analyzer recognizes
`.opd` records as function descriptors during auto-analysis and
back-propagates the entry-point reference. `Listing.getDataAt(opd_addr)`
returns a structured `Data` whose first field is the entry address,
and the function manager's `getFunctionContaining` accepts both the
descriptor address and the entry address as references to the same
function. The v2 classifier therefore emits a single `local_func` /
`ext_func` identity regardless of whether the calling convention
reaches the descriptor or the entry.

## Where angr remains fully functional

The angr backend continues to work for:

- **Architectures.** x86, x86_64, AArch64 (ARM64), ARM32 (including
  Thumb), MIPS (MIPS32 and MIPS64, both endians), PowerPC ELFv2
  (modern Linux PPC64LE / PPC64 ELFv2 ABI; the descriptor problem
  above is ELFv1-specific). Capstone is the disassembler for all of
  these and produces decoded operands suitable for the v2 token
  classes that do not require FP context, RTTI, switch-table
  recovery, or `Listing`-typed string analysis.
- **Basic classification.** The 11-step precedence list still applies;
  angr supplies enough signal for steps 2 (`.plt` membership via
  section name), 3–6 (function entry / containment / external object
  classification via `cfg.kb.functions` and the `is_simprocedure /
  is_plt` flags, with the symbol-type fix in
  `address_meta_data_lookup.py` deriving `STT_OBJECT` / `STT_FUNC` /
  `STT_TLS` from CLE's exposed ELF symbol type), 9–10 (section-based
  rodata / data / bss / tdata / tbss classification), and 11
  (raw `valued_const`).
- **PLT-vs-local distinction.** The
  `is_simprocedure or is_plt` check at
  `address_meta_data_lookup.py:112` correctly identifies PLT stubs
  and SimProcedures and lets the v2 classifier route them to
  `plt_func` (precedence step 2) or `ext_func` (steps 5–6).
- **Cross-provider invariants.** For the architectures listed above
  and where the angr signal is present, the same x86_64 ELF
  tokenized through angr and through Ghidra emits the same
  `plt_func`, `local_func`, `ext_func`, and `block` identities at the
  same instruction boundaries (per the cross-provider invariants
  section of `vivid-tinkering-wilkes.md`). Where the angr signal is
  absent — FP operands, vtables, switch tables, strings — the
  fallback emits conservative tokens as documented in each section
  above; the two providers do not disagree, they emit at different
  precedence steps.

Ghidra is the v2 default disassembly provider per project memory
`ghidra_default_provider.md`. Use the angr provider only when its
limitations above are acceptable for the downstream consumer (e.g.,
a model trainer that does not need FP-vs-integer separation, vtable
tags, or switch-table recovery, and whose corpus does not include
RISC-V binaries).
