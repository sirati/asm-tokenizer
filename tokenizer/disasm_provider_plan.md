# Plan: Abstract the Disassembly Backend Behind a Provider Interface

## Context

The tokenizer is currently hardcoded to angr+Capstone for binary analysis (loading, CFG, function/block/instruction iteration, section parsing, metadata lookup). angr lacks RISC-V support and may have limitations for other tasks. We want to make the disassembly backend swappable (angr, Ghidra, etc.) without duplicating any tokenizer logic.

There are two distinct provider layers:
1. **DisassemblyProvider** (NEW) — abstracts the binary analysis backend (angr vs Ghidra)
2. **ArchitectureProvider** (EXISTING) — abstracts per-ISA tokenization logic (x86 prefixes, ARM condition codes, etc.)

The architecture providers stay untouched (they own tokenization, not disassembly). The DisassemblyProvider owns everything that touches the binary analysis tool.

---

## Current angr Coupling Points (Exhaustive Audit)

### Files that directly import angr

| File | What it uses angr for |
|------|----------------------|
| `run_tokenizer.py` | `angr.Project(path)`, `project.analyses.CFGFast(normalize=True)`, `project.loader.main_object.sections` (.text vaddr/memsize), type annotation `angr.analyses.cfg.cfg_fast.CFGFast` |
| `fill_constant_candidates.py` | Type annotation `angr.knowledge_plugins.functions.function.Function`, `func.blocks` iteration, `block.capstone.insns` access |
| `address_meta_data_lookup.py` | Creates its OWN `angr.Project(path, auto_load_libs=True)`, builds `CFGFast(normalize=True, regions=code_regions)`, accesses `loader.main_object.sections`, `loader.all_objects`, `loader.main_object.segments`, `project.kb.symbols`, `cfg.kb.functions.values()`, classifies functions via `func.binary == main_obj` |
| `csv_files.py` | `proj.loader.main_object.sections` (rodata), `proj.loader.memory.load(vaddr, memsize)`, `proj.arch.bytes` (word size for init_array) |
| `main_loop.py` | Consumes `cfg.functions.items()` (dict[int, Function]), accesses `cfg.functions[func_addr].name` |

### Capstone instruction object attribute usage (exhaustive)

All architecture providers receive an `insn` parameter that is angr's CapstoneInsn wrapper around Capstone's CsInsn.

**Universal (all architectures):**
- `insn.mnemonic` -> str
- `insn.op_str` -> str (used only for debug display in fill_constant_candidates.py)
- `insn.operands` -> list of operand objects
- `insn.insn.insn_name()` -> str (canonical instruction name via underlying Capstone CsInsn)
- `insn.reg_name(reg_id)` -> str (register ID to name resolution)

**x86-specific:**
- `insn.prefix` -> bytes (iterable of prefix bytes)
- `insn.reg_name(op.mem.segment)` -> str (segment register name)

**ARM32-specific:**
- `insn.insn.cc` -> int (condition code)
- `insn.insn.update_flags` -> bool (S suffix)
- `insn.insn.writeback` -> bool (! suffix)

**PPC-specific:**
- `insn.insn.bc` -> int (branch condition code)
- `insn.insn.update_cr0` -> bool (Rc bit / dot suffix)

**MIPS, RISC-V:** No arch-specific instruction attributes beyond the universal set.

### Capstone operand object attribute usage (exhaustive)

**Universal:**
- `op.type` -> int (1=REG, 2=IMM, 3=MEM; PPC also uses 64=CRX)
- `op.reg` -> int (register ID, when type==REG)
- `op.imm` -> int (immediate value, when type==IMM)

**Memory operand (type==MEM):**
- `op.mem.base` -> int (base register ID)
- `op.mem.disp` -> int (displacement value)
- `op.mem.index` -> int (index register ID; x86 and ARM32)
- `op.mem.scale` -> int (scale factor; x86 only)
- `op.mem.segment` -> int (segment register ID; x86 only)

**x86-specific:** `op.size` -> int (operand size in bytes, for pointer-length tokens)
**ARM32-specific:** `op.shift.type` -> int, `op.shift.value`/`op.shift.imm` -> int
**PPC-specific:** `op.crx.reg` -> int (condition register field, when type==64)

### Where `insn.reg_name(reg_id)` is called

1. `token_manager.py:254` — `VocabularyManager.get_registry_token(insn, reg_id)` calls `insn.reg_name(reg_id)`. This is the **ONLY** place outside architecture providers that touches the instruction object.
2. `arch/x86/operands.py:68` — `insn.reg_name(op.mem.segment)` for x86 segment register.
3. Indirectly via `vocab_manager.get_registry_token(insn, ...)` — called from every architecture provider and from `operands_base.py`.

---

## Key Design Decisions

### 1. Instruction objects: pass-through, NOT full abstraction

Architecture providers access deeply ISA-specific Capstone attributes (`insn.insn.cc`, `op.shift.type`, `insn.prefix`, etc.). Abstracting these into a universal dataclass would create a bloated union type for zero semantic gain.

**Decision:** Instruction objects remain Capstone-native and flow through opaquely. When a Ghidra backend is added, it produces Capstone-compatible adapter objects so architecture providers work unchanged. This avoids duplicating any tokenizer logic.

The correct layering is:
```
DisassemblyProvider (angr vs ghidra)
    |
    v produces
Instruction objects (Capstone-shaped vs Ghidra-adapter-shaped)
    |
    v consumed by
ArchitectureProvider (x86 vs arm32 vs mips vs ppc vs riscv)
```

**Rejected alternative — full Instruction dataclass:** Would require either a massive union type with every ISA's fields, or per-ISA subclasses that duplicate Capstone's own hierarchy. O(ISA * fields) work with zero value since architecture providers already know their ISA.

**Rejected alternative — runtime-enforced Protocol:** The attributes are so deeply ISA-specific that a single Protocol can't describe them. Better to document the contract and let the adapter pattern handle compatibility.

### 2. Decouple `get_registry_token` from instruction objects

`VocabularyManager.get_registry_token(insn, reg_id)` is the ONLY place outside architecture providers that touches the instruction object (calling `insn.reg_name(reg_id)`).

**Decision:** Change signature to accept the resolved string:

```python
# Before:
def get_registry_token(self, insn, reg_id) -> Tokens:
    register_str = insn.reg_name(reg_id)

# After:
def get_registry_token(self, reg_name: str, reg_id: int) -> Tokens:
    # reg_name already resolved by caller
```

Callers (all inside architecture providers that already know they have Capstone objects) change from `vocab_manager.get_registry_token(insn, op.reg)` to `vocab_manager.get_registry_token(insn.reg_name(op.reg), op.reg)`.

This cleanly removes the last instruction-object dependency from the core tokenizer.

### 3. AddressMetaDataLookup: keep separate, abstract construction

The `lookup(addr) -> (dict, str)` interface is already backend-agnostic. Only construction is angr-specific (it creates its own `angr.Project` with `auto_load_libs=True`, separate from the main project).

**Decision:** Define a `MetadataLookup` Protocol with just the `lookup()` method. The `DisassemblyProvider` gets `create_metadata_lookup() -> MetadataLookup`. Current class becomes `AngrMetadataLookup`.

### 4. Rodata parsing folds into DisassemblyProvider

`csv_files.py:parse_and_save_data_sections(proj)` uses angr's loader to iterate sections and read raw bytes.

**Decision:** Move into `AngrDisassemblyProvider.parse_data_sections()`. The provider returns the constant dict and handles file writing internally (same behavior, just moved). `csv_files.py` loses this function but keeps all other backend-agnostic utilities.

### 5. CFG function iteration goes through DisassemblyProvider

Currently `main_loop.py` iterates `cfg.functions.items()` directly and filters out pseudo-functions.

**Decision:** `DisassemblyProvider` exposes:
- `iter_functions() -> Iterable[tuple[int, str, Any]]` — yields `(addr, name, func_object)` pre-filtered
- `function_count() -> int` — for progress bar sizing

The `func_object` is backend-specific and passed opaquely to `fill_constant_candidates`, which accesses `.blocks` on it.

### 6. Pickle compatibility

Existing pickles contain raw `angr.Project` and `CFGFast` objects. The hash-checked pickle system (`hash_checked_pickles.py`) detects code changes and auto-invalidates. After refactoring, old pickles will be invalidated automatically. The `AngrDisassemblyProvider` wraps angr objects that are already picklable, so new pickles work.

---

## DisassemblyProvider Interface

```python
# tokenizer/disasm/__init__.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Protocol


class MetadataLookup(Protocol):
    """Protocol for address metadata lookup objects."""
    def lookup(self, addr: int) -> tuple[dict, str]:
        """Look up metadata for an address.
        Returns (metadata_dict, source_type) where source_type is
        'exact', 'range', or 'synthetic'.
        """
        ...


class DisassemblyProvider(ABC):
    """Abstract base class for disassembly backends.

    A DisassemblyProvider encapsulates:
    - Binary loading and project creation
    - CFG construction
    - Section parsing (text boundaries, rodata constants)
    - Function/block/instruction iteration
    - Address metadata lookup construction

    The instruction objects yielded by this provider are backend-specific
    (e.g., Capstone CsInsn for angr, Ghidra Instruction for Ghidra).
    Architecture providers consume these objects directly.
    """

    @abstractmethod
    def __init__(self, binary_path: Path) -> None:
        """Load the binary. Does NOT build CFG yet."""
        ...

    @abstractmethod
    def build_cfg(self) -> None:
        """Construct the control flow graph. Call after __init__."""
        ...

    @abstractmethod
    def get_text_section_bounds(self) -> tuple[int, int]:
        """Return (text_start, text_end) virtual addresses."""
        ...

    @abstractmethod
    def parse_data_sections(
        self,
        sections: list[str] | None = None,
        output_csv_path: str | None = None,
    ) -> dict[str, list[str]]:
        """Parse .rodata (and optionally other sections) for string constants.

        Returns dict mapping hex(start_addr) -> [hex(end_addr), section_name, value].
        """
        ...

    @abstractmethod
    def create_metadata_lookup(self) -> MetadataLookup:
        """Create an address metadata lookup object.

        This may create a separate project/analysis (e.g., angr with
        auto_load_libs=True) for richer symbol resolution.
        """
        ...

    @abstractmethod
    def function_count(self) -> int:
        """Return the total number of functions in the CFG."""
        ...

    @abstractmethod
    def iter_functions(self) -> Iterable[tuple[int, str, Any]]:
        """Iterate over functions in the CFG, sorted by name.

        Yields (func_addr, func_name, func_object) tuples.
        Pseudo-functions (UnresolvableCallTarget, etc.) are excluded.

        The func_object is backend-specific and will be passed to
        fill_constant_candidates, which accesses .blocks on it.
        """
        ...


def get_disassembly_provider(backend: str, binary_path: Path) -> DisassemblyProvider:
    """Factory function for creating disassembly providers."""
    if backend == "angr":
        from tokenizer.disasm.angr_provider import AngrDisassemblyProvider
        return AngrDisassemblyProvider(binary_path)
    else:
        raise ValueError(f"Unsupported disassembly backend: {backend}")
```

---

## Implementation Phases

### Phase 1: Decouple `get_registry_token` (smallest, safest, zero behavior change)

**`tokenizer/token_manager.py`** (line 249):
- Change signature: `get_registry_token(self, insn, reg_id)` -> `get_registry_token(self, reg_name: str, reg_id: int)`
- Remove `register_str = insn.reg_name(reg_id)`, use `reg_name` parameter directly

**All callers** (12 call sites across 9 files):

| File | Line | Before | After |
|------|------|--------|-------|
| `arch/x86/provider.py` | 78 | `get_registry_token(insn, op.reg)` | `get_registry_token(insn.reg_name(op.reg), op.reg)` |
| `arch/x86/operands.py` | 76 | `get_registry_token(insn, base)` | `get_registry_token(insn.reg_name(base), base)` |
| `arch/x86/operands.py` | 82 | `get_registry_token(insn, index)` | `get_registry_token(insn.reg_name(index), index)` |
| `arch/arm32/provider.py` | 96 | `get_registry_token(insn, op.reg)` | `get_registry_token(insn.reg_name(op.reg), op.reg)` |
| `arch/arm32/operands.py` | 51 | `get_registry_token(insn, base)` | `get_registry_token(insn.reg_name(base), base)` |
| `arch/arm32/operands.py` | 56 | `get_registry_token(insn, index)` | `get_registry_token(insn.reg_name(index), index)` |
| `arch/mips/provider.py` | 51 | `get_registry_token(insn, op.reg)` | `get_registry_token(insn.reg_name(op.reg), op.reg)` |
| `arch/ppc/provider.py` | 78 | `get_registry_token(insn, op.reg)` | `get_registry_token(insn.reg_name(op.reg), op.reg)` |
| `arch/ppc/provider.py` | 109 | `get_registry_token(insn, op.crx.reg)` | `get_registry_token(insn.reg_name(op.crx.reg), op.crx.reg)` |
| `arch/riscv/provider.py` | 51 | `get_registry_token(insn, op.reg)` | `get_registry_token(insn.reg_name(op.reg), op.reg)` |
| `arch/operands_base.py` | 40 | `get_registry_token(insn, base)` | `get_registry_token(insn.reg_name(base), base)` |

### Phase 2: Create abstraction layer (additive only, no breakage)

**NEW `tokenizer/disasm/__init__.py`**:
- `MetadataLookup` Protocol
- `DisassemblyProvider` ABC with all abstract methods
- `get_disassembly_provider()` factory

**NEW `tokenizer/disasm/angr_provider.py`**:
- `AngrDisassemblyProvider(DisassemblyProvider)`:
  - `__init__(binary_path)`: creates `angr.Project(binary_path, auto_load_libs=False)`, stores project
  - `build_cfg()`: calls `project.analyses.CFGFast(normalize=True)`, stores cfg
  - `get_text_section_bounds()`: iterates `project.loader.main_object.sections` for `.text`
  - `parse_data_sections()`: absorbs logic from `csv_files.py:parse_and_save_data_sections()` — regex scan of `.rodata` bytes via `project.loader.memory.load()`, returns constant dict, writes `_consts.txt`
  - `create_metadata_lookup()`: creates `AngrMetadataLookup(binary_path)` (the renamed current class)
  - `function_count()`: `len(self.cfg.functions)`
  - `iter_functions()`: yields `(addr, name, func_obj)` sorted by name, filtering out `UnresolvableCallTarget`/`UnresolvableJumpTarget`
  - Properties: `project`, `cfg` (for pickle compatibility)

**MODIFY `tokenizer/address_meta_data_lookup.py`**:
- Rename class `AddressMetaDataLookup` -> `AngrMetadataLookup`
- Add backward-compat alias: `AddressMetaDataLookup = AngrMetadataLookup`

### Phase 3: Wire the provider into the pipeline

**MODIFY `tokenizer/run_tokenizer.py`**:
- Remove `import angr`, `from tokenizer.csv_files import parse_and_save_data_sections`
- Add `from tokenizer.disasm import get_disassembly_provider, DisassemblyProvider`
- In `run_tokenizer()`: replace `angr.Project()` + `CFGFast()` + `parse_and_save_data_sections()` with:
  ```python
  provider = get_disassembly_provider("angr", binary_path)
  constants = provider.parse_data_sections(output_csv_path=str(csv_path))
  provider.build_cfg()
  ```
- In `disassemble_to_tokens()`: replace `angr.Project()`, section iteration, `AddressMetaDataLookup()` with:
  ```python
  text_start, text_end = provider.get_text_section_bounds()
  lookup = provider.create_metadata_lookup()
  ```
- Pass `provider` to `main_loop()` instead of `cfg`
- Pickle: serialize `provider` (its internal angr objects are already picklable)

**MODIFY `tokenizer/main_loop.py`**:
- Replace `cfg` parameter with `provider: DisassemblyProvider`
- Replace `sorted(cfg.functions.items(), ...)` with `provider.iter_functions()`
- Replace `len(cfg.functions.items())` with `provider.function_count()`
- Remove `cfg.functions[func_addr].name` lookup (name comes from iterator)
- Remove `UnresolvableCallTarget`/`UnresolvableJumpTarget` filter (provider handles it)

**MODIFY `tokenizer/fill_constant_candidates.py`**:
- Remove `import angr`
- Change `func: angr.knowledge_plugins.functions.function.Function` to `func: Any`
- No logic changes — `func.blocks`, `block.addr`, `block.capstone.insns` are part of the pass-through contract

**MODIFY `tokenizer/csv_files.py`**:
- Remove `parse_and_save_data_sections()` function (moved to `AngrDisassemblyProvider`)
- Keep all other functions (they are backend-agnostic)

### Phase 4: Verify isolation and document contracts

- `grep -r "import angr" tokenizer/` should return only:
  - `tokenizer/disasm/angr_provider.py`
  - `tokenizer/address_meta_data_lookup.py`
- No other tokenizer file should reference angr
- Document instruction object contract as docstring in `tokenizer/disasm/__init__.py`

---

## File Change Summary

| File | Action | What changes |
|------|--------|-------------|
| `tokenizer/disasm/__init__.py` | **NEW** | `MetadataLookup` Protocol, `DisassemblyProvider` ABC, factory |
| `tokenizer/disasm/angr_provider.py` | **NEW** | `AngrDisassemblyProvider` wrapping all angr logic |
| `tokenizer/token_manager.py` | MODIFY | `get_registry_token` signature change |
| `tokenizer/run_tokenizer.py` | MODIFY | Use `DisassemblyProvider`, remove angr imports |
| `tokenizer/main_loop.py` | MODIFY | `provider` instead of `cfg` |
| `tokenizer/fill_constant_candidates.py` | MODIFY | Remove angr import/type annotation |
| `tokenizer/csv_files.py` | MODIFY | Remove `parse_and_save_data_sections` |
| `tokenizer/address_meta_data_lookup.py` | MODIFY | Rename to `AngrMetadataLookup` |
| `tokenizer/arch/operands_base.py` | MODIFY | Update `get_registry_token` calls |
| `tokenizer/arch/x86/provider.py` | MODIFY | Update `get_registry_token` calls |
| `tokenizer/arch/x86/operands.py` | MODIFY | Update `get_registry_token` calls |
| `tokenizer/arch/arm32/provider.py` | MODIFY | Update `get_registry_token` calls |
| `tokenizer/arch/arm32/operands.py` | MODIFY | Update `get_registry_token` calls |
| `tokenizer/arch/mips/provider.py` | MODIFY | Update `get_registry_token` calls |
| `tokenizer/arch/ppc/provider.py` | MODIFY | Update `get_registry_token` calls |
| `tokenizer/arch/riscv/provider.py` | MODIFY | Update `get_registry_token` calls |

**Unchanged** (already backend-agnostic):
`arch/provider.py`, `constant_handler.py`, `instruction_sets.py`, `tokens.py`, `function_filter.py`, `function_token_list.py`, `token_lists.py`, `__main__.py`

---

## Risks and Mitigations

### Risk 1: Pickle backward compatibility
Old pickles contain raw angr objects. After refactoring, the code hash changes and `hash_checked_pickles` auto-invalidates them. No manual intervention needed. New pickles serialize the `AngrDisassemblyProvider` (which wraps the same picklable angr objects).

### Risk 2: Future Ghidra backend instruction compatibility
A Ghidra backend needs to produce instruction objects that existing architecture providers can consume. The recommended approach is Capstone-compatible adapter/wrapper objects. If Ghidra's instruction model is fundamentally incompatible for a specific ISA, a per-ISA adapter can be written. This is O(ISA count) work, not O(ISA x instruction) work.

### Risk 3: `func.blocks` / `block.capstone.insns` pass-through contract
`fill_constant_candidates.py` accesses `func.blocks`, `block.addr`, `block.size`, `block.capstone.insns` on the opaque function object. A Ghidra backend must either produce compatible wrapper objects or these accessors need a thin abstraction. The current design chooses wrappers to avoid touching the tokenizer pipeline.

---

## Verification

After each phase, run the tokenizer against known-good binaries and compare output:

```bash
# Test all previously verified architectures
cd /home/sirati/devel/python/asm-tokenizer
for bin in \
  src/zlib/x86-gcc-5-O3_minigzipsh \
  src/zlib/arm32-gcc-5-O3_minigzipsh \
  src/zlib/mips32-gcc-5-O3_minigzipsh \
  src/hello/ppc64-clang22-O2-baseline-unhardened_hello; do
  nix develop --command bash -c "python -m tokenizer --single $bin --source ./src"
done
```

Expected: identical CSV output, identical function counts, 0 errors.

Final isolation check:
```bash
grep -r "import angr" tokenizer/ | grep -v disasm/ | grep -v address_meta_data_lookup
# Should return nothing
```

---

## Data Flow Diagram (Before vs After)

### Before (angr hardcoded everywhere):
```
run_tokenizer.py
  angr.Project(path)
  angr.CFGFast()
  parse_and_save_data_sections(proj)  [csv_files.py, uses angr loader]
  AddressMetaDataLookup(path)         [creates own angr.Project]
    |
    v
  main_loop(cfg=cfg, lookup=lookup, ...)
    for func_addr, func in cfg.functions.items():     [angr CFG]
      fill_constant_candidates(func=func, ...)        [angr Function type]
        for block in func.blocks:                     [angr Block]
          for insn in block.capstone.insns:            [Capstone instruction]
            arch_provider.parse_instruction(insn, ...) [architecture provider]
              vocab_manager.get_registry_token(insn, op.reg)  [touches insn]
```

### After (DisassemblyProvider abstraction):
```
run_tokenizer.py
  provider = get_disassembly_provider("angr", path)   [factory]
  provider.parse_data_sections()                       [angr isolated]
  provider.build_cfg()                                 [angr isolated]
  provider.create_metadata_lookup()                    [angr isolated]
    |
    v
  main_loop(provider=provider, lookup=lookup, ...)
    for func_addr, func_name, func in provider.iter_functions():  [abstract]
      fill_constant_candidates(func=func, ...)                    [opaque obj]
        for block in func.blocks:                                 [pass-through]
          for insn in block.capstone.insns:                       [pass-through]
            arch_provider.parse_instruction(insn, ...)            [unchanged]
              vocab_manager.get_registry_token(reg_name, reg_id)  [decoupled]
```
