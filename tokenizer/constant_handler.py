"""Constant tokenization — v1 legacy path + v2 precedence-list classifier.

Module-owned concern: turn one integer ``value`` plus its provider-side
metadata (``meta``) into a sequence of category tokens. The v2 entry point
``ConstantHandler.process_constant_v2`` walks the 11-step precedence list
documented in ``tokenizer/disasm/precedence.md`` (literal ``_PRECEDENCE``
list at module scope; first match wins). The legacy ``process_constant``
remains for v1 callers until Phase 1.C.3 migrates them.

Boundary contract for v2:

- Caller does the address lookup (``metadata_lookup.lookup(addr) ->
  AddressMetadataView``) once, then hands ``value`` + the typed view to
  ``process_constant_v2``. The typed view exposes typed enums
  (``meta.kind``, ``meta.string_encoding``) and concrete fields
  (``meta.name``, ``meta.start_addr``, ...) — ConstantHandler reads them
  exclusively (no dict access, no string-keyed lookups).
- Caller passes ``is_arithmetic=True`` when the operand context is
  arithmetic (the value being arithmetically combined, not an address
  dereference candidate). Per ``precedence.md`` the address steps 2–10
  short-circuit; step 1 (disassembler-reported FP type) and step 11
  (``valued_const``) still apply.
- Caller passes ``fp_immediate_type`` (an ``FpType`` member) when the
  disassembler reports the operand itself is an FP **immediate** (the
  value at hand IS the IEEE bit pattern) — triggers step 1's inline-FP
  emission. Caller passes ``fp_postfix_type`` when the disassembler
  reports the **load instruction** is FP-typed for an address-bearing
  operand — triggers a postfix ``floatXX`` annotation appended after the
  ptr token emitted by steps 7–10. Two separate signals because precedence
  step 1 fires regardless of ``is_arithmetic`` (per the plan and
  precedence.md), so a single combined flag would conflate the two cases.

Identity allocation (per-function category counters) goes through
``TokenResolver.get_identity(Category.*, addr, meta_dict)`` — the
accumulated ``meta_dict`` becomes the per-function metadata JSON consumed
by Phase 2.A.1 (CSV writer). The dict is built from typed view fields by
each emitter, never echoed from the lookup payload directly.

Removed in this rewrite (legacy v1 frequency-sort + metadata aggregation,
to be replaced in Phase 2.A.1 / 2.B.7):
- ``create_opaque_mapping``      → stub raises ``NotImplementedError``
- ``reorder_metadata_for_mapping`` → stub raises ``NotImplementedError``
- ``get_sorted_opaque_constants`` → stub raises ``NotImplementedError``
- ``get_metadata_list_by_opaque_id`` → stub raises ``NotImplementedError``
- ``get_metadata``               → stub raises ``NotImplementedError``
- ``get_usage_stats``            → stub raises ``NotImplementedError``

Known broken downstream callers (left for their phase to fix):
- ``tokenizer/opaque_remapping.py`` (uses ``reorder_metadata_for_mapping``)
- ``tokenizer/main_loop.py`` (uses ``create_opaque_mapping``,
  ``get_metadata_list_by_opaque_id``)
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from tokenizer.disasm.metadata import AddressKind, AddressMetadataView
from tokenizer.disasm.types import FpType
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import (
    BlockToken,
    Category,
    MemoryOperandSymbol,
    OpaqueConstToken,
    TokenResolver,
    Tokens,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# v2 precedence-list infrastructure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ctx:
    """Per-call context bundle, passed to every emitter predicate.

    Bundles the orthogonal signals the predicates discriminate on so that
    each predicate is a clean callable of ``(meta, value, ctx)`` and no
    positional argument shuffle leaks into the precedence list itself.
    ``meta`` is an ``AddressMetadataView`` (typed); ``value`` is the
    constant being processed (needed by the local_func vs. block
    discriminator at steps 3 and 4).
    """
    is_arithmetic: bool
    fp_immediate_type: Optional[FpType]
    fp_postfix_type: Optional[FpType]


# Predicate type. Returns True when the emitter should fire for this
# ``(meta, value, ctx)`` triple. Predicates are total — they look only at
# the typed view, the value, and the context flags; no side effects.
_Predicate = Callable[[Optional[AddressMetadataView], int, _Ctx], bool]


def _is_function_entry(meta: AddressMetadataView, value: int) -> bool:
    """True iff ``value`` is exactly the entry point of a function range.

    The provider reports a function range as ``[start_addr, end_addr)``
    with ``start_addr == entry``. Any value strictly inside the range is
    a ``block`` (step 4), not the entry.
    """
    return meta.start_addr is not None and value == meta.start_addr


# ---- Step predicates -----------------------------------------------------

# Step 1 fires whenever the caller signals "this value is an FP
# immediate". Per ``precedence.md``: top precedence, fires regardless of
# ``is_arithmetic`` ("an arithmetic FP immediate is a value, but it is a
# floatXX value, not an integer valued_const"). The task pseudocode
# included ``not is_arithmetic`` here — deviation documented in the
# commit message because precedence.md is canonical.
def _pred_fp_immediate(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return ctx.fp_immediate_type is not None


# Steps 2–10 are gated on ``not is_arithmetic`` per the plan's
# is_arithmetic short-circuit ("steps 2–10 are skipped" for arithmetic
# operands; step 11 catches them). They also short-circuit when ``meta``
# is None (no lookup performed); only step 11 fires in that case.
def _pred_plt_func(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return not ctx.is_arithmetic and meta is not None and meta.kind == AddressKind.PLT_FUNCTION


def _pred_local_func(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 3: address EQUALS function entry in main object.
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.LOCAL_FUNCTION
        and _is_function_entry(meta, value)
    )


def _pred_block(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 4: address STRICTLY INSIDE a function in main object.
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.LOCAL_FUNCTION
        and not _is_function_entry(meta, value)
    )


def _pred_ext_func_real(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 5: extern function, NOT a synthetic stub.
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.EXT_FUNCTION_REAL
    )


def _pred_ext_func_synthetic(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    # Step 6: extern function, IS a synthetic stub.
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind == AddressKind.EXT_FUNCTION_SYNTHETIC
    )


def _pred_string_ptr(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return not ctx.is_arithmetic and meta is not None and meta.kind == AddressKind.STRING


def _pred_slot_with_modifier(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    if ctx.is_arithmetic or meta is None:
        return False
    # Vtables surface as CODE_PTR_TABLE_SLOT (precedence above the bare
    # rodata/data kinds). ``is_vtable`` is also exposed as a separate
    # boolean for the modifier-selection step in the emitter.
    return meta.kind in {AddressKind.CODE_PTR_TABLE_SLOT, AddressKind.JUMP_TABLE_SLOT} or meta.is_vtable


def _pred_ro_data_ptr(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return not ctx.is_arithmetic and meta is not None and meta.kind == AddressKind.RODATA


def _pred_rw_data_ptr(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return (
        not ctx.is_arithmetic
        and meta is not None
        and meta.kind in {AddressKind.DATA, AddressKind.BSS, AddressKind.THREAD_LOCAL_DATA}
    )


def _pred_fallback(meta: Optional[AddressMetadataView], value: int, ctx: _Ctx) -> bool:
    return True


# --------------------------------------------------------------------------
# ConstantHandler — owner of v1 legacy state + v2 emitters
# --------------------------------------------------------------------------


class ConstantHandler:
    """Handles constant value processing and token creation.

    v1 state (``opaque_const_tokens``, ``opaque_const_usage``,
    ``opaque_metadata``) is preserved while ``process_constant`` (the
    v1 entry point) is still callable. v2 emission does not touch any
    of that state — identity goes through ``TokenResolver`` and metadata
    through the resolver's per-category ``metadata`` lists.
    """

    def __init__(
        self,
        vocab_manager: VocabularyManager,
        resolver: TokenResolver,
        constant_dict: Dict[str, List[str]],
        block_ranges: np.ndarray,
    ):
        self.vocab_manager = vocab_manager
        self.resolver = resolver
        self.constant_dict = constant_dict
        self.block_ranges = block_ranges

        # v1 legacy state — kept until Phase 1.C.3 migrates all callers
        # to the v2 entry point and the v1 ``process_constant`` is removed.
        self.opaque_const_usage: Dict[int, int] = {}
        self.opaque_const_tokens: Dict[int, Tokens] = {}
        self.opaque_metadata: Dict[int, Tuple] = {}
        self.block_tokens: Dict[int, Tokens] = {}

        # v2 precedence table — literal ordered list of (predicate, emitter
        # method name) pairs. First match wins. Emitters are bound methods
        # on ``self`` resolved at dispatch time so the table stays purely
        # declarative (no instance reference baked in).
        self._precedence: List[Tuple[_Predicate, str]] = [
            # 1. FP immediate (provider-authoritative; top precedence)
            (_pred_fp_immediate,       "_emit_fp_immediate"),
            # 2. PLT stub
            (_pred_plt_func,           "_emit_plt_func"),
            # 3. Local function entry
            (_pred_local_func,         "_emit_local_func"),
            # 4. Block inside local function
            (_pred_block,              "_emit_block"),
            # 5. Real extern function entry
            (_pred_ext_func_real,      "_emit_ext_func_real"),
            # 6. Synthetic extern (CLE) function entry
            (_pred_ext_func_synthetic, "_emit_ext_func_synthetic"),
            # 7. String pointer
            (_pred_string_ptr,         "_emit_string_ptr"),
            # 8. Vtable / code_ptr_table / jump_table slot with modifier
            (_pred_slot_with_modifier, "_emit_slot_with_modifier"),
            # 9. Read-only data pointer
            (_pred_ro_data_ptr,        "_emit_ro_data_ptr"),
            # 10. Read-write data pointer (+ thread_local prefix when tls)
            (_pred_rw_data_ptr,        "_emit_rw_data_ptr"),
            # 11. Fallback — raw value
            (_pred_fallback,           "_emit_valued_const"),
        ]

    # ----------------------------------------------------------------------
    # v2 entry point
    # ----------------------------------------------------------------------

    def process_constant_v2(
        self,
        value: int,
        *,
        meta: Optional[AddressMetadataView] = None,
        is_arithmetic: bool = False,
        fp_immediate_type: Optional[FpType] = None,
        fp_postfix_type: Optional[FpType] = None,
    ) -> List[Tokens]:
        """Resolve a constant into v2 tokens following ``precedence.md``.

        Args:
            value: the immediate's raw integer value (signed Python int).
                Negative values reach step 11 (``valued_const``) which
                emits ``[MEM_MINUS, valued_const_v2(abs(value))]`` per
                the v2 sign convention (memory file
                ``open_design_v2_negative_valued_const.md`` option 2).
            meta: typed ``AddressMetadataView`` returned by
                ``MetadataLookup.lookup(addr)``. May be ``None`` when the
                caller hasn't done a lookup; the fallback emitter still
                handles that.
            is_arithmetic: when True, address-classification steps 2–10
                are skipped (the value is a pure arithmetic operand).
                Step 1 (FP immediate) and step 11 (valued_const) still
                fire — see precedence.md "is_arithmetic short-circuit".
            fp_immediate_type: typed ``FpType`` when the operand IS an FP
                immediate (the value is the IEEE bit pattern). Triggers
                step 1's inline-FP emission. ``None`` skips step 1.
            fp_postfix_type: typed ``FpType`` when the operand is an
                address with FP-typed load instruction. Appended as a
                postfix annotation after the ptr token emitted by
                steps 7–10 (no inline digits).
        """
        ctx = _Ctx(
            is_arithmetic=bool(is_arithmetic),
            fp_immediate_type=fp_immediate_type,
            fp_postfix_type=fp_postfix_type,
        )

        for predicate, emitter_name in self._precedence:
            if predicate(meta, value, ctx):
                return getattr(self, emitter_name)(value, meta, ctx)

        # Unreachable: _pred_fallback returns True unconditionally.
        raise AssertionError("v2 precedence table is missing a fallback rule")

    # ----------------------------------------------------------------------
    # v2 emitters — one per precedence step
    # ----------------------------------------------------------------------

    def _fp_factory(self, fp_type: FpType):
        """Map ``FpType`` to the v2 Inner-class factory.

        BFloat16 is reachable: the Ghidra provider reclassifies width=2
        FP operands to ``FpType.BFLOAT16`` based on the per-ISA mnemonic
        tables in ``ghidra_provider.py`` (ARM ``BFCVT``/``BFDOT``/...,
        x86 ``VCVTNE2PS2BF16``/...). Other widths map directly through
        the ``FpType`` -> Inner-class table below.
        """
        vm = self.vocab_manager
        return {
            FpType.FLOAT16:  vm.Float16,
            FpType.BFLOAT16: vm.BFloat16,
            FpType.FLOAT32:  vm.Float32,
            FpType.FLOAT64:  vm.Float64,
            FpType.FLOAT80:  vm.Float80,
            FpType.FLOAT128: vm.Float128,
        }[fp_type]

    def _emit_fp_immediate(self, value: int, meta: Optional[AddressMetadataView], ctx: _Ctx) -> List[Tokens]:
        """Step 1: operand IS the FP bit pattern → inline ``floatXX``."""
        factory = self._fp_factory(ctx.fp_immediate_type)
        return [factory(value)]

    def _emit_plt_func(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 2: PLT stub address → ``plt_func`` identity."""
        emitter_meta = {
            "name": meta.name,
            "library": meta.library,
            "addr": hex(value),
        }
        ident = self.resolver.get_identity(Category.PLT_FUNC, value, emitter_meta)
        return [self.vocab_manager.Plt_Func(ident)]

    def _emit_local_func(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 3: function entry in main object → ``local_func`` identity."""
        emitter_meta = {
            "name": meta.name,
            "addr": hex(value),
        }
        ident = self.resolver.get_identity(Category.LOCAL_FUNC, value, emitter_meta)
        return [self.vocab_manager.Local_Func(ident)]

    def _emit_block(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 4: address inside a function body → ``block`` identity.

        ``Block_V2`` carries a per-function block identity counter (resets
        on ``TokenResolver.reset_function``). No human-readable metadata
        is recorded — blocks are intra-function and the offset can be
        recovered from the function's own block ranges if a downstream
        consumer needs it.
        """
        ident = self.resolver.get_identity(Category.BLOCK, value, {})
        return [self.vocab_manager.Block_V2(ident)]

    def _emit_ext_func_real(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 5: real function entry in another loaded object."""
        emitter_meta = {
            "name": meta.name,
            "library": meta.library,
            "synthetic": False,
        }
        ident = self.resolver.get_identity(Category.EXT_FUNC, value, emitter_meta)
        return [self.vocab_manager.Ext_Func(ident)]

    def _emit_ext_func_synthetic(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 6: synthetic extern-object slot."""
        emitter_meta = {
            "name": meta.name,
            "library": meta.library,
            "synthetic": True,
        }
        ident = self.resolver.get_identity(Category.EXT_FUNC, value, emitter_meta)
        return [self.vocab_manager.Ext_Func(ident)]

    def _emit_string_ptr(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 7: provider-confirmed string → ``string_ptr`` identity.

        The metadata entry references ``{line, start_offset, encoding}``
        in the per-binary ``_strings.bin`` sidecar (Phase 2.A.2). This
        emitter records ``line=-1, start_offset=-1`` as placeholders
        and passes ``_string_bytes`` / ``_string_encoding`` through under
        underscore-prefixed keys so the downstream sidecar writer can
        register the bytes and rewrite the placeholders to real line
        numbers.
        """
        # ``_start_addr`` is the string's base address from the lookup
        # metadata (NOT ``value`` itself — ``value`` may be a substring
        # access pointing N bytes into the string). The CSV writer in
        # Phase 2.A.1 computes ``start_offset = value - _start_addr``
        # before stripping the underscore-prefixed internal keys.
        emitter_meta = {
            "line": -1,
            "start_offset": -1,
            "encoding": meta.string_encoding,
            "_string_bytes": meta.string_bytes,
            "_string_encoding": meta.string_encoding,
            "_start_addr": meta.start_addr,
            "addr": hex(value),
        }
        ident = self.resolver.get_identity(Category.STRING_PTR, value, emitter_meta)
        tokens: List[Tokens] = [self.vocab_manager.String_Ptr(ident)]
        # Postfix FP annotation: a string ptr loaded as FP is highly
        # unusual but the rule applies uniformly to steps 7–10.
        tokens.extend(self._postfix_fp_annotation(ctx))
        return tokens

    def _emit_slot_with_modifier(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 8: code-pointer-array slot → modifier + decomposed target.

        Without slot-target resolution available from the providers today,
        we always decompose into the base-pointer + offset form documented
        in precedence.md: ``[<modifier> <ro_data_ptr> <valued_const_v2>]``
        where ``ro_data_ptr`` identifies the table's base (the slot's
        containing range start) and ``valued_const_v2`` is the slot
        offset. When a future provider enrichment exposes the resolved
        target, this emitter swaps the decomposition for the resolved
        token.
        """
        # Pick the modifier — vtable beats code_ptr_table beats jump_table
        # for stable ordering. precedence.md treats jump_table slots under
        # step 8 with the ``code_ptr_table`` modifier semantically (a
        # jump table is also a code-pointer array), so the precedence
        # collapses to vtable-vs-code_ptr_table at emission time.
        if meta.is_vtable:
            modifier = self.vocab_manager.Vtable()
        else:
            # CODE_PTR_TABLE_SLOT or JUMP_TABLE_SLOT
            modifier = self.vocab_manager.Code_Ptr_Table()

        # Decompose: emit a ro_data_ptr for the base + valued_const for
        # the in-range offset. ``start_addr`` is the slot table's base.
        # If meta lacks a start_addr (unusual for step-8 enrichments), the
        # base IS the value and the offset is zero.
        base_addr = meta.start_addr if meta.start_addr is not None else value
        offset = value - base_addr

        base_emitter_meta = {
            "section": meta.name,
            "addr": hex(base_addr),
            "name": meta.name,
            "size": meta.size if meta.size is not None else 0,
        }
        base_ident = self.resolver.get_identity(
            Category.RO_DATA_PTR, base_addr, base_emitter_meta
        )
        tokens: List[Tokens] = [
            modifier,
            self.vocab_manager.Ro_Data_Ptr(base_ident),
        ]
        if offset != 0:
            tokens.append(self.vocab_manager.Valued_Const_V2(offset))
        tokens.extend(self._postfix_fp_annotation(ctx))
        return tokens

    def _emit_ro_data_ptr(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 9: read-only data pointer → ``ro_data_ptr`` identity.

        Postfix FP annotation rule (precedence.md): if the load is
        FP-typed (``ctx.fp_postfix_type`` set), append a ``floatXX``
        token with no inline payload — the actual value lives at the
        pointed-to address.
        """
        emitter_meta = {
            "section": meta.name,
            "addr": hex(value),
            "name": meta.name,
            "size": meta.size if meta.size is not None else 0,
        }
        ident = self.resolver.get_identity(Category.RO_DATA_PTR, value, emitter_meta)
        tokens: List[Tokens] = [self.vocab_manager.Ro_Data_Ptr(ident)]
        tokens.extend(self._postfix_fp_annotation(ctx))
        return tokens

    def _emit_rw_data_ptr(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 10: data / bss / TLS pointer → ``rw_data_ptr``.

        TLS sections (``tls=True``) get a ``thread_local`` modifier prefix.
        """
        emitter_meta = {
            "section": meta.name,
            "addr": hex(value),
            "name": meta.name,
            "size": meta.size if meta.size is not None else 0,
            "tls": meta.tls,
        }
        ident = self.resolver.get_identity(Category.RW_DATA_PTR, value, emitter_meta)
        tokens: List[Tokens] = []
        if meta.tls:
            tokens.append(self.vocab_manager.Thread_Local())
        tokens.append(self.vocab_manager.Rw_Data_Ptr(ident))
        tokens.extend(self._postfix_fp_annotation(ctx))
        return tokens

    def _emit_valued_const(self, value: int, meta: Optional[AddressMetadataView], ctx: _Ctx) -> List[Tokens]:
        """Step 11: fallback → ``valued_const_v2``.

        Negative values use the MEM_MINUS-prefix convention (memory file
        ``open_design_v2_negative_valued_const.md`` option 2): emit
        ``[MEM_MINUS, valued_const_v2(abs(value))]``. The ``valued_const_v2``
        Inner class is unsigned-only so the absolute value goes inline;
        the sign is recovered from the MEM_MINUS prefix at decode time.
        Deliberately mirrors v1's negative-immediate handling and is
        open for user revisit.
        """
        if value < 0:
            return [
                self.vocab_manager.MemoryOperand(MemoryOperandSymbol.MINUS),
                self.vocab_manager.Valued_Const_V2(-value),
            ]
        return [self.vocab_manager.Valued_Const_V2(value)]

    def _postfix_fp_annotation(self, ctx: _Ctx) -> List[Tokens]:
        """Append a postfix ``floatXX`` annotation when the load is FP-typed.

        Per precedence.md "Postfix FP annotation rule": no inline digits
        — the bits=None branch of the ``_V2FloatInner`` mixin emits just
        the type id. Reader rule: a ``floatXX`` token followed by a
        token ≥ 256 (i.e., no inline digits) annotates the previous
        ptr token's load type.
        """
        if ctx.fp_postfix_type is None:
            return []
        factory = self._fp_factory(ctx.fp_postfix_type)
        return [factory(None)]

    # ----------------------------------------------------------------------
    # v1 legacy entry point — preserved verbatim for callers that haven't
    # migrated to ``process_constant_v2``. Phase 1.C.3 (task #10) replaces
    # every call site; this method goes away then.
    # ----------------------------------------------------------------------

    def process_constant(
        self,
        value: int,
        is_arithmetic: bool = False,
        meta: Optional[AddressMetadataView] = None,
        library_type: str = "unknown",
        insn_mnemonic: Optional[str] = None,
    ) -> List[Tokens]:
        """v1 legacy entry — heuristic-driven Block / Opaque / Valued_Const.

        See ``process_constant_v2`` for the v2 successor. The decision
        logic is byte-for-byte the pre-rewrite behavior; only the read
        API has been migrated from dict-shaped ``meta`` to typed
        ``AddressMetadataView`` (Phase D.3).
        """
        # Metadata ranges starting at 0 represent abstract constant
        # domains, not real memory segments — treat as arithmetic.
        if meta is not None and meta.start_addr == 0:
            is_arithmetic = True

        # Small-constant / arithmetic short-circuit (legacy 0..0xFF rule).
        if is_arithmetic or 0x00 <= value <= 0xFF or value in self.constant_dict:
            return [self.vocab_manager.Valued_Const(value)]

        match_mask = (self.block_ranges[:, 0] <= value) & (value < self.block_ranges[:, 1])
        if np.any(match_mask):
            idx = match_mask.nonzero()[0][0]
            if self.block_ranges[idx, 0] == value:
                return [self.vocab_manager.Block(idx)]
            return [
                self.vocab_manager.Block(idx),
                self.vocab_manager.MemoryOperand(MemoryOperandSymbol.PLUS),
                self.vocab_manager.Valued_Const(value - self.block_ranges[idx, 0]),
            ]
        return self._create_opaque_const_with_offset(value, meta, library_type, insn_mnemonic)

    # v1 mapping from typed ``AddressKind`` back to the legacy decision
    # vocabulary. Used by ``_create_opaque_const_with_offset`` to keep the
    # heuristic predicates (function-range guard, decomposable-type list)
    # readable without spelling out the AddressKind set inline.
    _V1_FUNCTION_KINDS = frozenset({
        AddressKind.LOCAL_FUNCTION,
        AddressKind.EXT_FUNCTION_REAL,
        AddressKind.EXT_FUNCTION_SYNTHETIC,
        AddressKind.PLT_FUNCTION,
        AddressKind.UNKNOWN,  # legacy "unknown_function" mapped to UNKNOWN
    })
    _V1_DECOMPOSABLE_KINDS = frozenset({
        AddressKind.DATA,
        AddressKind.RODATA,
        AddressKind.BSS,
        # legacy "code" was decomposable; AddressKind.UNKNOWN covers it
        # (provider mapping for "code" -> UNKNOWN per metadata.py).
        AddressKind.UNKNOWN,
    })

    def _create_opaque_const_with_offset(
        self,
        value: int,
        meta: Optional[AddressMetadataView] = None,
        library_type: str = "unknown",
        insn_mnemonic: Optional[str] = None,
    ) -> List[Tokens]:
        """Create an opaque constant token, decomposing into base+offset if pointing into a range."""
        if (
            meta is not None
            and meta.start_addr is not None
            and meta.end_addr is not None
            and value > meta.start_addr
        ):
            start_addr = meta.start_addr
            end_addr = meta.end_addr
            range_length = end_addr - start_addr
            offset = value - start_addr

            # Apply heuristics to decide if we should decompose
            should_decompose = True
            reason = ""

            # Heuristic 5: Don't decompose local_function / library_function /
            # unknown_function ranges (they should be exact).
            if meta.kind in self._V1_FUNCTION_KINDS:
                should_decompose = False
                reason = f"function range (kind={meta.kind.name})"

            # Heuristic 3: Call instructions should not be decomposed
            # (must point to function header).
            elif insn_mnemonic and insn_mnemonic.startswith("call"):
                should_decompose = False
                reason = "call instruction must point to function header"

            # Heuristic 1: Range cannot start at 0 (not in binary).
            elif start_addr == 0:
                should_decompose = False
                reason = "range starts at 0"

            # Heuristic 2: Range cannot be longer than 2^16.
            elif range_length > (1 << 16):
                should_decompose = False
                reason = f"range too large (length={range_length:#x} > 0x10000)"

            # Heuristic 6: Prefer decomposing data/rodata/bss sections.
            elif meta.kind not in self._V1_DECOMPOSABLE_KINDS:
                should_decompose = False
                reason = f"unexpected metadata kind: {meta.kind.name}"

            # If value points into the range (not at the start), consider decomposition.
            if start_addr < value < end_addr and should_decompose:
                insn_info = f" in {insn_mnemonic}" if insn_mnemonic else ""
                logger.debug(
                    f"Decomposing: range {start_addr:#x}-{end_addr:#x} (length={range_length:#x}, kind={meta.kind.name}) "
                    f"for target {value:#x}, offset={offset:#x}{insn_info}"
                )
                base_token = self._create_opaque_const(start_addr, meta, library_type)
                return [
                    self.vocab_manager.MemoryOperand.OPEN_BRACKET,
                    base_token,
                    self.vocab_manager.MemoryOperand.PLUS,
                    self.vocab_manager.Valued_Const(offset),
                    self.vocab_manager.MemoryOperand.CLOSE_BRACKET,
                ]
            elif start_addr < value < end_addr and not should_decompose:
                insn_info = f" in {insn_mnemonic}" if insn_mnemonic else ""
                logger.debug(
                    f"Skipping decomposition: range {start_addr:#x}-{end_addr:#x} (length={range_length:#x}, kind={meta.kind.name}) "
                    f"for target {value:#x}, offset={offset:#x}{insn_info}, reason: {reason}"
                )

        # Otherwise just create a simple opaque constant.
        return [self._create_opaque_const(value, meta, library_type)]

    def _create_opaque_const(
        self,
        value: int,
        meta: Optional[AddressMetadataView] = None,
        library_type: str = "unknown",
    ) -> Tokens:
        """Create an opaque constant token (v1)."""
        if value not in self.opaque_const_tokens:
            opaque_id = self.resolver.get_opaque_id(value)
            token = self.vocab_manager.Opaque_Const(opaque_id)
            self.opaque_const_tokens[value] = token
            self.opaque_const_usage[value] = 1

            if meta is not None:
                # Mirrors the v1 tuple shape; ``library_type`` is the
                # caller-supplied label (kept separate from the typed
                # view's ``library`` because v1 callers passed
                # heuristic library tags here).
                self.opaque_metadata[value] = (
                    hex(meta.start_addr) if meta.start_addr is not None else "",
                    hex(meta.end_addr) if meta.end_addr is not None else "",
                    meta.name,
                    meta.kind.name,
                    library_type,
                )
        else:
            self.opaque_const_usage[value] += 1

        return self.opaque_const_tokens[value]

    # ----------------------------------------------------------------------
    # Removed v1 aggregation surface — stubs raising NotImplementedError so
    # downstream callers (opaque_remapping.py, main_loop.py) fail loudly
    # rather than silently producing wrong output. Phase 2.A.1 / 2.B.7
    # rewires those callers to read the per-category metadata directly
    # off ``TokenResolver.metadata[category]``.
    # ----------------------------------------------------------------------

    def get_sorted_opaque_constants(self):
        raise NotImplementedError(
            "ConstantHandler.get_sorted_opaque_constants removed in v2 — "
            "v2 identity is monotonic per category (no usage-frequency sort). "
            "See Phase 2.A.1 (CSV writer) / 2.B.7 (opaque_remapping migration)."
        )

    def create_opaque_mapping(self):
        raise NotImplementedError(
            "ConstantHandler.create_opaque_mapping removed in v2 — "
            "v2 identity is monotonic per category (no usage-frequency sort). "
            "See Phase 2.B.7 (opaque_remapping migration)."
        )

    def get_usage_stats(self):
        raise NotImplementedError(
            "ConstantHandler.get_usage_stats removed in v2 — "
            "usage tracking is not part of the v2 design. "
            "See Phase 2.A.1 (CSV writer)."
        )

    def get_metadata(self):
        raise NotImplementedError(
            "ConstantHandler.get_metadata removed in v2 — "
            "v2 metadata lives in TokenResolver.metadata[Category.*]. "
            "See Phase 2.A.1 (CSV writer)."
        )

    def get_metadata_list_by_opaque_id(self):
        raise NotImplementedError(
            "ConstantHandler.get_metadata_list_by_opaque_id removed in v2 — "
            "v2 metadata is per-category, indexed by per-category identity. "
            "See Phase 2.A.1 (CSV writer)."
        )

    def reorder_metadata_for_mapping(self, opaque_mapping):
        raise NotImplementedError(
            "ConstantHandler.reorder_metadata_for_mapping removed in v2 — "
            "v2 has no frequency-sort reordering step. "
            "See Phase 2.B.7 (opaque_remapping migration)."
        )
