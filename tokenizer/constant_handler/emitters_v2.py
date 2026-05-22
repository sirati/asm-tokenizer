"""v2 precedence-list emitters (mixin).

``_V2EmittersMixin`` holds one ``_emit_*`` method per precedence step
plus the FP-factory and postfix-annotation helpers. The class is
composed into ``ConstantHandler`` via subclassing in ``core.py``.

The mixin assumes the composed subclass exposes:
- ``self.vocab_manager`` (``VocabularyManager``)
- ``self.resolver`` (``TokenResolver``)
- ``self.process_constant_v2`` (recursive call for resolved slot targets)
"""

from __future__ import annotations

from typing import List, Optional

from tokenizer.constant_handler.ctx import _Ctx
from tokenizer.disasm.metadata import AddressMetadataView
from tokenizer.disasm.types import FpType
from tokenizer.tokens import Category, Tokens


class _V2EmittersMixin:
    """v2 precedence-list emitter implementations.

    All emitters take ``(value, meta, ctx)`` and return a ``list[Tokens]``.
    They consume the typed ``AddressMetadataView`` exclusively (no dict
    access, no string-keyed lookups).
    """

    # ----------------------------------------------------------------------
    # v2 emitters -- one per precedence step
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
        """Step 1: operand IS the FP bit pattern -> inline ``floatXX``."""
        factory = self._fp_factory(ctx.fp_immediate_type)
        return [factory(value)]

    def _emit_plt_func(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 2: PLT stub address -> ``plt_func`` identity."""
        emitter_meta = {
            "name": meta.name,
            "library": meta.library,
            "addr": hex(value),
        }
        ident = self.resolver.get_identity(Category.PLT_FUNC, value, emitter_meta)
        return [self.vocab_manager.Plt_Func(ident)]

    def _emit_local_func(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 3: function entry in main object -> ``local_func`` identity."""
        emitter_meta = {
            "name": meta.name,
            "addr": hex(value),
        }
        ident = self.resolver.get_identity(Category.LOCAL_FUNC, value, emitter_meta)
        return [self.vocab_manager.Local_Func(ident)]

    def _emit_block(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 4: address inside a function body -> ``block`` identity.

        ``Block_V2`` carries a per-function block identity counter (resets
        on ``TokenResolver.reset_function``). No human-readable metadata
        is recorded -- blocks are intra-function and the offset can be
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
        """Step 7: provider-confirmed string -> ``string_ptr`` identity.

        The metadata entry references ``{line, start_offset, encoding}``
        in the per-binary ``_strings.bin`` sidecar (Phase 2.A.2). This
        emitter records ``line=-1, start_offset=-1`` as placeholders
        and passes ``_string_bytes`` / ``_string_encoding`` through under
        underscore-prefixed keys so the downstream sidecar writer can
        register the bytes and rewrite the placeholders to real line
        numbers.
        """
        # ``_start_addr`` is the string's base address from the lookup
        # metadata (NOT ``value`` itself -- ``value`` may be a substring
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
        # unusual but the rule applies uniformly to steps 7-10.
        tokens.extend(self._postfix_fp_annotation(ctx))
        return tokens

    def _emit_jump_table_slot(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 8a: jump-table slot -> ``[jump_table(id), valued_const(offset)]``.

        Distinct from the vtable / code_ptr_table emission shape: NO
        modifier prefix; instead emits the same ``Category.JUMP_TABLE``
        identity used by the function-level footer in
        ``fill_constant_candidates._emit_jump_table_footer``. Sharing the
        identity bridges the two paths -- a slot accessed before the
        footer fires registers the table with the resolver, and the
        footer-fallback path (Phase E.1 unified-emit) picks it up so the
        function still emits a footer for the table.
        """
        jt_id = self.resolver.get_identity(
            Category.JUMP_TABLE,
            meta.jump_table_base_addr,
            {"jump_table_addr": hex(meta.jump_table_base_addr)},
        )
        return [
            self.vocab_manager.Jump_Table(jt_id),
            self.vocab_manager.Valued_Const_V2(meta.jump_table_offset),
            *self._postfix_fp_annotation(ctx),
        ]

    def _emit_slot_with_modifier(self, value: int, meta: AddressMetadataView, ctx: _Ctx) -> List[Tokens]:
        """Step 8b: code-pointer-array slot -> modifier + (resolved target | decomposed base+offset).

        Modifier selection: ``vtable`` beats ``code_ptr_table`` (vtable is
        the more specific signal -- see ``tokenizer/disasm/precedence.md``).

        Two emission shapes share this emitter, both prefixed by the
        modifier and post-fixed by the operand-level FP annotation:

        - **Resolved-target branch (Ghidra path, Phase D.1).** When the
          provider has pre-resolved ``meta.slot_target`` to the typed view
          of the *target* of the slot (the function/data the slot points
          to), recursively classify the target via ``process_constant_v2``
          and splice its tokens between the modifier and the postfix. The
          recursion is bounded: the provider guarantees
          ``slot_target.kind`` is never itself a slot kind, so the inner
          call cannot recurse back into this emitter.
        - **Decomposed fallback (angr path always; Ghidra path when the
          slot target could not be resolved -- e.g. null slot or out of
          any recognized region).** Emit a ``ro_data_ptr`` for the
          table's base + ``valued_const_v2`` for the in-range offset.
          ``start_addr`` is the slot table's base; if absent (unusual for
          step-8 enrichments), the base IS ``value`` and the offset is 0.

        Jump-table slots are handled by ``_emit_jump_table_slot`` (Phase
        E.1) and never reach this emitter -- the precedence list routes
        ``AddressKind.JUMP_TABLE_SLOT`` to the dedicated emitter first.

        Provider parity. Resolved-target emission is Ghidra-only by
        current provider capability; angr always leaves
        ``slot_target=None`` (see
        ``tokenizer/disasm/angr_limitations.md``) so the angr path always
        takes the decomposed fallback.
        """
        # Pick the modifier -- vtable beats code_ptr_table for stable
        # ordering (vtable is the more specific signal).
        if meta.is_vtable:
            modifier = self.vocab_manager.Vtable()
        else:
            # CODE_PTR_TABLE_SLOT
            modifier = self.vocab_manager.Code_Ptr_Table()

        # Resolved-target branch (Ghidra-only; angr always leaves
        # slot_target=None). Recursively classify the target itself --
        # bounded recursion: provider guarantees slot_target.kind is never
        # a slot kind. Pass fp_*_type=None because the recursive classify
        # is for the slot's referent, not the operand-level FP context;
        # the OUTER postfix annotation below preserves the original
        # operand-level FP postfix on the outside of the target tokens.
        if meta.slot_target is not None:
            target_value = meta.slot_target.start_addr or value
            target_tokens = self.process_constant_v2(
                value=target_value,
                meta=meta.slot_target,
                is_arithmetic=False,
                fp_immediate_type=None,
                fp_postfix_type=None,
            )
            return [modifier, *target_tokens, *self._postfix_fp_annotation(ctx)]

        # Decomposed fallback: emit a ro_data_ptr for the base +
        # valued_const for the in-range offset. ``start_addr`` is the
        # slot table's base. If meta lacks a start_addr (unusual for
        # step-8 enrichments), the base IS the value and the offset is
        # zero.
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
        """Step 9: read-only data pointer -> ``ro_data_ptr`` identity.

        Postfix FP annotation rule (precedence.md): if the load is
        FP-typed (``ctx.fp_postfix_type`` set), append a ``floatXX``
        token with no inline payload -- the actual value lives at the
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
        """Step 10: data / bss / TLS pointer -> ``rw_data_ptr``.

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
        """Step 11: fallback -> ``valued_const_v2``.

        Sole owner of sign handling for v2 valued_const. The
        ``ValuedConstV2Inner`` class is unsigned-only (its contract
        asserts ``value >= 0``), so the emitter decomposes a signed
        Python int into an unsigned magnitude + an optional postfix
        ``value_negative`` metatoken:

        - non-negative: ``[Valued_Const_V2(value), fp_postfix?]``
        - negative:     ``[Valued_Const_V2(|value|), Value_Negative(), fp_postfix?]``

        ``value_negative`` and the FP-postfix metatokens both have ids
        >= 256, so neither can be misread as a digit-stream byte by the
        v2 decoder (which terminates the inline-digit run on the first
        id >= 256). The decoder consumes them as separate metatokens.

        Sign-decomposition uses an explicit unary-minus rather than
        ``abs()`` for symmetry with the ``value < 0`` discriminant and
        to avoid relying on ``abs()`` for the (already-impossible-in-
        Python's arbitrary-precision ints) signed-overflow case.
        """
        abs_value = -value if value < 0 else value
        tokens: List[Tokens] = [self.vocab_manager.Valued_Const_V2(abs_value)]
        if value < 0:
            tokens.append(self.vocab_manager.Value_Negative())
        tokens.extend(self._postfix_fp_annotation(ctx))
        return tokens

    def _postfix_fp_annotation(self, ctx: _Ctx) -> List[Tokens]:
        """Append a postfix ``floatXX`` annotation when the load is FP-typed.

        Per precedence.md "Postfix FP annotation rule": no inline digits
        -- the bits=None branch of the ``_V2FloatInner`` mixin emits just
        the type id. Reader rule: a ``floatXX`` token followed by a
        token >= 256 (i.e., no inline digits) annotates the previous
        ptr token's load type.
        """
        if ctx.fp_postfix_type is None:
            return []
        factory = self._fp_factory(ctx.fp_postfix_type)
        return [factory(None)]
