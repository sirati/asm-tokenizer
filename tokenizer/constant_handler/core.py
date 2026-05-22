"""``ConstantHandler`` core class + v2 entry point + precedence-list registration.

Owns:
- ``ConstantHandler`` -- composed of ``_V1LegacyMixin`` (legacy
  ``process_constant`` + opaque-const helpers + removed-aggregation stubs)
  and ``_V2EmittersMixin`` (per-precedence-step emitters + FP factory +
  postfix annotation).
- ``__init__``: builds the v1 in-memory state + v2 precedence table.
- ``process_constant_v2``: the v2 entry point that walks the precedence
  list (first match wins) and dispatches to the matching emitter.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from tokenizer.constant_handler.ctx import _Ctx, _Predicate
from tokenizer.constant_handler.emitters_v2 import _V2EmittersMixin
from tokenizer.constant_handler.legacy_v1 import _V1LegacyMixin
from tokenizer.constant_handler.predicates import (
    _pred_block,
    _pred_ext_func_real,
    _pred_ext_func_synthetic,
    _pred_fallback,
    _pred_fp_immediate,
    _pred_jump_table_slot,
    _pred_local_func,
    _pred_plt_func,
    _pred_ro_data_ptr,
    _pred_rw_data_ptr,
    _pred_slot_with_modifier,
    _pred_string_ptr,
)
from tokenizer.disasm.metadata import AddressMetadataView
from tokenizer.disasm.types import FpType
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenResolver, Tokens


# --------------------------------------------------------------------------
# ConstantHandler -- owner of v1 legacy state + v2 emitters
# --------------------------------------------------------------------------


class ConstantHandler(_V1LegacyMixin, _V2EmittersMixin):
    """Handles constant value processing and token creation.

    v1 state (``opaque_const_tokens``, ``opaque_const_usage``,
    ``opaque_metadata``) is preserved while ``process_constant`` (the
    v1 entry point) is still callable. v2 emission does not touch any
    of that state -- identity goes through ``TokenResolver`` and metadata
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

        # v1 legacy state -- kept until Phase 1.C.3 migrates all callers
        # to the v2 entry point and the v1 ``process_constant`` is removed.
        self.opaque_const_usage: Dict[int, int] = {}
        self.opaque_const_tokens: Dict[int, Tokens] = {}
        self.opaque_metadata: Dict[int, Tuple] = {}
        self.block_tokens: Dict[int, Tokens] = {}

        # v2 precedence table -- literal ordered list of (predicate, emitter
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
            # 8a. Jump-table slot -- own emitter (no modifier; bridges to
            # the function-level footer via shared Category.JUMP_TABLE
            # identity). MUST appear before the slot-with-modifier rule.
            (_pred_jump_table_slot,    "_emit_jump_table_slot"),
            # 8b. Vtable / code_ptr_table slot with modifier
            (_pred_slot_with_modifier, "_emit_slot_with_modifier"),
            # 9. Read-only data pointer
            (_pred_ro_data_ptr,        "_emit_ro_data_ptr"),
            # 10. Read-write data pointer (+ thread_local prefix when tls)
            (_pred_rw_data_ptr,        "_emit_rw_data_ptr"),
            # 11. Fallback -- raw value
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
                emits ``[valued_const_v2(|value|), value_negative]`` --
                the v2 emitter is the sole owner of sign decomposition
                (postfix ``value_negative`` metatoken, see
                ``_emit_valued_const``).
            meta: typed ``AddressMetadataView`` returned by
                ``MetadataLookup.lookup(addr)``. May be ``None`` when the
                caller hasn't done a lookup; the fallback emitter still
                handles that.
            is_arithmetic: when True, address-classification steps 2-10
                are skipped (the value is a pure arithmetic operand).
                Step 1 (FP immediate) and step 11 (valued_const) still
                fire -- see precedence.md "is_arithmetic short-circuit".
            fp_immediate_type: typed ``FpType`` when the operand IS an FP
                immediate (the value is the IEEE bit pattern). Triggers
                step 1's inline-FP emission. ``None`` skips step 1.
            fp_postfix_type: typed ``FpType`` when the operand is an
                address with FP-typed load instruction. Appended as a
                postfix annotation after the ptr token emitted by
                steps 7-10 (no inline digits).
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
