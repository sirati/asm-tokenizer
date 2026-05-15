"""Constant tokenization -- v1 legacy path + v2 precedence-list classifier.

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
  (``meta.name``, ``meta.start_addr``, ...) -- ConstantHandler reads them
  exclusively (no dict access, no string-keyed lookups).
- Caller passes ``is_arithmetic=True`` when the operand context is
  arithmetic (the value being arithmetically combined, not an address
  dereference candidate). Per ``precedence.md`` the address steps 2-10
  short-circuit; step 1 (disassembler-reported FP type) and step 11
  (``valued_const``) still apply.
- Caller passes ``fp_immediate_type`` (an ``FpType`` member) when the
  disassembler reports the operand itself is an FP **immediate** (the
  value at hand IS the IEEE bit pattern) -- triggers step 1's inline-FP
  emission. Caller passes ``fp_postfix_type`` when the disassembler
  reports the **load instruction** is FP-typed for an address-bearing
  operand -- triggers a postfix ``floatXX`` annotation appended after the
  ptr token emitted by steps 7-10. Two separate signals because precedence
  step 1 fires regardless of ``is_arithmetic`` (per the plan and
  precedence.md), so a single combined flag would conflate the two cases.

Identity allocation (per-function category counters) goes through
``TokenResolver.get_identity(Category.*, addr, meta_dict)`` -- the
accumulated ``meta_dict`` becomes the per-function metadata JSON consumed
by Phase 2.A.1 (CSV writer). The dict is built from typed view fields by
each emitter, never echoed from the lookup payload directly.

Removed in this rewrite (legacy v1 frequency-sort + metadata aggregation,
to be replaced in Phase 2.A.1 / 2.B.7):
- ``create_opaque_mapping``      -> stub raises ``NotImplementedError``
- ``reorder_metadata_for_mapping`` -> stub raises ``NotImplementedError``
- ``get_sorted_opaque_constants`` -> stub raises ``NotImplementedError``
- ``get_metadata_list_by_opaque_id`` -> stub raises ``NotImplementedError``
- ``get_metadata``               -> stub raises ``NotImplementedError``
- ``get_usage_stats``            -> stub raises ``NotImplementedError``

Known broken downstream callers (left for their phase to fix):
- ``tokenizer/opaque_remapping.py`` (uses ``reorder_metadata_for_mapping``)
- ``tokenizer/main_loop.py`` (uses ``create_opaque_mapping``,
  ``get_metadata_list_by_opaque_id``)

Submodule layout (single-concern split, task #63):
- ``ctx``: ``_Ctx`` per-call context + ``_Predicate`` typedef.
- ``predicates``: ``_pred_*`` step predicates + ``_is_function_entry``.
- ``emitters_v2``: ``_V2EmittersMixin`` (per-precedence-step emitters +
  FP factory + postfix annotation).
- ``legacy_v1``: ``_V1LegacyMixin`` (``process_constant`` + opaque-const
  helpers + removed-aggregation stubs).
- ``core``: ``ConstantHandler`` composed class + v2 entry point +
  precedence-list registration.
"""

from tokenizer.constant_handler.core import ConstantHandler

__all__ = ["ConstantHandler"]
