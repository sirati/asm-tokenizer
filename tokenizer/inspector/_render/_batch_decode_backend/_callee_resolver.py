"""Per-call FUNCTION-category callee resolver.

Single concern: map ``(call_kind, counter_id)`` to a
:class:`SectionPointerSpec` (or ``None``) via the current
call_target's ``call_targets_section`` table + a caller-supplied
session-backed ``callee_arm_resolver`` closure. Mirrors
:func:`tokenizer.inspector._render._render_block._emit_call_entry`'s
typed dispatch so both rendering backends share the same resolution
semantics.

Lifted out of :mod:`._row_walk` so the resolver algorithm stays small
+ unit-testable; the walker only calls the public
:func:`resolve_callee_pointer`.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.matched_sections_bin import CallTarget


__all__ = ["resolve_callee_pointer"]


def resolve_callee_pointer(
    *,
    call_kind: CallTargetType,
    counter: int,
    call_targets_section: Sequence[CallTarget],
    kind_to_called_idx: Mapping[CallTargetType, list[int]],
    callee_arm_resolver: Callable[[int], Optional[SectionPointerSpec]],
) -> Optional[SectionPointerSpec]:
    """Map ``(call_kind, counter)`` to a section pointer.

    For LOCAL / PLT calls, reads
    ``call_targets_section[kind_to_called_idx[kind][counter]].function_section_ptr``
    and passes it through the session-backed ``callee_arm_resolver``;
    EXTERN calls always return ``None`` (no body to inline).

    Returns ``None`` when:

    * The kind is EXTERN.
    * The ``(kind, counter)`` pair is out of range for the partition
      table (defensive: a corrupt counter / partition surfaces as
      "non-expandable" rather than crashing).
    * The resolver returns ``None`` (cross-arm pointer or missing-
      section demotion).
    """
    if call_kind is CallTargetType.EXTERN:
        # No body to inline; the UI hides expansion on None.
        return None
    indices_for_kind = kind_to_called_idx.get(call_kind)
    if indices_for_kind is None or counter >= len(indices_for_kind):
        return None
    called_idx = indices_for_kind[counter]
    if called_idx >= len(call_targets_section):
        return None
    function_section_ptr = int(
        call_targets_section[called_idx].function_section_ptr
    )
    return callee_arm_resolver(function_section_ptr)
