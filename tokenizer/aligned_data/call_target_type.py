"""Single source of truth for the ``call_target`` category enum.

A call_target on a function's section header records WHO the function
calls. The category — ``local`` (in-binary user function), ``plt``
(import stub), ``extern`` (resolved-extern body, e.g. inlined libc) —
is a property of the call site, not of the callee record. The same
underlying callee can appear under different categories from different
call sites; the section-bin layout (see
``matched_sections_bin.py``) keys call_target dedup on the
``(function_name_ptr, type)`` pair so a PLT stub ``foo`` and an
extern body ``foo`` from the same caller remain two distinct entries.

The integer values are wire-format: they ride in 2 bits of the
section_bin's per-call_target flags field. Do not reorder; do not
renumber.
"""

from __future__ import annotations

from enum import IntEnum


class CallTargetType(IntEnum):
    LOCAL = 0
    PLT = 1
    EXTERN = 2
