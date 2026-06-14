"""Per-call context bundle for v2 precedence-list dispatch.

Owns ``_Ctx`` (the orthogonal-signal bundle the predicates discriminate
on) + the ``_Predicate`` callable typedef.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from tokenizer.disasm.metadata import AddressMetadataView
from tokenizer.disasm.types import FpType


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
    # Dereferenced FP-postfix payload: the ``width_bytes`` raw image bytes
    # the caller read at the resolved load address, or ``None`` when the
    # value was unobtainable (unmapped / ``.bss`` / unreadable). Drives
    # ``_postfix_fp_annotation``: bytes -> valued ``floatXX``; ``None`` ->
    # value-less ``float_annotation`` marker. Always ``None`` when
    # ``fp_postfix_type`` is ``None`` (no FP postfix applies).
    fp_postfix_bytes: Optional[bytes] = None


# Predicate type. Returns True when the emitter should fire for this
# ``(meta, value, ctx)`` triple. Predicates are total -- they look only at
# the typed view, the value, and the context flags; no side effects.
_Predicate = Callable[[Optional[AddressMetadataView], int, _Ctx], bool]
