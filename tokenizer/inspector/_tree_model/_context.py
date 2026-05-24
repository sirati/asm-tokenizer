"""Per-FunctionNode decode-context bundle threaded to descendants.

The :class:`DecodeContext` ties together the per-FunctionNode batch-
decode artefacts that every descendant node (variant, block, inline-
call, inline-jump) needs to render. Lifetime equals the parent
FunctionNode's :class:`BatchDecodeResult`; the dataclass holds only
plain references to numpy views + mappings + a typed resolver
callable.

Also owns the session-side helper that fetches ``line_to_name`` off
the public :class:`BinarySession.get_metadata` accessor -- a thin
shim so the rest of the tree model never reaches into session
internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.metadata_loader import SectionKind


if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.session import BinarySession
    from tokenizer.token_manager import VocabularyManager


__all__ = [
    "DecodeContext",
    "session_line_to_name",
]


@dataclass(frozen=True)
class DecodeContext:
    """Per-FunctionNode batch-decode context threaded to descendants.

    Lifetime equals the parent FunctionNode's ``BatchDecodeResult``;
    holds only plain references to numpy views + mappings, plus the
    section-arm tag (:class:`SectionKind`) so descendant render calls
    have the parent's arm without re-deriving it, plus the
    :attr:`callee_arm_resolver` closure that maps a section byte offset
    (the ``function_section_ptr`` field on a LOCAL/PLT call_target) to
    the matching :class:`SectionPointerSpec`. The resolver is built
    once at :meth:`FunctionNode.expand` time, binding the session +
    arm, so descendant nodes never reach into session internals.
    """

    arm: SectionKind
    fid_sidecar: np.ndarray | None
    fid_row_offsets: np.ndarray | None
    line_to_name: Mapping[int, str]
    vocab_manager: "VocabularyManager"
    callee_arm_resolver: Callable[[int], SectionPointerSpec | None]


def session_line_to_name(session: "BinarySession") -> Mapping[int, str]:
    """Extract ``line_to_name`` from a session via its public metadata
    accessor. Empty mapping when absent -- name resolution then falls
    back to ``"?"`` per plan D4. ``session`` is always a live session
    per the UI contract: :meth:`FunctionNode.expand` only runs after
    the inspector has opened one, so a ``None`` guard would mask a
    contract violation rather than handle a real case.
    """
    return session.get_metadata("line_to_name") or {}
