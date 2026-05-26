"""Per-binary semantic-deduplication of functions emitted by the tokenizer.

Concern
-------
Decide whether a function about to be emitted is the SAME logical
function as one already emitted in this run. The decision is the
four-way AND of:

1. Same ``func_name`` — the obvious group key.
2. Same ``comment`` — the provider-supplied *context* string (typically
   the demangled C++ scoped signature, e.g.
   ``ARPHeader::reset(...)``). Two distinct C++ methods sharing an
   unqualified name (``ARPHeader::reset`` vs ``EthernetHeader::reset``)
   surface as same-``name`` Ghidra ``Function``s with *different*
   plate comments; the comment is the natural disambiguator.
3. Same ``identity_key`` — the provider-supplied "stronger-than-name"
   identity (today: the thunked-function entry-point offset for PLT
   thunks). ``None`` when the provider declines.
4. Same ``tokens_base64`` body — the emitted token-body string. When
   ``comment`` AND ``identity_key`` are both ``None`` we still merge
   same-named functions whose body matches (the LTO-clone case: same
   static helper inlined into many TUs surfaces with identical body
   under one shared name).

When all four match this is a duplicate emission and downstream
should fold it (skip the CSV row, skip the FunctionDataManager record,
do not increment the occurrence counter). When any axis differs this
is a fresh function and downstream emits a new row / allocates a new
slot.

The body-divergence warning case (same name + same comment + same
identity_key but different body) is a rare diagnostic surface: the
first-recorded body wins as the canonical for the identity tuple;
the second call is reported as ``body_divergence_warning=True`` and
is NOT folded (it becomes a fresh record that the caller may suffix
or warn-log as it sees fit).

Module boundary
---------------
This module owns ONE concern: the four-axis merge decision. It does
not know about Ghidra, about PLT thunks, about Capstone, the CSV
writer, or the FDM slot table. Providers compute ``comment`` +
``identity_key`` (they know what counts as same-logical-function in
their disassembly model). Callers ask :meth:`FunctionDeduper.resolve`
once per function and act on the returned :class:`DedupResolution`.

A folded duplicate's resolution carries the FIRST recorded call's
``slot_id``: an opaque token the caller can use to look up its own
side-table entries (FDM array index, CSV final-name, etc.). The
deduper itself never assigns names; it only allocates slot_ids and
tracks which body each slot's canonical entry recorded.

Producer pipelines (the tokenizer main loop, the FunctionDataManager
in the verification path) hold their own ``FunctionDeduper`` instance
per binary. The provider's ``iter_functions`` sort puts colliding
names consecutively, so the deduper's memory is tight in practice
(same-name entries arrive back-to-back), but the implementation keeps
no ordering assumption — the keyed map handles any iteration order
correctly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Tuple


# ---------------------------------------------------------------------------
# Canonical name derivation
# ---------------------------------------------------------------------------
# The deduper's three identity axes (``name``, ``comment``, ``identity_key``)
# are the SAME inputs from which the final on-disk function name must be
# derived. The helper below collapses them into one deterministic string,
# so EVERY caller that needs a final function name (the FDM, the metadata
# lookup, any inspector) calls one function and gets one answer.
#
# Cross-ISA stability follows directly: the demangled C++ comment and the
# thunked-entry-offset identity_key are ISA-invariant by construction
# (the demangler emits the same signature for the same source-level symbol
# regardless of ISA; the thunked-function entry offset is the resolved
# external's address, identical across all trampoline slots that resolve
# to it AND identical across ISA-variant re-disassemblies of the same
# external).
#
# Sanitisation rule:
# - The comment is the C++ signature in human-readable form (e.g.
#   ``ARPHeader::storeRecvData(unsigned char const*, unsigned int)``).
#   We KEEP it readable; opaque hashes are awful to debug.
# - Whitespace runs collapse to a single ``_``.
# - Characters outside ``[A-Za-z0-9_.:<>()*&]`` get replaced with ``_``
#   (so commas, slashes, quotes, newlines can never end up in a CSV cell
#   or a sidecar line that callers expect to be one record).
# - Pathologically long suffixes (>200 chars after sanitisation) get
#   truncated to a 192-char prefix + ``~<sha1[:7]>`` so they still fit
#   in CSV cells and filesystem path components.
_CANONICAL_SUFFIX_LEN_CAP = 200
_CANONICAL_SUFFIX_TRUNC_PREFIX = 192
_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "_.:<>()*&"
)


def _sanitize_comment_suffix(comment: str) -> str:
    """Convert a comment string to a CSV/filename-safe suffix.

    The sanitisation rule is intentionally permissive on characters that
    appear naturally in C++ signatures (``::``, ``<>``, ``()``, ``*``,
    ``&``) so the suffix stays readable. Whitespace runs collapse to a
    single ``_``; everything else outside the allow-list also becomes
    ``_``. The result never contains characters that would corrupt a
    CSV cell, a sidecar line, or a filesystem path component (no
    commas, no newlines, no slashes, no quotes).
    """
    # Per-char replacement: allow-listed survives, everything else
    # (whitespace, punctuation, commas, quotes, slashes) maps to ``_``.
    # Runs of replacement underscores then collapse to a single ``_`` so
    # the suffix doesn't carry decorative ``__`` clusters from common
    # ``, `` (comma-space) shapes in demangled signatures.
    collapsed_chars: list[str] = []
    in_underscore_run = False
    for ch in comment:
        if ch in _ALLOWED_CHARS:
            collapsed_chars.append(ch)
            in_underscore_run = False
        else:
            if not in_underscore_run:
                collapsed_chars.append("_")
                in_underscore_run = True
    suffix = "".join(collapsed_chars)
    # Strip leading/trailing underscores so the suffix doesn't end with a
    # placeholder character; the demangled signature usually starts with
    # the qualified scope (``Class::method``) so this is rarely needed,
    # but it keeps the output tidy when the comment had leading whitespace.
    suffix = suffix.strip("_")
    if len(suffix) > _CANONICAL_SUFFIX_LEN_CAP:
        digest = hashlib.sha1(comment.encode("utf-8", errors="replace")).hexdigest()[:7]
        suffix = f"{suffix[:_CANONICAL_SUFFIX_TRUNC_PREFIX]}~{digest}"
    return suffix


def canonical_function_name(
    name: str,
    comment: Optional[str],
    identity_key: Optional[Hashable],
) -> str:
    """Produce the final on-disk function name from the three identity axes.

    Deterministic + cross-ISA-stable:

    * ``comment`` populated -> ``f"{name}@{sanitised_comment}"``. C++
      demangled signatures collide on unqualified ``name`` (e.g.
      ``ARPHeader::reset`` vs ``EthernetHeader::reset`` both surface as
      Ghidra ``Function``s with ``name=='reset'``); the demangled
      signature is the natural cross-ISA-stable disambiguator.
    * ``comment`` is None AND ``identity_key`` populated ->
      ``f"{name}@thunk:{identity_key}"``. PLT thunks share the resolved
      external's name; the resolved-entry-offset
      (``Function.getThunkedFunction(True).getEntryPoint().getOffset()``)
      is the cross-ISA-stable identity.
    * Both None -> ``name`` verbatim. The deduper's body-divergence
      diagnostic and the FDM's positional ``_N`` allocator are the only
      callers that touch this branch's downstream disambiguation (which
      is genuinely positional / per-binary, NOT cross-ISA-stable; the
      provider had no axes to assert otherwise).

    The output is safe for CSV cells, file paths, and the function-names
    sidecar (no commas, no newlines, no slashes, no quotes). Empty
    ``comment`` strings are treated as ``None`` so providers can pass
    ``""`` interchangeably with ``None``.
    """
    if comment == "":
        comment = None
    if comment is not None:
        suffix = _sanitize_comment_suffix(comment)
        return f"{name}@{suffix}"
    if identity_key is not None:
        return f"{name}@thunk:{identity_key}"
    return name


# ``@thunk:<digits>`` — the only canonical suffix that is NOT cross-ISA-
# stable in practice. The Ghidra-supplied identity_key for PLT thunks is
# ``getThunkedFunction(True).getEntryPoint().getOffset()``; for externals
# that offset is a placeholder address in the per-binary EXTERNAL block
# that varies across ELF variants (link order, library count, layout).
# The C++ comment-derived suffix is omitted on purpose: it's the
# demangled signature, which IS deterministic across builds.
_THUNK_SUFFIX_RE = re.compile(r"@thunk:\d+$")


def logical_function_name(canonical_name: str) -> str:
    """Strip the ``@thunk:<digits>`` suffix to recover the cross-binary
    logical function name.

    Consumers that need to group same-source-symbol functions across
    binaries (the inspector function list is the only current caller)
    feed every canonical name through this helper and use the result as
    the grouping key. Names without the suffix pass through verbatim,
    so callers don't need a fast-path branch.

    The comment-derived ``@<sanitized_signature>`` suffix from
    :func:`canonical_function_name` is intentionally preserved -- it is
    the demangled C++ signature and is deterministic across builds of
    the same source.
    """
    return _THUNK_SUFFIX_RE.sub("", canonical_name)


@dataclass(frozen=True)
class DedupResolution:
    """Outcome of :meth:`FunctionDeduper.resolve` for one function.

    Fields:

    * ``slot_id`` — opaque per-binary identifier of the canonical record
      for this identity tuple. Two calls returning the same ``slot_id``
      refer to the same logical function (folded). Callers maintain
      their own ``slot_id -> downstream_state`` side-tables.
    * ``is_duplicate`` — ``True`` iff this call folded into a prior
      resolution with the same ``(name, comment, identity_key, body)``
      tuple. Callers that fold (skip the row, skip the FDM slot) gate
      on this flag.
    * ``body_divergence_warning`` — ``True`` iff this call has the
      same ``(name, comment, identity_key)`` as a prior call but a
      DIFFERENT body. The first-recorded body wins as canonical; this
      call is NOT folded (it spawns a fresh ``slot_id``). The flag is
      informational — callers that surface deduper diagnostics can
      log on it.
    """

    slot_id: int
    is_duplicate: bool
    body_divergence_warning: bool = False


class FunctionDeduper:
    """Per-binary stateful gate over the four-axis identity.

    Lifecycle: one instance per binary tokenization. ``resolve`` is
    called once per function in iteration order. The instance is not
    threadsafe (the tokenizer main loop is single-threaded).
    """

    __slots__ = ("_seen", "_next_slot_id")

    def __init__(self) -> None:
        # (name, comment, identity_key) -> (canonical_slot_id, body)
        # of the first accepted function with that identity tuple.
        # Subsequent matching-body calls fold into ``canonical_slot_id``
        # (return is_duplicate=True). Divergent-body calls surface a
        # warning and spawn a fresh ``slot_id`` (no fold).
        self._seen: Dict[
            Tuple[str, Optional[str], Optional[Hashable]],
            Tuple[int, str],
        ] = {}
        # Monotonic counter; slot_id allocation is the only ordering
        # signal the deduper exposes. Encounter-order across binaries
        # IS NOT a cross-ISA-stable property — callers that need a
        # cross-ISA-stable name should derive one from the SAME
        # ``(name, comment, identity_key)`` axes the deduper consumes
        # (the per-binary slot_id is correct ONLY within one binary).
        self._next_slot_id = 0

    def resolve(
        self,
        func_name: str,
        comment: Optional[str],
        identity_key: Optional[Hashable],
        tokens_base64: str,
    ) -> DedupResolution:
        """Resolve one function to its ``(slot_id, is_duplicate)`` pair.

        The four-way merge condition is documented at module level.
        Side effect on a first sighting: the identity tuple is recorded
        so a future matching call returns ``is_duplicate=True``.

        Args:
            func_name: The provider-reported function name.
            comment: The provider-reported plate comment (the C++
                scoped signature for C++ symbols; ``None`` otherwise).
                Empty strings are treated as ``None``.
            identity_key: The provider-reported stronger-than-name
                identity (e.g. thunked-offset for PLT thunks; ``None``
                otherwise).
            tokens_base64: The function's emitted token body, already
                serialised for the CSV row. Used verbatim as the body
                equality key.
        """
        # Normalise empty comment to None so callers can pass either
        # shape without semantic difference (the demangler may emit
        # ``""`` for some edge cases).
        if comment == "":
            comment = None

        key = (func_name, comment, identity_key)
        recorded = self._seen.get(key)
        if recorded is None:
            # First sighting under this identity tuple.
            slot_id = self._next_slot_id
            self._next_slot_id += 1
            self._seen[key] = (slot_id, tokens_base64)
            return DedupResolution(slot_id=slot_id, is_duplicate=False)

        recorded_slot, recorded_body = recorded
        if recorded_body == tokens_base64:
            # Four-way match — fold into the canonical record.
            return DedupResolution(slot_id=recorded_slot, is_duplicate=True)

        # Same identity tuple but DIFFERENT body — body divergence.
        # The first-recorded body wins as canonical; this call gets a
        # fresh slot_id but a warning flag the caller surfaces.
        # First-recorded body stays canonical for ANY future
        # identical-body fold attempt under this same tuple.
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        return DedupResolution(
            slot_id=slot_id,
            is_duplicate=False,
            body_divergence_warning=True,
        )


__all__ = (
    "DedupResolution",
    "FunctionDeduper",
    "canonical_function_name",
    "logical_function_name",
)
