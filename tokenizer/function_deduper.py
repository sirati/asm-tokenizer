"""Per-binary semantic-deduplication of functions emitted by the tokenizer.

Concern
-------
Decide whether a function about to be emitted is the SAME logical
function as one already emitted in this run. The decision is the
three-way AND of:

1. Same ``func_name`` — the obvious group key.
2. Same ``identity_key`` — a stable provider-supplied identity that is
   stronger than name. The disasm provider's ``FunctionView`` exposes
   this (see ``tokenizer/disasm/types.py``'s ``FunctionView.identity_key``
   contract). ``identity_key=None`` short-circuits the decision to
   "not a duplicate" — providers that cannot assert any identity
   beyond name (today: angr/Capstone) put every same-named function
   on the legacy occurrence-suffix disambiguation path.
3. Same content — the emitted token-body, threaded in as
   ``tokens_base64`` so the check is a single string equality (each
   function's tokens were already serialised to base64 for the CSV
   row build, no extra hashing on the hot path).

When all three match, this is a duplicate emission and downstream
should fold it (skip the CSV row, skip the FunctionDataManager record,
do not increment the occurrence counter). When any one fails, this is
a fresh function and downstream proceeds as before.

Module boundary
---------------
This module owns ONE concern: the merge decision. It does not know
about Ghidra, about PLT thunks, about Capstone, or about the CSV
writer. Providers compute the ``identity_key`` (they know what counts
as same-logical-function in their disassembly model). Callers ask
:meth:`FunctionDeduper.is_duplicate` and act on the boolean.

Producer pipelines (the tokenizer main loop, the FunctionDataManager
in the verification path) hold their own ``FunctionDeduper`` instance
per binary. The provider's ``iter_functions`` sort puts colliding
names consecutively, so the deduper's memory is tight in practice
(same-(name,key) entries arrive back-to-back), but the implementation
keeps no ordering assumption — the keyed map handles any iteration
order correctly.
"""

from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple


class FunctionDeduper:
    """Per-binary stateful gate: "is this (name, identity_key, content)
    a duplicate of something we already accepted?"

    Lifecycle: one instance per binary tokenization. ``is_duplicate``
    is called once per function in iteration order. The instance is
    not threadsafe (the tokenizer main loop is single-threaded).
    """

    __slots__ = ("_seen",)

    def __init__(self) -> None:
        # (name, identity_key) -> tokens_base64 of the first accepted
        # function with that key.
        self._seen: Dict[Tuple[str, Hashable], str] = {}

    def is_duplicate(
        self,
        func_name: str,
        identity_key: Optional[Hashable],
        tokens_base64: str,
    ) -> bool:
        """Return True iff this function is a semantic duplicate of one
        already recorded (same name, same identity_key, same content).

        Side effect on the False return: when the function is NOT a
        duplicate but DOES carry an identity_key, record it so a future
        matching call returns True. Functions with ``identity_key=None``
        are never recorded and never reported as duplicates — they
        flow through on the legacy disambiguation path.

        Args:
            func_name: The Ghidra/angr function name (the dedup
                trigger when same as a previous function).
            identity_key: Provider-supplied stronger-than-name
                identity, or ``None`` when the provider declines to
                assert one. See ``FunctionView.identity_key`` in
                ``tokenizer/disasm/types.py``.
            tokens_base64: The function's token body, already
                serialised for the CSV row. Used verbatim as the
                content-equality key.
        """
        if identity_key is None:
            return False
        key = (func_name, identity_key)
        recorded = self._seen.get(key)
        if recorded is None:
            self._seen[key] = tokens_base64
            return False
        if recorded == tokens_base64:
            return True
        # Same (name, identity_key) but DIFFERENT content. The
        # provider's identity assertion was insufficient (or the
        # function bodies diverged across ISA-variant disassembly).
        # Fall back to the legacy disambiguation path; do NOT
        # overwrite the recorded content (the first-accepted wins
        # as the canonical body for this identity).
        return False
