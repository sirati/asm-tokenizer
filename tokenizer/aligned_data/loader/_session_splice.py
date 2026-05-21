"""Decoded-splicer bridge for :class:`BinarySession`.

Single concern of this module: translate the session's per-arm
``(idx, version) -> FunctionData`` API into the
``(section_offset, arm) -> (DecodedFunction, Section)`` callbacks the
:mod:`tokenizer.aligned_data.loader.decoded.splice` walker consumes.

The walker is pure on its inputs (per Phase 3 design); on-disk layout
+ vocab introspection are session-owned, so we keep that wiring here
rather than in either :mod:`session` (which would balloon past the
file-size cap) or :mod:`decoded.splice` (which would couple the pure
walker to the session's I/O surface).

Exposed as a mixin :class:`_BinarySessionSpliceMixin` so the public
:py:meth:`splice_with_callees` method stays on :class:`BinarySession`
itself -- callers do not need to know about the split. Mixin
inheritance is purely additive: every attribute it reads
(``_vocab_manager``, ``_meta_get``, ``_load_matched_section_and_versions``,
``_load_unmatched_record_and_section``, ``_unmatched_record_slot_base``,
``_category_token_ids``, ``_number_token_ids``) is owned by
:class:`BinarySession` itself.

Plan reference: ``## Wiring into the existing loader``, ``## Locked-in
decisions`` items 7-13 (per-Category rebase, DAG-active-path cycle
keys, section-call_target ordering, ``(arm, section_offset)`` cycle
keys), and ``## Open questions`` item 1 (the inverse
section-offset -> idx helper that previously did not exist).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from tokenizer.tokens import Category, TokenType

from ..matched_sections_bin import Section
from .function_data import FunctionData

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .decoded.decoded_function import DecodedFunction


class _BinarySessionSpliceMixin:
    """Mixin providing the decoded-splice surface for :class:`BinarySession`.

    Every method reads attributes / methods that :class:`BinarySession`
    owns; this class deliberately holds no state of its own. The
    inheritance is a code-locality tool, not a polymorphic boundary.
    """

    # The mixin is consumed in a strict order: BinarySession declares
    # the underlying state + load helpers, this mixin adds the public
    # splice entry + the private bridge helpers. ``self`` is typed
    # ``Any`` inside the bodies because the concrete attributes live on
    # the subclass; declaring them here would duplicate the source of
    # truth and risk drift.

    # --- token-id cache ----------------------------------------------

    def _get_token_id_caches(  # type: ignore[no-untyped-def]
        self,
    ) -> Tuple[Dict[Category, int], Dict[TokenType, int]]:
        """Lazily resolve + cache the v2 Category + number-TokenType ids.

        Both maps are derived from ``self._vocab_manager`` via the
        :mod:`decoded.category_tokens` resolvers; the result is cached
        per session (cleared in ``BinarySession.__exit__``). Plain
        ``dict`` so no further lifecycle hooks are needed.
        """
        if self._category_token_ids is None or self._number_token_ids is None:
            from .decoded.category_tokens import (
                resolve_category_token_ids,
                resolve_number_token_ids,
            )
            self._category_token_ids = resolve_category_token_ids(
                self._vocab_manager
            )
            self._number_token_ids = resolve_number_token_ids(
                self._vocab_manager
            )
        return self._category_token_ids, self._number_token_ids

    # --- inverse section lookup --------------------------------------

    def _idx_for_section_offset(  # type: ignore[no-untyped-def]
        self, section_offset: int, arm: str
    ) -> Optional[int]:
        """Inverse lookup: BIN section offset -> per-arm idx.

        For ``arm="matched"`` the result is the per-function idx (the
        :py:meth:`BinarySession.load_matched` argument). For
        ``arm="unmatched"`` the result is the FIRST-RECORD idx of the
        section at ``section_offset`` (the conventional
        :py:meth:`BinarySession.load_unmatched` argument when callees
        carry only a section-level pointer; pass-2 keeps the BIN's
        ``function_section_ptr`` aligned with the first-record's owning
        section).

        Returns ``None`` when no section at ``section_offset`` exists
        in the requested arm (extern callee, missing-section demotion,
        or cross-arm pointer). Raises :class:`ValueError` on an
        unknown ``arm`` name -- the splicer is strict on its inputs.
        """
        if arm == "matched":
            ma = self._meta_get("matched_arm")
            starts = getattr(ma, "bin_starts", None) if ma is not None else None
            matches = _exact_section_matches(starts, section_offset)
            if matches is None or len(matches) == 0:
                return None
            if len(matches) > 1:
                # Matched ``bin_starts`` are emitted one-per-function in
                # encounter order; a duplicate offset would mean the
                # writer wrote two function entries at the same BIN
                # section -- corruption, flagged loudly.
                raise ValueError(
                    f"multiple matched sections at offset {section_offset}: "
                    f"indices {list(matches)}"
                )
            return int(matches[0])
        if arm == "unmatched":
            ua = self._meta_get("unmatched_arm")
            starts = (
                getattr(ua, "section_starts", None) if ua is not None else None
            )
            matches = _exact_section_matches(starts, section_offset)
            if matches is None or len(matches) == 0:
                return None
            if len(matches) > 1:
                raise ValueError(
                    f"multiple unmatched sections at offset "
                    f"{section_offset}: indices {list(matches)}"
                )
            section_idx = int(matches[0])
            # Convert section idx -> first-record idx for downstream
            # ``load_unmatched`` / ``_load_unmatched_for_splice``.
            return self._unmatched_record_slot_base(ua, section_idx)
        raise ValueError(f"unknown arm: {arm!r}")

    # --- per-arm load wrappers ---------------------------------------

    def _load_matched_for_splice(  # type: ignore[no-untyped-def]
        self, idx: int, version: int
    ) -> Tuple[FunctionData, Section, int]:
        """Resolve one matched function-version for splicing.

        ``idx`` is the per-function idx; ``version`` is the index into
        :py:attr:`MatchedFunction.versions`. Returns the single
        :class:`FunctionData`, the parsed :class:`Section`, and the
        section's BIN byte offset. Raises :class:`IndexError` if
        ``version`` is out of range -- the caller chose to splice a
        specific version, so silent wrap-around would hide the bug.
        """
        section, section_offset, matched = (
            self._load_matched_section_and_versions(idx)
        )
        if version < 0 or version >= len(matched.versions):
            raise IndexError(
                f"matched function idx={idx} ({matched.func_name!r}) has "
                f"{len(matched.versions)} versions; version {version} "
                "out of range"
            )
        return matched.versions[version], section, section_offset

    def _load_unmatched_for_splice(  # type: ignore[no-untyped-def]
        self, idx: int
    ) -> Tuple[FunctionData, Section, int]:
        """Resolve one unmatched record for splicing.

        ``idx`` is the per-record idx (same input shape as
        :py:meth:`BinarySession.load_unmatched`). Returns the parsed
        :class:`FunctionData`, the owning :class:`Section`, and that
        section's BIN byte offset.
        """
        section, section_offset, fd = (
            self._load_unmatched_record_and_section(idx)
        )
        return fd, section, section_offset

    # --- public splice entry ----------------------------------------

    def splice_with_callees(  # type: ignore[no-untyped-def]
        self,
        idx: int,
        *,
        arm: str,
        max_depth: int,
        version: int = 0,
    ) -> "DecodedFunction":
        """Decode + splice the function at ``(arm, idx)`` with depth cap.

        Per plan ``## Locked-in decisions`` items 7, 8, 9, 13: per-
        ``Category`` identity rebase with sentinel handling; DAG-
        active-path cycle detection on ``(arm, section_offset)``;
        section-call_target ordering.

        For ``arm="matched"`` ``version`` selects the index into the
        :py:attr:`MatchedFunction.versions` list (default ``0`` -- the
        first version). For ``arm="unmatched"`` ``version`` is ignored
        (one record per FunctionData by construction); callers must
        pass the per-record idx the same way
        :py:meth:`BinarySession.load_unmatched` does.

        Callee recursion stays inside the same arm: the splicer's
        ``decode_callee`` callback resolves the callee's
        ``function_section_ptr`` via :py:meth:`_idx_for_section_offset`
        on the caller's arm. Matched callees of a matched caller
        always pick version ``0`` (the canonical first version --
        callees inherit no version-selection context from the caller).
        Cross-arm callees do not resolve and surface as
        ``is_callee_present=False``: their call-site tokens stay in
        the caller's stream but their bodies are not spliced.
        """
        from .decoded.extract import decode_raw_tokens
        from .decoded.splice import splice_with_callees as _splice_walker

        cat_ids, num_ids = self._get_token_id_caches()
        if arm == "matched":
            root_fd, root_section, root_offset = self._load_matched_for_splice(
                idx, version
            )
        elif arm == "unmatched":
            root_fd, root_section, root_offset = (
                self._load_unmatched_for_splice(idx)
            )
        else:
            raise ValueError(f"unknown arm: {arm!r}")

        root_decoded = decode_raw_tokens(
            root_fd.tokens,
            id_token_ids=cat_ids,
            number_token_ids=num_ids,
            func_name=root_fd.func_name,
            metadata=root_fd.metadata,
        )

        def _decode_callee(callee_offset: int, callee_arm: str):
            callee_idx = self._idx_for_section_offset(
                callee_offset, callee_arm
            )
            if callee_idx is None:
                # Guarded by ``_is_present`` upstream; defensive raise
                # here to fail loud if the walker ever calls us with
                # an absent callee.
                raise ValueError(
                    f"splice decode_callee called on absent "
                    f"({callee_arm}, {callee_offset}) -- "
                    "is_callee_present must gate this"
                )
            if callee_arm == "matched":
                fd, sec, _off = self._load_matched_for_splice(callee_idx, 0)
            else:
                fd, sec, _off = self._load_unmatched_for_splice(callee_idx)
            decoded = decode_raw_tokens(
                fd.tokens,
                id_token_ids=cat_ids,
                number_token_ids=num_ids,
                func_name=fd.func_name,
                metadata=fd.metadata,
            )
            return decoded, sec

        def _is_present(callee_offset: int, callee_arm: str) -> bool:
            return (
                self._idx_for_section_offset(callee_offset, callee_arm)
                is not None
            )

        return _splice_walker(
            root_decoded=root_decoded,
            root_arm=arm,
            root_section=root_section,
            root_section_offset=root_offset,
            decode_callee=_decode_callee,
            is_callee_present=_is_present,
            max_depth=max_depth,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _exact_section_matches(
    starts: Any, section_offset: int
) -> Optional[np.ndarray]:
    """Return ``np.where(starts == section_offset)[0]`` or ``None``.

    ``None`` covers both the "arm not present" + "arm has no
    sections" cases so the inverse-lookup caller can early-return.
    Kept module-level rather than mixin-static so call sites do not
    depend on a class scope for a stateless utility.
    """
    if starts is None or len(starts) == 0:
        return None
    return np.where(np.asarray(starts) == section_offset)[0]
