"""Decoded-splicer bridge for :class:`BinarySession`.

Single concern of this module: translate the session's per-arm
``(idx, variant_index) -> FunctionData`` API into the
``(section_offset, arm, callee_variant_index) -> (DecodedFunction,
Section)`` callbacks the :mod:`tokenizer.aligned_data.loader.decoded.splice`
walker consumes, and assemble the N selected primary variants into
the public ``list[DecodedFunction]`` return shape.

The walker is pure on its inputs (per Phase 3 design); on-disk layout
+ vocab introspection are session-owned, so we keep that wiring here
rather than in either :mod:`session` (which would balloon past the
file-size cap) or :mod:`decoded.splice` (which would couple the pure
walker to the session's I/O surface).

Both the root function and every callee are fed to
:func:`decode_raw_tokens` via
:py:meth:`~tokenizer.aligned_data.loader.function_data.FunctionData.full_token_stream`
rather than the body-only ``FunctionData.tokens`` slice. The wire-
format stream that the v2 codec was designed for is the concatenated
``variant_tokens + tokens`` view: the variant-axis prefix carries its
own inline-digit runs which the body-only slice would amputate
mid-metatoken, dropping the leading real-token invariant
``decode_raw_tokens`` enforces.

Exposed as a mixin :class:`_BinarySessionSpliceMixin` so the public
:py:meth:`splice_with_callees` method stays on :class:`BinarySession`
itself -- callers do not need to know about the split. Mixin
inheritance is purely additive: every attribute it reads
(``_vocab_manager``, ``_meta_get``, ``_load_matched_section_and_variants``,
``_load_unmatched_record_and_section``, ``_unmatched_record_slot_base``,
``_category_token_ids``, ``_number_token_ids``) is owned by
:class:`BinarySession` itself.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.tokens import Category, TokenType

from ..matched_sections_bin import Section
from .decoded.category_tokens import FID_KEYED_CATEGORIES
from .function_data import FunctionData

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .decoded.decoded_function import DecodedFunction


# Splice's FID-resolution lookup keys by Category; the BIN's call_target
# rows key by CallTargetType. Single source of truth for the mapping
# the session-level FID lookup walks.
_CALL_TARGET_TYPE_TO_FID_CATEGORY: Dict[CallTargetType, Category] = {
    CallTargetType.LOCAL: Category.LOCAL_FUNC,
    CallTargetType.PLT: Category.PLT_FUNC,
    CallTargetType.EXTERN: Category.EXT_FUNC,
}


def _build_fids_per_category(section: Section) -> Dict[Category, np.ndarray]:
    """Per-Category ``uint32`` FID array indexed by caller-local id.

    Plan Decisions 21 + 22: ``Section.call_targets[]`` is encounter-
    ordered within each :class:`CallTargetType`, categories concatenated
    LOCAL -> PLT -> EXT. The K-th LOCAL call_target row is the function
    with encoder-allocated LOCAL_FUNC identity ``K`` (mirror for PLT /
    EXT); the row's ``function_name_ptr`` is the callee's globally-
    unique FID. So each per-category list collected here, in the order
    the call_targets table emits, is exactly the lookup the decoder
    consumes to resolve a caller-local id into a FID.
    """
    per_category: Dict[Category, list] = {
        category: [] for category in FID_KEYED_CATEGORIES
    }
    for ct in section.call_targets:
        category = _CALL_TARGET_TYPE_TO_FID_CATEGORY.get(ct.type)
        if category is None:  # defensive -- no other CallTargetType exists today
            continue
        per_category[category].append(ct.function_name_ptr)
    return {
        category: np.array(values, dtype=np.uint32)
        for category, values in per_category.items()
    }


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
    ) -> Tuple[Dict[Category, int], Dict[TokenType, int], Optional[int]]:
        """Lazily resolve + cache the v2 Category + number-TokenType ids
        plus the ``value_negative`` postfix-metatoken id.

        All three are derived from ``self._vocab_manager`` via the
        :mod:`decoded.category_tokens` resolvers; the result is cached
        per session (cleared in ``BinarySession.__exit__``). The
        ``value_negative`` id may legitimately resolve to ``None`` when
        the vocab predates the postfix shape; the decoder treats the
        ``None`` case as "skip sign-handling" so the legacy-decode path
        stays exercised by such vocabs.
        """
        if self._category_token_ids is None or self._number_token_ids is None:
            from .decoded.category_tokens import (
                resolve_category_token_ids,
                resolve_number_token_ids,
                resolve_value_negative_token_id,
            )
            self._category_token_ids = resolve_category_token_ids(
                self._vocab_manager
            )
            self._number_token_ids = resolve_number_token_ids(
                self._vocab_manager
            )
            self._value_negative_token_id = resolve_value_negative_token_id(
                self._vocab_manager
            )
        return (
            self._category_token_ids,
            self._number_token_ids,
            self._value_negative_token_id,
        )

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
        self, idx: int, variant_index: int
    ) -> Tuple[FunctionData, Section, int]:
        """Resolve one matched function variant for splicing.

        ``idx`` is the per-function idx; ``variant_index`` is the index
        into :py:attr:`MatchedFunction.variants`. Returns the single
        :class:`FunctionData`, the parsed :class:`Section`, and the
        section's BIN byte offset. Raises :class:`IndexError` if
        ``variant_index`` is out of range -- the caller chose to splice
        a specific variant, so silent wrap-around would hide the bug.
        """
        section, section_offset, matched = (
            self._load_matched_section_and_variants(idx)
        )
        if variant_index < 0 or variant_index >= len(matched.variants):
            raise IndexError(
                f"matched function idx={idx} ({matched.func_name!r}) has "
                f"{len(matched.variants)} variants; variant_index "
                f"{variant_index} out of range"
            )
        return matched.variants[variant_index], section, section_offset

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
        max_variants: int = 1,
        inlined_equivalent_call_targets_only: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> "list[DecodedFunction]":
        """Decode + splice up to ``max_variants`` variant streams.

        Returns ``list[DecodedFunction]`` of length ``min(max_variants,
        len(section.variants))``. Each list element is one independent
        splice stream rooted at one selected variant; callers asking
        for one stream do ``result[0]``.

        Args:
            idx: Per-arm function index.
            arm: ``"matched"`` or ``"unmatched"``.
            max_depth: Recursion budget passed through to the walker.
            max_variants: Upper bound on how many variant streams to
                return. Must be ``>= 1`` -- ``0`` is rejected with
                :class:`ValueError` as a programmer error. ``arm=
                "unmatched"`` sections have exactly one variant by
                construction (matched_sections_bin invariant), so the
                same sampling logic yields a 1-element list there
                regardless of ``max_variants``.
            inlined_equivalent_call_targets_only: When ``True``, splice
                a call_target ``K`` iff the current selection of
                variants is split on whether they called ``K``. Variants
                that did NOT call ``K`` presumably inlined it, so
                splicing ``K``'s body gives the model the equivalent
                view. When ``False`` (default), the walker uses
                standard cycle + present checks only.
            rng: Sampling source. ``None`` (default) constructs a
                fresh :class:`numpy.random.Generator` per call --
                non-deterministic by design. Tests + trainers thread
                their own seeded generators when reproducibility is
                required.

        Per-stream variant sampling:
          * ``selection_size = min(max_variants, len(section.variants))``.
          * When ``selection_size == len(section.variants)`` all
            variants are returned in their existing index order (no
            shuffle).
          * Otherwise ``rng.choice(n, size=selection_size,
            replace=False)`` picks the variant indices and
            :func:`numpy.sort` makes the order deterministic given the
            rng's state.

        Each selected variant ``V`` drives ONE walker invocation rooted
        at variant ``V``'s body. The walker's ``decode_callee_to_staging``
        callback receives the callee variant index from the per-call
        entries lookup so callee bodies follow the primary variant's
        choice. Cross-arm callees do not resolve and surface as
        ``is_callee_present=False``: their call-site tokens stay in
        the caller's stream but their bodies are not spliced.
        """
        if max_variants < 1:
            raise ValueError(
                f"max_variants must be >= 1; got {max_variants}"
            )

        from .decoded.extract import _decode_to_staging
        from .decoded.splice import splice_with_callees as _splice_walker

        cat_ids, num_ids, vneg_id = self._get_token_id_caches()

        # Resolve the root section + each selected variant's FunctionData
        # exactly once -- the per-stream loop body parses tokens but does
        # NOT re-parse the section catalog. ``_load_root_variants`` is the
        # only arm-dispatch site; the per-stream and callback bodies are
        # arm-agnostic past the FunctionData handle.
        section, section_offset, per_variant_fd = self._load_root_variants(
            idx=idx,
            arm=arm,
            max_variants=max_variants,
            rng=rng,
        )

        selected_v_idxs = sorted(per_variant_fd.keys())
        initial_selection_vkeys = frozenset(
            int(section.variants[v].variant_ref_offset)
            for v in selected_v_idxs
        )

        def _decode_callee_to_staging(
            callee_offset: int,
            callee_arm: str,
            callee_variant_index: int,
        ):
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
                fd, sec, _off = self._load_matched_for_splice(
                    callee_idx, callee_variant_index
                )
            else:
                # Unmatched callees have exactly one variant by the
                # matched_sections_bin invariant; ignore the
                # walker-supplied variant index.
                fd, sec, _off = self._load_unmatched_for_splice(callee_idx)
            callee_fids = _build_fids_per_category(sec)
            staging = _decode_to_staging(
                fd.full_token_stream(),
                id_token_ids=cat_ids,
                number_token_ids=num_ids,
                fids_per_category=callee_fids,
                value_negative_token_id=vneg_id,
                func_name=fd.func_name,
                metadata=fd.metadata,
            )
            return staging, sec

        def _is_present(callee_offset: int, callee_arm: str) -> bool:
            return (
                self._idx_for_section_offset(callee_offset, callee_arm)
                is not None
            )

        root_fids = _build_fids_per_category(section)
        results: "list[DecodedFunction]" = []
        for v_idx in selected_v_idxs:
            root_fd = per_variant_fd[v_idx]
            root_staging = _decode_to_staging(
                root_fd.full_token_stream(),
                id_token_ids=cat_ids,
                number_token_ids=num_ids,
                fids_per_category=root_fids,
                value_negative_token_id=vneg_id,
                func_name=root_fd.func_name,
                metadata=root_fd.metadata,
            )
            spliced = _splice_walker(
                root_staging=root_staging,
                root_arm=arm,
                root_section=section,
                root_section_offset=section_offset,
                decode_callee_to_staging=_decode_callee_to_staging,
                is_callee_present=_is_present,
                max_depth=max_depth,
                primary_variant_idx=int(v_idx),
                initial_selection_vkeys=initial_selection_vkeys,
                inlined_equivalent_call_targets_only=(
                    inlined_equivalent_call_targets_only
                ),
            )
            results.append(spliced)

        return results

    # --- internal helpers ------------------------------------------

    def _load_root_variants(  # type: ignore[no-untyped-def]
        self,
        *,
        idx: int,
        arm: str,
        max_variants: int,
        rng: Optional[np.random.Generator],
    ) -> Tuple[Section, int, Dict[int, FunctionData]]:
        """Resolve the root section + each selected variant's FunctionData.

        Single arm-dispatch site for ``splice_with_callees``. Matched
        arm: parses the section once, then picks the variants chosen by
        :func:`_select_variant_indices` from the pre-built
        :class:`MatchedFunction`. Unmatched arm: the section has one
        variant by construction so the selection logic yields exactly
        ``{0: fd}`` regardless of ``max_variants``.

        Returns ``(section, section_offset, {v_idx: FunctionData})``.
        Raises :class:`ValueError` on an unknown ``arm`` name.
        """
        if arm == "matched":
            section, section_offset, matched = (
                self._load_matched_section_and_variants(idx)
            )
            selected = _select_variant_indices(
                n_variants=len(section.variants),
                max_variants=max_variants,
                rng=rng,
            )
            per_variant_fd: Dict[int, FunctionData] = {
                int(v): matched.variants[int(v)] for v in selected
            }
            return section, section_offset, per_variant_fd
        if arm == "unmatched":
            root_fd, section, section_offset = (
                self._load_unmatched_for_splice(idx)
            )
            # An unmatched section has exactly one variant by the
            # matched_sections_bin invariant; the sampling rule still
            # validates ``max_variants`` (a 1-element list).
            _select_variant_indices(
                n_variants=len(section.variants),
                max_variants=max_variants,
                rng=rng,
            )
            return section, section_offset, {0: root_fd}
        raise ValueError(f"unknown arm: {arm!r}")


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


def _select_variant_indices(
    *,
    n_variants: int,
    max_variants: int,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Pick variant indices for the per-stream splice loop.

    ``selection_size = min(max_variants, n_variants)``. If the
    selection covers every variant the indices are returned in their
    existing order (no shuffle); otherwise ``rng.choice`` samples
    without replacement and the result is sorted for determinism.

    ``rng=None`` constructs a fresh non-deterministic
    :class:`numpy.random.Generator` per call -- callers that need
    reproducibility (tests, seeded trainers) thread their own.
    """
    if n_variants <= 0:
        raise ValueError(
            f"section has zero variants; cannot sample (n_variants={n_variants})"
        )
    selection_size = min(max_variants, n_variants)
    if selection_size == n_variants:
        return np.arange(n_variants, dtype=np.int64)
    generator = rng if rng is not None else np.random.default_rng()
    chosen = generator.choice(
        n_variants, size=selection_size, replace=False
    )
    return np.sort(chosen)
