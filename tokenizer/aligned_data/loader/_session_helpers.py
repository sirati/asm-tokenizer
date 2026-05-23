"""Per-arm load helpers + variant sampler for :class:`BinarySession`.

Single concern of this module: provide the session helpers the
batch-decode pipeline consumes -- the per-arm
``(idx, variant_index) -> (FunctionData, Section, section_offset)``
load methods, the inverse ``section_offset -> idx`` lookup, and the
RNG-driven variant-index sampler. The actual decode + splice walk
lives in :mod:`tokenizer.aligned_data.loader.batch_decode`; this
module owns only the I/O shape that surrounds it.

Exposed as a mixin :class:`_BinarySessionHelpersMixin` so the helpers
stay on :class:`BinarySession` itself -- callers do not need to know
about the split. Mixin inheritance is purely additive: every attribute
it reads (``_meta_get``, ``_load_matched_section_and_variants``,
``_load_unmatched_record_and_section``,
``_unmatched_record_slot_base``) is owned by :class:`BinarySession`
itself.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ..matched_sections_bin import Section
from .function_data import FunctionData


class _BinarySessionHelpersMixin:
    """Mixin providing the per-arm load + inverse-lookup helpers.

    Every method reads attributes / methods that :class:`BinarySession`
    owns; this class deliberately holds no state of its own. The
    inheritance is a code-locality tool, not a polymorphic boundary.
    """

    # The mixin is consumed in a strict order: BinarySession declares
    # the underlying state + load helpers, this mixin adds the public
    # helpers + the inverse-section lookup. ``self`` is typed ``Any``
    # inside the bodies because the concrete attributes live on the
    # subclass; declaring them here would duplicate the source of truth
    # and risk drift.

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
        unknown ``arm`` name -- the caller is strict on its inputs.
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
        """Resolve one matched function variant for the batch pipeline.

        ``idx`` is the per-function idx; ``variant_index`` is the index
        into :py:attr:`MatchedFunction.variants`. Returns the single
        :class:`FunctionData`, the parsed :class:`Section`, and the
        section's BIN byte offset. Raises :class:`IndexError` if
        ``variant_index`` is out of range -- the caller chose to load
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
        """Resolve one unmatched record for the batch pipeline.

        ``idx`` is the per-record idx (same input shape as
        :py:meth:`BinarySession.load_unmatched`). Returns the parsed
        :class:`FunctionData`, the owning :class:`Section`, and that
        section's BIN byte offset.
        """
        section, section_offset, fd = (
            self._load_unmatched_record_and_section(idx)
        )
        return fd, section, section_offset


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
    """Pick variant indices for per-section variant sampling.

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
