"""Deterministic EXHAUSTIVE whole-corpus section enumeration (no rng).

Single concern of this module: walk a fixed, ordered set of per-binary
:class:`SortedIndexReader` readers and emit EVERY non-excluded section
EXACTLY ONCE, with ALL of its variants, in a stable deterministic order.
It is a MODULAR SIBLING to :class:`SequentialValidationSampler` -- NOT a
branch in it, never imported by it. Where the validation sampler
shuffles + chunks + drops a per-section variant subset off a seeded
xoshiro256** stream, this enumeration does the opposite extreme: NO rng,
NO shuffle, NO chunk, NO drop -- the whole corpus, every section once,
every variant in raw index order.

Concern boundaries (identical to the validation sampler's):

* The reader owns the in-band ENUMERATION
  (:meth:`SortedIndexReader.enumerate_in_band`); this module owns only
  the pointer construction on top of it. The band is the FULL non-excluded
  band ``EXCLUDED_LENGTH + 1 .. reader.max_length`` per reader, so every
  non-excluded section is enumerated (length truncation is an eval-side
  filter, never baked into enumeration).
* The module does NOT own session lifecycle or decode -- it consumes an
  INJECTED ``count_provider`` (``binary_name, section_indices ->
  int64[n_variants]``), the SAME :data:`VariantCountProvider` contract the
  validation sampler uses, so it stays decode/session-agnostic.
* Each emitted pointer carries an :class:`ExplicitIndicesSelection` of
  ``range(n_variants)`` -- all variants in raw order -- so the all-variants
  subset rides through the EXISTING :func:`resolve_section_pointers` seam
  verbatim; this module never touches the decode path. Under an explicit
  selection the variant SLOT index equals the raw variant index (a clean
  identity), so a downstream RAGGED ``batch_idx_to_section_variant``
  recovers each row's raw variant directly from its variant column. The
  section column of that mapping is the GROUP-LOCAL pointer position (NOT
  the matched-arm idx); the eval-side AsmRowId reconstruction maps that
  position back through the decoded pointer list to the real
  ``(binary, matched_idx)`` -- so this module emits clean per-section
  pointers and leaves the row-id assembly to the eval.

This enumeration is a FAITHFUL pass-through of the index: it emits exactly
the sections the non-excluded band contains, with exactly their variant
counts, and never filters. The production builder already stamps every
0-variant or gated-out section with ``EXCLUDED_LENGTH`` (see
``_length_compute``: ``emitted = n_variants > 0 & gate``), so a real
non-excluded band never enumerates a 0-variant section -- filtering is the
builder's concern, not the enumeration's.

Canonical deterministic order (stable across runs + machines):

* Binaries in the GIVEN ``readers_in_order`` order. The eval framework
  builds that list from :func:`discover_members`, whose ``members`` are
  returned sorted by ``qualified_name`` (alphabetical, filesystem- and
  machine-independent) -- so the canonical corpus order IS the
  alphabetical-by-qualified-name binary order. This module honors the
  given order verbatim (it never re-sorts), mirroring the validation
  sampler; the alphabetical guarantee lives at the construction seam.
* Within a binary, sections in :meth:`SortedIndexReader.enumerate_in_band`
  order -- the body's stable length-bucketed sorted layout.
* Within a section, variants in ascending raw index order
  ``range(n_variants)``.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.batch_decode._variant_selection import (
    ExplicitIndicesSelection,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from .._reader import SortedIndexReader
from .._types import IndexSpec, MultiBinarySectionPointer
from .._wire import EXCLUDED_LENGTH


__all__ = [
    "ExhaustiveSectionSampler",
    "all_section_pointers",
]


# ``count_provider(binary_name, section_indices) -> int64[n_variants]``:
# the injected per-binary, body-free variant-count source -- the SAME
# contract :data:`._validation.VariantCountProvider` defines. Keyed by the
# in-band matched-section indices the reader enumerated, in that order.
VariantCountProvider = Callable[[str, np.ndarray], np.ndarray]


class ExhaustiveSectionSampler:
    """Ordered deterministic generator of every section's all-variants pointer.

    Binds a FIXED, ordered list of ``(binary_name, spec, reader)`` triples
    (the GIVEN order is the emission order -- the eval supplies the
    canonical alphabetical-by-qualified-name order from
    :func:`discover_members`). Each call to :meth:`all_pointers` re-derives
    the SAME sequence: it walks every reader in order, enumerates that
    reader's FULL non-excluded band
    ``EXCLUDED_LENGTH + 1 .. reader.max_length`` via the rng-free
    :meth:`SortedIndexReader.enumerate_in_band`, and emits one
    :class:`MultiBinarySectionPointer` per enumerated section whose
    :class:`ExplicitIndicesSelection` pins ``range(n_variants)`` (all
    variants, raw order).

    There is NO rng anywhere: no seed, no shuffle, no chunk, no drop. Two
    calls with the same ``(readers_in_order, count provider per-section
    counts)`` emit a byte-identical pointer sequence (order + every pointer
    field).
    """

    def __init__(
        self,
        readers_in_order: List[Tuple[str, IndexSpec, SortedIndexReader]],
    ) -> None:
        self._readers_in_order = list(readers_in_order)

    def all_pointers(
        self, count_provider: VariantCountProvider
    ) -> List[MultiBinarySectionPointer]:
        """Every non-excluded section's all-variants pointer, in canonical order.

        ``count_provider(binary_name, section_indices) -> int64[]`` returns
        the per-section variant counts for the reader's in-band sections, in
        enumeration order (the SAME injected contract the validation sampler
        consumes). For each reader, in input order, every enumerated section
        becomes one pointer carrying ``ExplicitIndicesSelection(range(n))``.

        Sections enumerate over the FULL non-excluded band per reader
        (``EXCLUDED_LENGTH + 1 .. reader.max_length``); a reader with no
        in-band sections (empty index) contributes nothing. The result is a
        fresh list, identical across repeated calls for the same inputs --
        no :class:`numpy.random.Generator` is constructed or touched.
        """
        pointers: List[MultiBinarySectionPointer] = []
        for binary_name, spec, reader in self._readers_in_order:
            # FULL non-excluded band: EXCLUDED_LENGTH+1 .. max_length.
            # enumerate_in_band clamps lo past EXCLUDED_LENGTH itself, so
            # the lo here is documentation; an empty index yields max_length
            # 0 -> an empty band -> no sections.
            sec_idxs = reader.enumerate_in_band(
                EXCLUDED_LENGTH + 1, reader.max_length
            )
            if sec_idxs.size == 0:
                continue
            n_variants = np.asarray(
                count_provider(binary_name, sec_idxs), dtype=np.int64
            )
            self._emit_sections(
                pointers, binary_name, spec, sec_idxs, n_variants
            )
        return pointers

    @staticmethod
    def _emit_sections(
        pointers: List[MultiBinarySectionPointer],
        binary_name: str,
        spec: Optional[IndexSpec],
        sec_idxs: np.ndarray,
        n_variants: np.ndarray,
    ) -> None:
        """One all-variants pointer per enumerated section for one reader.

        ``sec_idxs[i]`` is the matched section idx; ``n_variants[i]`` is its
        variant count. The pointer pins ``ExplicitIndicesSelection`` of
        ``tuple(range(n_variants[i]))`` -- every variant once, in ascending
        raw order, so the variant slot index equals the raw variant index.
        """
        for matched_idx, n in zip(sec_idxs.tolist(), n_variants.tolist()):
            pointers.append(
                MultiBinarySectionPointer(
                    binary_name=binary_name,
                    section_pointer=SectionPointerSpec(
                        arm=SectionKind.MATCHED,
                        idx=int(matched_idx),
                        variant_selection=ExplicitIndicesSelection(
                            tuple(range(int(n)))
                        ),
                    ),
                    spec=spec,
                )
            )


def all_section_pointers(
    readers_in_order: List[Tuple[str, IndexSpec, SortedIndexReader]],
    count_provider: VariantCountProvider,
) -> Sequence[MultiBinarySectionPointer]:
    """Every non-excluded section's all-variants pointer, in canonical order.

    The whole-corpus exhaustive sibling to the validation sampler's
    deterministic bunch stream: enumerates EVERY non-excluded section
    across ALL binaries EXACTLY ONCE (full band per reader), with ALL of
    its variants (``range(n_variants)``), in the stable canonical order
    documented on :class:`ExhaustiveSectionSampler`. No rng.

    ``readers_in_order`` is the eval-built ``(binary_name, spec, reader)``
    list in canonical (alphabetical-by-qualified-name) order;
    ``count_provider`` is the injected body-free variant-count source (the
    :data:`VariantCountProvider` contract). Returns a fresh
    ``Sequence[MultiBinarySectionPointer]`` the eval batches through
    :func:`decode_pointer_batch` with ``variant_padding=RAGGED``.
    """
    return ExhaustiveSectionSampler(readers_in_order).all_pointers(
        count_provider
    )
