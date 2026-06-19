"""Deterministic sequential validation sampler (the ordered generator).

Single concern of this module: walk a fixed, ordered set of per-binary
:class:`SortedIndexReader` readers and emit a deterministic, reproducible
stream of fixed-size variant bunches over the in-band sections. It is a
MODULAR SIBLING to the cross-binary urn samplers in :mod:`._sample` --
NOT a branch in them, never imported by them. Where the urn samplers draw
an unbiased random subset, this generator enumerates EVERY in-band section
and chunks a deterministic per-section shuffle into ``batch_size`` bunches,
dropping the per-section remainder.

The determinism comes entirely from the Rust
``variant_shuffle_chunk_kernel`` (re-exported as
:func:`dedup_hashmap.variant_shuffle_chunk_kernel`): ONE xoshiro256**
stream, seeded once via :func:`._validation_oracle.derive_initial_state`,
threaded across sections AND across readers (each reader's ``state_out``
feeds the next reader's ``state_in``), so the whole multi-file pass is one
continuous reproducible stream for a fixed seed.

Concern boundaries:

* The sampler does NOT own session lifecycle or decode -- it consumes an
  INJECTED ``count_provider`` (``binary_name, section_indices ->
  int64[n_variants]``) so it stays decode/session-agnostic. The validation
  dataloader entry (:mod:`._validation_batch`) supplies the closure that
  opens a body-free session per binary.
* The reader owns the in-band ENUMERATION
  (:meth:`SortedIndexReader.enumerate_in_band`); this sampler owns only the
  shuffle/chunk/drop + pointer construction on top of that enumeration.
* The emitted :class:`ValidationBatch` carries an
  :class:`ExplicitIndicesSelection`, so the explicit variant subset rides
  through the EXISTING :func:`resolve_section_pointers` seam verbatim -- the
  sampler never touches the decode path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from dedup_hashmap import variant_shuffle_chunk_kernel

from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.batch_decode._variant_selection import (
    ExplicitIndicesSelection,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind

from .._reader import SortedIndexReader
from .._types import IndexSpec, MultiBinarySectionPointer
from ._validation_oracle import derive_initial_state


__all__ = [
    "ValidationBatch",
    "SequentialValidationSampler",
    "VariantCountProvider",
]


# ``count_provider(binary_name, section_indices) -> int64[n_variants]``:
# the injected per-binary, body-free variant-count source. Keyed by the
# in-band matched-section indices the reader enumerated, in that order.
VariantCountProvider = Callable[[str, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ValidationBatch:
    """One emitted bunch: ``batch_size`` variants of one matched section.

    Carries the binary name, the section pointer (whose
    :class:`ExplicitIndicesSelection` pins exactly ``batch_size``
    deterministic variant indices), and the optional source
    :class:`IndexSpec`. :meth:`as_pointer` lifts it into the cross-binary
    :class:`MultiBinarySectionPointer` the decode entry feeds verbatim to
    :func:`decode_pointer_batch`.
    """

    binary_name: str
    section_pointer: SectionPointerSpec
    spec: Optional[IndexSpec]

    def as_pointer(self) -> MultiBinarySectionPointer:
        """Lift into the cross-binary pointer the decode entry consumes."""
        return MultiBinarySectionPointer(
            binary_name=self.binary_name,
            section_pointer=self.section_pointer,
            spec=self.spec,
        )


class SequentialValidationSampler:
    """Ordered deterministic generator of fixed-size variant bunches.

    Binds a FIXED, ordered list of ``(binary_name, spec, reader)`` triples
    (the GIVEN order is the emission order -- typically index-file order)
    plus a ``batch_size``, a length ``band``, and a ``seed``. Each call to
    :meth:`iter_batches` re-derives the SAME stream from ``seed`` and walks
    every reader in order, emitting one :class:`ValidationBatch` per bunch.

    Per reader, per in-band section: the kernel Fisher-Yates-shuffles the
    section's ``n_variants`` via the shared stream, keeps
    ``floor(n/batch_size) * batch_size``, and chunks the kept prefix into
    bunches of ``batch_size`` (sections with ``n < batch_size`` emit
    nothing). Bunches are section-major within a reader; readers are walked
    in input order. The stream state threads across sections AND across
    readers so the whole pass is one continuous xoshiro256** sequence.
    """

    def __init__(
        self,
        readers_in_order: List[Tuple[str, IndexSpec, SortedIndexReader]],
        batch_size: int,
        band: Tuple[int, int],
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._readers_in_order = list(readers_in_order)
        self._batch_size = int(batch_size)
        self._band = (int(band[0]), int(band[1]))
        self._seed = int(seed)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def band(self) -> Tuple[int, int]:
        return self._band

    def iter_batches(
        self, count_provider: VariantCountProvider
    ) -> Iterator[ValidationBatch]:
        """Yield the deterministic bunch stream over every reader in order.

        ``count_provider(binary_name, section_indices) -> int64[]`` returns
        the per-section variant counts for the reader's in-band sections,
        in enumeration order. The kernel state is seeded once from ``seed``
        and threaded reader-to-reader, so two calls with the same
        ``(readers_in_order, batch_size, band, seed)`` emit an identical
        sequence (independent of ``count_provider`` identity, given the same
        per-section counts).
        """
        lo, hi = self._band
        # Single continuous xoshiro256** stream: seed once, thread forward.
        state = np.asarray(
            derive_initial_state(self._seed), dtype=np.uint64
        )
        for binary_name, spec, reader in self._readers_in_order:
            sec_idxs = reader.enumerate_in_band(lo, hi)
            if sec_idxs.size == 0:
                continue
            n_variants = np.asarray(
                count_provider(binary_name, sec_idxs), dtype=np.int64
            )
            variant_idx, bunch_offsets, bunch_section, state = (
                variant_shuffle_chunk_kernel(
                    n_variants, self._batch_size, state
                )
            )
            yield from self._emit_bunches(
                binary_name,
                spec,
                sec_idxs,
                variant_idx,
                bunch_offsets,
                bunch_section,
            )

    def _emit_bunches(
        self,
        binary_name: str,
        spec: Optional[IndexSpec],
        sec_idxs: np.ndarray,
        variant_idx: np.ndarray,
        bunch_offsets: np.ndarray,
        bunch_section: np.ndarray,
    ) -> Iterator[ValidationBatch]:
        """One :class:`ValidationBatch` per kernel bunch for one reader.

        ``bunch_section[b]`` is the OWNING in-band-section ordinal (an
        index into ``sec_idxs``); it is mapped to the actual matched
        section index ``sec_idxs[bunch_section[b]]``. The bunch's variant
        indices are the exact ``batch_size`` slice
        ``variant_idx[bunch_offsets[b]:bunch_offsets[b + 1]]`` the kernel
        emitted, pinned on the pointer as an
        :class:`ExplicitIndicesSelection`.
        """
        n_bunches = int(bunch_section.shape[0])
        for b in range(n_bunches):
            start = int(bunch_offsets[b])
            stop = int(bunch_offsets[b + 1])
            indices = tuple(int(v) for v in variant_idx[start:stop])
            matched_idx = int(sec_idxs[int(bunch_section[b])])
            yield ValidationBatch(
                binary_name=binary_name,
                section_pointer=SectionPointerSpec(
                    arm=SectionKind.MATCHED,
                    idx=matched_idx,
                    variant_selection=ExplicitIndicesSelection(indices),
                ),
                spec=spec,
            )
