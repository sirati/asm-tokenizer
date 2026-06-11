"""Corpus-level collection over many indexed memmap directories.

Single concern: *serve unbiased length-bucketed batches from a corpus
of per-binary indexed memmap directories with persistent per-binary
sessions.*

A real corpus is many memmap directories (one per package), each
holding several per-binary catalogs plus their sorted-index ``.idx``
files. :class:`IndexedMemmapCollection` is the single typed entry point
that discovers every binary across the whole collection (via
:func:`discover_members`), wires them into ONE
:class:`MultiBinarySortedIndexSampler` (so the urn draw is unbiased over
the entire corpus, not biased per directory), and serves length-bucketed
batches through persistent sessions.

Boundary contract -- everything is REUSE, never reimplementation:

* discovery + naming -> :func:`discover_members`;
* sampling -> :class:`MultiBinarySortedIndexSampler` (the
  multivariate-hypergeometric urn draw is the unbiasedness mechanism);
* decode -> :func:`decode_pointer_batch` verbatim;
* per-binary sessions -> :meth:`BinaryDataset.open_session`, opened
  lazily on first need and held on a :class:`contextlib.ExitStack` until
  :meth:`close`.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import VariantPadding
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession

from .._reader import SortedIndexReader
from .._sampler import MultiBinarySortedIndexSampler, decode_pointer_batch
from .._types import (
    LengthReduction,
    MultiBinaryBatchDecodeResult,
    MultiBinarySectionPointer,
)
from ._discovery import discover_members
from ._member import CollectionMember, MissingIndexPolicy


__all__ = ["IndexedMemmapCollection"]


class IndexedMemmapCollection:
    """Unbiased length-bucketed batch source over a memmap-dir collection.

    Construct via :meth:`discover`. The collection owns:

    * the typed :class:`CollectionMember` list (alphabetical by
      ``qualified_name``);
    * one :class:`MultiBinarySortedIndexSampler` over
      ``{qualified_name -> SortedIndexReader}`` -- the single unbiased
      urn draw over the WHOLE corpus;
    * lazily-opened persistent :class:`BinarySession` instances (one per
      member, opened on first sample, held on an :class:`ExitStack`).

    Use as a context manager (or call :meth:`close`) to release every
    open session deterministically. After :meth:`close`, :meth:`load_batch`
    raises :class:`RuntimeError`.
    """

    def __init__(
        self,
        members: List[CollectionMember],
        readers: Dict[str, SortedIndexReader],
        datasets: Dict[str, BinaryDataset],
    ) -> None:
        # ``members`` arrives alphabetical by qualified_name from
        # :meth:`discover`; keep that order for the public properties.
        self._members = members
        self._sampler = MultiBinarySortedIndexSampler(readers)
        self._datasets = datasets
        self._sessions: Dict[str, BinarySession] = {}
        self._stack = ExitStack()
        self._closed = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def discover(
        cls,
        memmap_dirs: Sequence[Path],
        *,
        reduction: LengthReduction,
        depth: int,
        vocab_manager: Optional[Any] = None,
        on_missing: MissingIndexPolicy = MissingIndexPolicy.RAISE,
    ) -> "IndexedMemmapCollection":
        """Assemble the collection from ``memmap_dirs``.

        Delegates the filesystem discovery + cross-directory naming +
        per-member reader/dataset construction to
        :func:`discover_members`; missing ``(reduction, depth)`` indices
        are handled per ``on_missing``.
        """
        members, readers, datasets = discover_members(
            memmap_dirs,
            reduction=reduction,
            depth=depth,
            vocab_manager=vocab_manager,
            on_missing=on_missing,
        )
        return cls(members, readers, datasets)

    # ------------------------------------------------------------------
    # Static surface
    # ------------------------------------------------------------------
    @property
    def members(self) -> List[CollectionMember]:
        """Discovered members, alphabetical by ``qualified_name``."""
        return list(self._members)

    @property
    def binary_names(self) -> List[str]:
        """Qualified names, alphabetical (the sampler's ``binary_id`` map)."""
        return self._sampler.binary_names

    def count_at(self, target_length: int) -> int:
        """Corpus-wide pool size at the exact ``target_length`` bucket."""
        return self._sampler.count_at(target_length)

    def count_in_band(self, lo: int, hi: int) -> int:
        """Corpus-wide pool size for lengths in ``[lo, hi]`` inclusive."""
        return self._sampler.count_in_band(lo, hi)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_section_pointers(
        self,
        target_length: int,
        count: int,
        rng: np.random.Generator,
        *,
        band: Optional[Tuple[int, int]] = None,
    ) -> List[MultiBinarySectionPointer]:
        """Unbiased cross-corpus sample (delegates to the sampler)."""
        self._check_open()
        return self._sampler.sample_section_pointers(
            target_length, count, rng, band=band,
        )

    # ------------------------------------------------------------------
    # Batch loading
    # ------------------------------------------------------------------
    def load_batch(
        self,
        target_length: int,
        batch_size: int,
        *,
        rng: np.random.Generator,
        band: Optional[Tuple[int, int]] = None,
        context_len: int,
        num_variants_per_section: int,
        max_depth: int,
        variant_padding: VariantPadding = VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only: bool = False,
        include_fid_sidecar: bool = False,
    ) -> MultiBinaryBatchDecodeResult:
        """Sample ``batch_size`` pointers and decode them across the corpus.

        Samples via the internal unbiased sampler (with ``band``
        threading), opens (lazily, reusing already-open ones) one
        persistent session per sampled binary, and decodes via
        :func:`decode_pointer_batch` -- the per-call session lifecycle of
        :func:`open_length_bucketed_batch` is replaced here by sessions
        that persist across calls until :meth:`close`.

        Raises
        ------
        RuntimeError
            After :meth:`close`.
        ValueError
            When the sampler returns no pointers (empty pool at
            ``target_length`` or across the whole ``band``).
        """
        self._check_open()
        pointers = self._sampler.sample_section_pointers(
            target_length, batch_size, rng, band=band,
        )
        if not pointers:
            if band is not None:
                raise ValueError(
                    "IndexedMemmapCollection.load_batch: empty sampler pool "
                    f"in band {band}",
                )
            raise ValueError(
                "IndexedMemmapCollection.load_batch: empty sampler pool at "
                f"target_length={target_length}",
            )

        sampled = {ptr.binary_name for ptr in pointers}
        sessions = {name: self._session_for(name) for name in sampled}
        return decode_pointer_batch(
            sessions,
            pointers,
            context_len=context_len,
            num_variants_per_section=num_variants_per_section,
            max_depth=max_depth,
            rng=rng,
            variant_padding=variant_padding,
            inlined_equivalent_call_targets_only=(
                inlined_equivalent_call_targets_only
            ),
            include_fid_sidecar=include_fid_sidecar,
        )

    # ------------------------------------------------------------------
    # Session lifetime
    # ------------------------------------------------------------------
    def _session_for(self, qualified_name: str) -> BinarySession:
        """Return the persistent session for ``qualified_name``.

        Opens it lazily on first need (a member never sampled never
        opens a session) and registers it on the :class:`ExitStack` so
        :meth:`close` releases every handle. Reuses an already-open
        session on subsequent calls.
        """
        session = self._sessions.get(qualified_name)
        if session is None:
            session = self._stack.enter_context(
                self._datasets[qualified_name].open_session()
            )
            self._sessions[qualified_name] = session
        return session

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "IndexedMemmapCollection: collection is closed",
            )

    def close(self) -> None:
        """Release every open session; idempotent.

        After this, :meth:`load_batch` and :meth:`sample_section_pointers`
        raise :class:`RuntimeError`.
        """
        if self._closed:
            return
        self._closed = True
        self._sessions.clear()
        self._stack.close()

    def __enter__(self) -> "IndexedMemmapCollection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
