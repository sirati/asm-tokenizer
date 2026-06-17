"""Corpus-level collection over many indexed memmap directories.

Single concern: *serve unbiased length-bucketed batches from a corpus
of per-binary indexed memmap directories with persistent per-binary
sessions, for any of several configured ``(reduction, depth)`` specs.*

A real corpus is many memmap directories (one per package), each
holding several per-binary catalogs plus their sorted-index ``.idx``
files -- one ``.idx`` per ``(reduction, depth)`` :class:`IndexSpec`.
:class:`IndexedMemmapCollection` is the single typed entry point that
discovers every binary across the whole collection (via
:func:`discover_members`), wires them into ONE
:class:`MultiBinarySortedIndexSampler` PER spec (so each spec's urn draw
is unbiased over the entire corpus, not biased per directory), and
serves length-bucketed batches through persistent sessions.

Membership is spec-INDEPENDENT: a binary is a member iff it carries the
``.idx`` for EVERY configured spec (see :func:`discover_members`). The
per-spec samplers therefore differ only in their bucket lengths, never
in their population, and ``members`` / qualified naming / the session
pool stay collection-level. One warm session per member serves batches
for depth-0 AND depth-3 alike -- the whole point of binding several
specs in one collection.

Boundary contract -- everything is REUSE, never reimplementation:

* discovery + naming -> :func:`discover_members`;
* spec-list boundary + per-call selection -> :mod:`._spec`;
* sampling -> :class:`MultiBinarySortedIndexSampler` (one per spec; the
  multivariate-hypergeometric urn draw is the unbiasedness mechanism);
* decode -> :func:`decode_pointer_batch` verbatim;
* per-binary sessions -> :meth:`BinaryDataset.open_session`, opened
  lazily on first need and held on a :class:`contextlib.ExitStack` until
  :meth:`close` -- shared across every spec.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import VariantPadding
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession

from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    VectorBatchArmSet,
    open_vector_batch_arm_set,
)

from .._reader import SortedIndexReader
from .._sampler import (
    CrossSpecSortedIndexSampler,
    DecodeEngine,
    MultiBinarySortedIndexSampler,
    decode_pointer_batch,
)
from .._types import (
    IndexSpec,
    LengthReduction,
    MultiBinaryBatchDecodeResult,
    MultiBinarySectionPointer,
)
from ._discovery import discover_members
from ._member import CollectionMember, MissingIndexPolicy
from ._spec import normalize_specs, resolve_spec, sorted_specs


__all__ = ["IndexedMemmapCollection"]


class IndexedMemmapCollection:
    """Unbiased length-bucketed batch source over a memmap-dir collection.

    Construct via :meth:`discover`. The collection owns:

    * the typed :class:`CollectionMember` list (alphabetical by
      ``qualified_name``) -- spec-independent;
    * one :class:`MultiBinarySortedIndexSampler` PER configured spec
      over ``{qualified_name -> SortedIndexReader}`` -- each the unbiased
      urn draw over the WHOLE corpus at that spec's lengths;
    * lazily-opened persistent :class:`BinarySession` instances (one per
      member, opened on first sample, held on an :class:`ExitStack`)
      SHARED across every spec.

    Every per-call method (:meth:`count_at`, :meth:`count_in_band`,
    :meth:`sample_section_pointers`, :meth:`load_batch`) takes an
    optional ``spec``: ``None`` resolves to the single configured spec
    (full back-compat) or raises when several are configured; an unknown
    spec raises. The configured specs are exposed in stable order via
    :attr:`specs`.

    Use as a context manager (or call :meth:`close`) to release every
    open session deterministically. After :meth:`close`, :meth:`load_batch`
    raises :class:`RuntimeError`.
    """

    def __init__(
        self,
        members: List[CollectionMember],
        readers_by_spec: Dict[IndexSpec, Dict[str, SortedIndexReader]],
        vocab_manager: Optional[Any] = None,
    ) -> None:
        # ``members`` arrives alphabetical by qualified_name from
        # :meth:`discover`; keep that order for the public properties.
        self._members = members
        # ``vocab_manager`` + a ``{qualified_name -> member}`` index are
        # the only state needed to build each per-binary
        # :class:`BinaryDataset` lazily (on first session/handles use);
        # discovery no longer parses any section arm. NOTE: the
        # ``_variants.bin``/``_variants.csv`` sidecar-PAIR validation that
        # ``BinaryDataset.__init__`` performs therefore fires at a
        # binary's FIRST sample (in :meth:`_dataset_for`) rather than at
        # discovery -- same loud ValueError, just deferred.
        self._vocab = vocab_manager
        self._member_by_name: Dict[str, CollectionMember] = {
            m.qualified_name: m for m in members
        }
        # Memoized per-binary datasets: one parse per binary, built on
        # first use and registered on the shared ExitStack (released with
        # the sessions on :meth:`close`).
        self._dataset_cache: Dict[str, BinaryDataset] = {}
        # One sampler per spec; specs held in stable display order.
        self._specs: List[IndexSpec] = sorted_specs(readers_by_spec.keys())
        self._samplers: Dict[IndexSpec, MultiBinarySortedIndexSampler] = {
            spec: MultiBinarySortedIndexSampler(readers_by_spec[spec])
            for spec in self._specs
        }
        # One cross-(binary x spec) sampler over the SAME readers the
        # per-spec samplers use, so a cross-depth draw is unbiased across
        # the binary AND the depth axis at once (each pointer carries the
        # spec it was drawn from).
        self._cross_sampler = CrossSpecSortedIndexSampler(readers_by_spec)
        self._sessions: Dict[str, BinarySession] = {}
        # Lazy per-binary vector_batch handle bundles, opened on first
        # VECTOR_BATCH decode and held on the same ExitStack as the
        # sessions (released together on :meth:`close`). Empty unless the
        # vector_batch engine is ever selected.
        self._vb_handles: Dict[str, VectorBatchArmSet] = {}
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
        specs: Optional[Sequence[IndexSpec]] = None,
        reduction: Optional[LengthReduction] = None,
        depth: Optional[int] = None,
        vocab_manager: Optional[Any] = None,
        on_missing: MissingIndexPolicy = MissingIndexPolicy.RAISE,
    ) -> "IndexedMemmapCollection":
        """Assemble the collection from ``memmap_dirs`` for one or more specs.

        Pass EITHER ``specs=[IndexSpec, ...]`` (one or more
        ``(reduction, depth)`` pairs) OR the single-spec convenience
        ``reduction=... depth=...`` -- exactly one form, never both nor
        neither (:func:`._spec.normalize_specs` validates this boundary;
        duplicate specs raise). Membership is uniform across specs: a
        binary missing any spec's ``.idx`` is excluded from the whole
        collection per ``on_missing``.

        Delegates the filesystem discovery + cross-directory naming +
        per-member reader construction to :func:`discover_members`. The
        per-binary :class:`BinaryDataset` objects are NOT built here; the
        collection builds each lazily on first use (see
        :meth:`_dataset_for`), so ``vocab_manager`` is threaded into the
        ctor for that deferred construction.
        """
        resolved_specs = normalize_specs(specs, reduction, depth)
        members, readers_by_spec = discover_members(
            memmap_dirs,
            specs=resolved_specs,
            on_missing=on_missing,
        )
        return cls(members, readers_by_spec, vocab_manager=vocab_manager)

    # ------------------------------------------------------------------
    # Static surface
    # ------------------------------------------------------------------
    @property
    def members(self) -> List[CollectionMember]:
        """Discovered members, alphabetical by ``qualified_name``.

        Spec-independent: every member carries the ``.idx`` for every
        configured spec.
        """
        return list(self._members)

    @property
    def binary_names(self) -> List[str]:
        """Qualified names, alphabetical (the ``binary_id`` reverse map).

        Spec-independent (membership is uniform across specs).
        """
        return [m.qualified_name for m in self._members]

    @property
    def specs(self) -> List[IndexSpec]:
        """Configured specs in stable order (by ``(filename_tag, depth)``)."""
        return list(self._specs)

    def _sampler_for(
        self, spec: Optional[IndexSpec]
    ) -> MultiBinarySortedIndexSampler:
        """Resolve ``spec`` and return its sampler (one helper for all four)."""
        return self._samplers[resolve_spec(spec, self._specs)]

    def count_at(
        self, target_length: int, *, spec: Optional[IndexSpec] = None
    ) -> int:
        """Corpus-wide pool size at the exact ``target_length`` bucket."""
        return self._sampler_for(spec).count_at(target_length)

    def count_in_band(
        self, lo: int, hi: int, *, spec: Optional[IndexSpec] = None
    ) -> int:
        """Corpus-wide pool size for lengths in ``[lo, hi]`` inclusive."""
        return self._sampler_for(spec).count_in_band(lo, hi)

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
        spec: Optional[IndexSpec] = None,
    ) -> List[MultiBinarySectionPointer]:
        """Unbiased cross-corpus sample from ``spec``'s pool (delegates)."""
        self._check_open()
        return self._sampler_for(spec).sample_section_pointers(
            target_length, count, rng, band=band,
        )

    def sample_section_pointers_cross_depth(
        self,
        target_length: int,
        count: int,
        rng: np.random.Generator,
        *,
        band: Optional[Tuple[int, int]] = None,
    ) -> List[MultiBinarySectionPointer]:
        """Unbiased cross-(binary x spec) sample over EVERY configured spec.

        Draws across all ``(binary, spec)`` cells at once (no ``spec=``
        selector -- the depth axis is part of the urn), so each returned
        :class:`MultiBinarySectionPointer` carries the ``spec`` it was
        drawn from. The downstream cross-depth decode reads each row's
        ``max_depth`` straight off ``spec.depth``.
        """
        self._check_open()
        return self._cross_sampler.sample_section_pointers(
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
        inlined_equivalent_call_targets_only: bool = True,
        include_fid_sidecar: bool = False,
        engine: DecodeEngine = DecodeEngine.BATCH_DECODE,
        spec: Optional[IndexSpec] = None,
    ) -> MultiBinaryBatchDecodeResult:
        """Sample ``batch_size`` pointers from ``spec`` and decode them.

        Samples via ``spec``'s unbiased sampler (with ``band``
        threading), opens (lazily, reusing already-open ones) one
        persistent session per sampled binary -- SHARED across every spec
        -- and decodes via :func:`decode_pointer_batch`. The per-call
        session lifecycle of :func:`open_length_bucketed_batch` is
        replaced here by sessions that persist across calls until
        :meth:`close`.

        ``engine`` selects the per-binary decode engine
        (:attr:`DecodeEngine.BATCH_DECODE`, the default; or
        :attr:`DecodeEngine.VECTOR_BATCH`, the geometry-first path).
        VECTOR_BATCH is byte-identical to BATCH_DECODE; the collection
        supplies its per-binary :class:`VectorBatchArmSet` handles lazily
        through :meth:`_handles_for` (held on the same :class:`ExitStack`
        as the sessions, released together on :meth:`close`).

        Raises
        ------
        RuntimeError
            After :meth:`close`.
        ValueError
            When the sampler returns no pointers (empty pool at
            ``target_length`` or across the whole ``band``).
        """
        self._check_open()
        pointers = self._sampler_for(spec).sample_section_pointers(
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
            engine=engine,
            handle_provider=self._handles_for,
        )

    def load_batch_cross_depth(
        self,
        target_length: int,
        batch_size: int,
        *,
        rng: np.random.Generator,
        band: Optional[Tuple[int, int]] = None,
        context_len: int,
        num_variants_per_section: int,
        variant_padding: VariantPadding = VariantPadding.PAD_NULL,
        inlined_equivalent_call_targets_only: bool = True,
        include_fid_sidecar: bool = False,
    ) -> MultiBinaryBatchDecodeResult:
        """Sample + decode a mixed-depth batch over the cross-(binary x spec) urn.

        Unlike :meth:`load_batch` (one ``spec`` -> one depth), this draws
        across EVERY configured spec at once, so a single batch mixes
        sections from different depths. Each sampled pointer's
        ``spec.depth`` becomes its row's ``max_depth``; the per-pointer
        depth vector is threaded to :func:`decode_pointer_batch`, which
        regroups it by binary and runs the geometry-first
        :func:`vector_batch_tokens` per depth group.

        Cross-depth is VECTOR_BATCH-only (the staged BATCH_DECODE engine
        has no per-row depth seam), so this method always selects
        :attr:`DecodeEngine.VECTOR_BATCH` and supplies the per-binary
        handle bundles via :meth:`_handles_for` -- exactly as
        :meth:`load_batch` does for that engine.

        Raises
        ------
        RuntimeError
            After :meth:`close`.
        ValueError
            When the cross-depth sampler returns no pointers (empty pool
            at ``target_length`` or across the whole ``band``).
        """
        self._check_open()
        pointers = self._cross_sampler.sample_section_pointers(
            target_length, batch_size, rng, band=band,
        )
        if not pointers:
            if band is not None:
                raise ValueError(
                    "IndexedMemmapCollection.load_batch_cross_depth: empty "
                    f"sampler pool in band {band}",
                )
            raise ValueError(
                "IndexedMemmapCollection.load_batch_cross_depth: empty "
                f"sampler pool at target_length={target_length}",
            )

        # Each row's depth = the spec it was drawn from. The cross-depth
        # sampler always stamps a spec, so this never reads None.
        max_depth_per_pointer = np.array(
            [ptr.spec.depth for ptr in pointers], dtype=np.int64,
        )

        sampled = {ptr.binary_name for ptr in pointers}
        sessions = {name: self._session_for(name) for name in sampled}
        return decode_pointer_batch(
            sessions,
            pointers,
            context_len=context_len,
            num_variants_per_section=num_variants_per_section,
            max_depth=max_depth_per_pointer,
            rng=rng,
            variant_padding=variant_padding,
            inlined_equivalent_call_targets_only=(
                inlined_equivalent_call_targets_only
            ),
            include_fid_sidecar=include_fid_sidecar,
            engine=DecodeEngine.VECTOR_BATCH,
            handle_provider=self._handles_for,
        )

    # ------------------------------------------------------------------
    # Session lifetime
    # ------------------------------------------------------------------
    def _dataset_for(self, qualified_name: str) -> BinaryDataset:
        """Return the per-binary :class:`BinaryDataset`, built lazily.

        Constructs ``BinaryDataset(member.memmap_dir, member.binary_name,
        vocab_manager=...)`` on first need (parsing that binary's section
        arms then, NOT at discovery), caches it so every later
        session/handles use shares ONE instance (one parse per binary),
        and registers a drop-on-close callback on the same
        :class:`ExitStack` as the sessions so :meth:`close` releases the
        cached parse. (A :class:`BinaryDataset` owns only parsed metadata,
        no OS handles -- those live in :class:`BinarySession` -- so it has
        nothing to close itself; the callback just frees the cached parse
        in lockstep with the sessions/handles.) A member never sampled
        never builds a dataset -- this is what keeps discovery
        section-parse-free.
        """
        dataset = self._dataset_cache.get(qualified_name)
        if dataset is None:
            member = self._member_by_name[qualified_name]
            dataset = BinaryDataset(
                member.memmap_dir,
                member.binary_name,
                vocab_manager=self._vocab,
            )
            self._dataset_cache[qualified_name] = dataset
            self._stack.callback(self._dataset_cache.pop, qualified_name, None)
        return dataset

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
                self._dataset_for(qualified_name).open_session()
            )
            self._sessions[qualified_name] = session
        return session

    def _handles_for(self, qualified_name: str) -> VectorBatchArmSet:
        """Return the persistent vector_batch handle bundle for ``qualified_name``.

        Mirrors :meth:`_session_for`: opens both arms' columnar +
        geometry + body handles lazily on first VECTOR_BATCH decode (a
        member never decoded under that engine never opens handles),
        registers the bundle on the same :class:`ExitStack` so
        :meth:`close` releases it, and reuses it on subsequent calls. The
        ``base_path`` / ``binary_name`` come from the member's
        :class:`BinaryDataset` -- the same keys the index build uses.
        """
        handles = self._vb_handles.get(qualified_name)
        if handles is None:
            dataset = self._dataset_for(qualified_name)
            handles = self._stack.enter_context(
                open_vector_batch_arm_set(
                    dataset.base_path, dataset.binary_name
                )
            )
            self._vb_handles[qualified_name] = handles
        return handles

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
        self._vb_handles.clear()
        self._stack.close()

    def __enter__(self) -> "IndexedMemmapCollection":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
