"""Cross-binary sampler + result concat + length-bucketed batch helper.

Three concerns wired into one module because they form one user-facing
flow (sample -> per-binary decode -> concat) and share private helpers
that should not leak to the public API:

* :class:`MultiBinarySortedIndexSampler` and the free
  :func:`sample_section_pointers` -- Reading-A unbiased without-
  replacement sample over per-binary
  :class:`SortedIndexReader` urns (plan ALG-3 + D6).
* :func:`_concat_results` -- stitches per-binary
  :class:`BatchDecodeResult` instances into one
  :class:`MultiBinaryBatchDecodeResult`, re-basing per-row cumsums and
  stamping a per-row ``binary_id`` sidecar (plan ALG-6).
* :func:`open_length_bucketed_batch` -- top-level helper: sample,
  group by binary, open one session per binary, run
  :func:`batch_decode`, then concat (plan D7).

Binary ordering is canonical alphabetical at every layer: the sampler's
:attr:`MultiBinarySortedIndexSampler.binary_names`, the sampler's
internal per-binary iteration order, the
:func:`open_length_bucketed_batch` per-binary loop, and the concat
helper's input list ALL use the same alphabetical order so the per-row
``binary_id`` numbering is stable across runs.

Imports from ``loader/batch_decode`` here are the typed handoff classes
plus the public :func:`batch_decode` entry; we do NOT touch any internal
stage helpers.
"""

from __future__ import annotations

from typing import Callable, ContextManager, Dict, List, Optional, Tuple

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._entry import batch_decode
from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession

from ._reader import SortedIndexReader
from ._types import (
    MultiBinaryBatchDecodeResult,
    MultiBinarySectionPointer,
)


__all__ = [
    "MultiBinarySortedIndexSampler",
    "open_length_bucketed_batch",
    "sample_section_pointers",
]


# ---------------------------------------------------------------------------
# Cross-binary unbiased sampler (Reading A, plan ALG-3 + D6)
# ---------------------------------------------------------------------------


class MultiBinarySortedIndexSampler:
    """Stateful Reading-A sampler binding a fixed set of per-binary readers.

    Construction canonicalises the per-binary order to alphabetical
    ``binary_name`` so downstream consumers (notably :func:`_concat_results`
    and :func:`open_length_bucketed_batch`) can rely on stable per-row
    ``binary_id`` numbering across runs.

    The class is intentionally thin: every per-call sampling decision is
    delegated to :func:`sample_section_pointers` so the algorithm is
    testable in isolation without a sampler instance.
    """

    def __init__(self, readers: Dict[str, SortedIndexReader]) -> None:
        ordered_names = sorted(readers)
        # Preserve only the alphabetical-order dict so per-binary
        # iteration in :func:`sample_section_pointers` and
        # :func:`open_length_bucketed_batch` matches the ordering
        # exposed by :attr:`binary_names`.
        self._readers: Dict[str, SortedIndexReader] = {
            name: readers[name] for name in ordered_names
        }
        self._binary_names: List[str] = ordered_names

    @property
    def binary_names(self) -> List[str]:
        """Alphabetical ``binary_name`` list -- the ``binary_id`` reverse map."""
        return list(self._binary_names)

    def count_at(self, target_length: int) -> int:
        """Pool size at ``target_length`` summed over every binary."""
        return sum(r.count_at(target_length) for r in self._readers.values())

    def sample_section_pointers(
        self,
        target_length: int,
        count: int,
        rng: np.random.Generator,
    ) -> List[MultiBinarySectionPointer]:
        """Delegate to :func:`sample_section_pointers` with our readers."""
        return sample_section_pointers(
            self._readers, target_length, count, rng,
        )


def sample_section_pointers(
    readers: Dict[str, SortedIndexReader],
    target_length: int,
    count: int,
    rng: np.random.Generator,
) -> List[MultiBinarySectionPointer]:
    """Reading-A unbiased sample over per-binary urns.

    Each ``(binary, section_idx)`` pair at ``target_length`` is equally
    likely; larger binaries contribute proportionally more samples
    (plan D6). Implementation:

    1. Sum each binary's ``count_at(target_length)`` into a per-binary
       urn size vector.
    2. Draw a per-binary count vector via
       :meth:`numpy.random.Generator.multivariate_hypergeometric` --
       exact without-replacement draw across all urns.
    3. Per binary, call ``sample_section_indices`` with the drawn
       count and build :class:`MultiBinarySectionPointer` rows.
    4. Shuffle the combined output to break per-binary row clustering
       (otherwise downstream batches would have a deterministic
       per-binary block layout that leaks the sampler's per-urn order).

    Empty pool (``total == 0``) returns ``[]``; the helper does NOT
    raise -- the caller (:func:`open_length_bucketed_batch`) is the
    layer that raises :class:`ValueError` to surface this to training
    loops.

    ``readers`` is iterated in dict-insertion order; callers wanting
    stable cross-run output should pass an alphabetically-canonical
    dict (:class:`MultiBinarySortedIndexSampler` does so internally).
    """
    per_binary_counts = {
        name: rdr.count_at(target_length) for name, rdr in readers.items()
    }
    total = sum(per_binary_counts.values())
    if total == 0:
        return []
    k = min(count, total)

    binary_names = list(per_binary_counts)
    counts_arr = np.array(
        [per_binary_counts[n] for n in binary_names], dtype=np.int64,
    )
    drawn = rng.multivariate_hypergeometric(counts_arr, k)

    out: List[MultiBinarySectionPointer] = []
    for name, draw in zip(binary_names, drawn):
        draw_int = int(draw)
        if draw_int == 0:
            continue
        idxs = readers[name].sample_section_indices(
            target_length, draw_int, rng,
        )
        out.extend(
            MultiBinarySectionPointer(
                binary_name=name,
                section_pointer=SectionPointerSpec(
                    arm=SectionKind.MATCHED, idx=int(i),
                ),
            )
            for i in idxs
        )

    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Cross-binary BatchDecodeResult concatenation (plan ALG-6)
# ---------------------------------------------------------------------------


def _concat_row_offsets(offsets_list: List[np.ndarray]) -> np.ndarray:
    """Concatenate per-batch cumsum offset arrays into one global cumsum.

    Each input is ``u32[batch_size_i + 1]`` with ``offsets_i[0] == 0``.
    Output is ``u32[sum(batch_size_i) + 1]`` re-based so offsets are
    cumulative across the concatenated batch.

    Computation happens in ``uint64`` so the re-base addition cannot
    overflow when individual offsets are near the u32 limit; the
    final cast back to ``uint32`` is safe under the assumption (held
    by the existing single-binary stages) that no single batch's
    cumulative count exceeds u32 range.
    """
    if not offsets_list:
        raise ValueError("_concat_row_offsets: empty input")
    pieces: List[np.ndarray] = [offsets_list[0].astype(np.uint64)]
    running = int(offsets_list[0][-1])
    for offsets in offsets_list[1:]:
        # Drop offsets[0] (== 0) when appending; re-base by running total.
        pieces.append(
            offsets[1:].astype(np.uint64) + np.uint64(running),
        )
        running += int(offsets[-1])
    return np.concatenate(pieces).astype(np.uint32)


def _concat_results(
    per_binary: List[Tuple[str, BatchDecodeResult]],
) -> MultiBinaryBatchDecodeResult:
    """Stitch per-binary :class:`BatchDecodeResult`s into one combined result.

    ``per_binary`` MUST be sorted by alphabetical ``binary_name`` (the
    canonical order :class:`MultiBinarySortedIndexSampler` exposes); the
    caller :func:`open_length_bucketed_batch` enforces this.

    Invariants enforced here:

    * **Shape precondition (plan D-2.1)**: every per-binary
      ``tokens.shape[1]`` must equal the first's; otherwise raise a
      clear :class:`ValueError` rather than letting
      :func:`numpy.concatenate` raise a cryptic shape mismatch.
    * **Row offsets** are re-based via :func:`_concat_row_offsets` so
      the stitched cumsums are globally monotone.
    * **batch_idx_to_section_variant** is NOT re-numbered; the new
      ``binary_id_per_row`` sidecar carries cross-binary identity.
    * **fid sidecar all-or-none**: every per-binary result must
      either supply :attr:`BatchDecodeResult.fid_sidecar` or none of
      them must; mixed inputs raise :class:`ValueError`.
    * **Intermediate is dropped**: the per-binary
      :class:`Stage3Batch` is single-binary-scoped and cannot be
      stitched cross-binary; the returned inner result's
      :attr:`BatchDecodeResult.intermediate` is always ``None``. The
      caller is expected to set ``keep_intermediate=False`` in
      :func:`open_length_bucketed_batch`.
    """
    if not per_binary:
        raise ValueError("_concat_results: empty input")

    # Shape precondition (plan D-2.1).
    context_len = per_binary[0][1].tokens.shape[1]
    mismatched = [
        name for name, r in per_binary if r.tokens.shape[1] != context_len
    ]
    if mismatched:
        raise ValueError(
            "_concat_results: tokens.shape[1] mismatch across binaries: "
            f"first sees {context_len}, diverging: {mismatched}",
        )

    binary_names = [name for name, _ in per_binary]
    name_to_id = {name: i for i, name in enumerate(binary_names)}

    # Token tensor: stack rows.
    tokens = np.concatenate([r.tokens for _, r in per_binary], axis=0)

    # Identities + per-row offsets.
    identities = np.concatenate([r.identities for _, r in per_binary])
    identity_offsets = _concat_row_offsets(
        [r.identity_row_offsets for _, r in per_binary],
    )

    # Numbers + per-row offsets.
    sig = np.concatenate(
        [r.numbers_significant for _, r in per_binary],
    )
    sgnexp = np.concatenate(
        [r.numbers_sign_exponent for _, r in per_binary],
    )
    number_offsets = _concat_row_offsets(
        [r.number_row_offsets for _, r in per_binary],
    )

    # batch_idx_to_section_variant: stack as-is; binary_id sidecar
    # below carries cross-binary identity.
    btv = np.concatenate(
        [r.batch_idx_to_section_variant for _, r in per_binary], axis=0,
    )

    # binary_id_per_row: repeat the binary's id once per row.
    binary_id_per_row = np.concatenate([
        np.full(r.tokens.shape[0], name_to_id[name], dtype=np.uint32)
        for name, r in per_binary
    ])

    # Optional fid sidecar (all-or-none across inputs).
    fid_present = [r.fid_sidecar is not None for _, r in per_binary]
    fid_sidecar: Optional[np.ndarray] = None
    fid_offsets: Optional[np.ndarray] = None
    if all(fid_present):
        fid_sidecar = np.concatenate(
            [r.fid_sidecar for _, r in per_binary],
        )
        fid_offsets = _concat_row_offsets(
            [r.fid_row_offsets for _, r in per_binary],
        )
    elif any(fid_present):
        raise ValueError(
            "_concat_results: include_fid_sidecar inconsistent across inputs",
        )

    inner = BatchDecodeResult(
        tokens=tokens,
        identities=identities,
        identity_row_offsets=identity_offsets,
        numbers_significant=sig,
        numbers_sign_exponent=sgnexp,
        number_row_offsets=number_offsets,
        batch_idx_to_section_variant=btv,
        fid_sidecar=fid_sidecar,
        fid_row_offsets=fid_offsets,
        intermediate=None,
    )
    return MultiBinaryBatchDecodeResult(
        inner=inner,
        binary_id_per_row=binary_id_per_row,
        binary_names=binary_names,
    )


# ---------------------------------------------------------------------------
# Top-level batch helper (plan D7)
# ---------------------------------------------------------------------------


def open_length_bucketed_batch(
    session_factory: Callable[[str], ContextManager[BinarySession]],
    sampler: MultiBinarySortedIndexSampler,
    target_length: int,
    batch_size: int,
    *,
    context_len: int,
    num_variants_per_section: int,
    max_depth: int,
    rng: np.random.Generator,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
    inlined_equivalent_call_targets_only: bool = False,
    include_fid_sidecar: bool = False,
    keep_intermediate: bool = False,
) -> MultiBinaryBatchDecodeResult:
    """Length-bucketed batch helper (plan D7).

    Samples ``batch_size`` section pointers via ``sampler`` at
    ``target_length``, groups by binary, opens one session per binary
    via ``session_factory``, runs :func:`batch_decode` over each
    group, and concatenates the per-binary results via
    :func:`_concat_results`.

    Binary ordering: per-binary results are concatenated in
    alphabetical ``binary_name`` order (the same order
    :attr:`MultiBinarySortedIndexSampler.binary_names` exposes). This
    determines the ``binary_id_per_row`` numbering and is stable
    across runs.

    Raises
    ------
    ValueError
        When the sampler returns 0 pointers (empty pool at
        ``target_length`` across every binary). The caller is
        expected to handle this -- either skip the training step or
        pick a different ``target_length``.
    ValueError
        When ``keep_intermediate=True``. The cross-binary
        :func:`_concat_results` boundary inherently drops per-binary
        :class:`Stage3Batch` intermediates (each is single-binary-
        scoped and not stitchable), so the helper rejects this flag
        rather than silently producing inconsistent state.
    """
    if keep_intermediate:
        raise ValueError(
            "open_length_bucketed_batch: keep_intermediate=True is not "
            "supported -- per-binary Stage3Batch intermediates cannot "
            "cross the concat boundary",
        )

    pointers = sampler.sample_section_pointers(
        target_length, batch_size, rng,
    )
    if not pointers:
        raise ValueError(
            "open_length_bucketed_batch: empty sampler pool at "
            f"target_length={target_length}",
        )

    # Group section pointers by binary_name. Iterate the sampled list
    # rather than the sampler's full binary set so binaries with zero
    # samples are skipped (no empty BinarySession opens).
    per_binary_pointers: Dict[str, List[SectionPointerSpec]] = {}
    for ptr in pointers:
        per_binary_pointers.setdefault(ptr.binary_name, []).append(
            ptr.section_pointer,
        )

    # Iterate per-binary work in alphabetical order so the concat
    # input list is canonical and the resulting binary_id_per_row
    # numbering is stable.
    per_binary_results: List[Tuple[str, BatchDecodeResult]] = []
    for binary_name in sampler.binary_names:
        section_pointers = per_binary_pointers.get(binary_name)
        if section_pointers is None:
            continue
        with session_factory(binary_name) as session:
            result = batch_decode(
                session,
                section_pointers,
                num_variants_per_section=num_variants_per_section,
                context_len=context_len,
                max_depth=max_depth,
                variant_padding=variant_padding,
                inlined_equivalent_call_targets_only=(
                    inlined_equivalent_call_targets_only
                ),
                include_fid_sidecar=include_fid_sidecar,
                keep_intermediate=False,
                rng=rng,
            )
        per_binary_results.append((binary_name, result))

    return _concat_results(per_binary_results)
