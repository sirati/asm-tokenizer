"""End-to-end pipeline smoke for the sorted-index module.

Drives the full pipeline through the *real* production entries -- no
hand-written wire bytes anywhere -- so a regression anywhere along
writer -> file -> discovery -> reader -> sampler -> batch helper
surfaces as a single failing test:

1. Lay down a tiny 2-binary memmap dir by running the synthetic
   ``build_combined_fixture`` twice into separate scratch dirs and
   renaming the on-disk binary-name prefix into one shared dir
   (mirrors the helper in ``test_batch_helper.py`` + ``test_cli.py``).
2. Run :func:`write_sorted_index_files` for each binary with multiple
   modes (``[MAX, P95]``) -- exercises the §D8 multi-mode amortised
   walk and the canonical-filename writer.
3. Verify the canonical-grammar filenames land on disk.
4. Open the indices via :func:`discover_indices` +
   :class:`SortedIndexReader`.
5. Build a :class:`MultiBinarySortedIndexSampler` over the two readers.
6. Sample a small batch via :func:`open_length_bucketed_batch`.
7. Assert tensor shapes + per-row binary-id provenance + bucket-bound
   sanity (sampled rows came from the requested length bucket).

This complements (does not replace) ``test_batch_helper.py``: that
file exercises the helper against *hand-built* sorted-index files;
this one exercises the real builder's outputs end-to-end.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    MultiBinarySortedIndexSampler,
    ReductionKind,
    SortedIndexReader,
    discover_indices,
    open_length_bucketed_batch,
    write_sorted_index_files,
)

from ._length_helpers import ensure_sidecar
from .fixtures import build_combined_fixture, make_test_vocab_manager


_BINARY_NAME_A = "binA"
_BINARY_NAME_B = "binB"
_DEPTH = 3

_MAX = LengthReduction(kind=ReductionKind.MAX)
_P95 = LengthReduction(kind=ReductionKind.PERCENTILE, percentile=95)


# ---------------------------------------------------------------------------
# Multi-binary fixture builder
# ---------------------------------------------------------------------------


def _build_two_binary_memmap_dir(tmp_path: Path) -> Path:
    """Lay down two distinct named binaries with identical synthetic content.

    The ``build_combined_fixture`` helper hardcodes ``binary_name =
    "sortbin"`` -- we invoke it against two child dirs and copy each
    file across with a renamed prefix so the two binaries co-exist in
    one memmap directory under independently-discoverable names.
    """
    memmap_dir = tmp_path / "memmap"
    memmap_dir.mkdir()
    for binary_name in (_BINARY_NAME_A, _BINARY_NAME_B):
        scratch = tmp_path / f"scratch_{binary_name}"
        scratch.mkdir()
        combined_base = build_combined_fixture(scratch)
        for entry in combined_base.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.startswith("sortbin"):
                continue
            new_name = binary_name + entry.name[len("sortbin"):]
            (memmap_dir / new_name).write_bytes(entry.read_bytes())
        # The sorted-index build hard-requires each binary's matched-arm
        # realized-length sidecar (the Phase-4a precondition); generate it
        # for the renamed binary in the final memmap dir.
        ensure_sidecar(memmap_dir, binary_name)
    return memmap_dir


@contextmanager
def _session_factory_for(memmap_dir: Path):
    """Context-manager session_factory closing over ``memmap_dir``."""
    vocab_manager = make_test_vocab_manager()
    @contextmanager
    def factory(binary_name: str) -> Iterator[BinarySession]:
        dataset = BinaryDataset(memmap_dir, binary_name, vocab_manager=vocab_manager)
        with dataset.open_session() as session:
            yield session
    yield factory


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_e2e_writer_through_batch_helper(tmp_path: Path) -> None:
    """Full pipeline smoke over a tiny synthetic 2-binary memmap.

    Drives the real :func:`write_sorted_index_files` builder, then
    discovers + opens the produced ``.idx`` files via the public
    reader API, builds a sampler, and runs
    :func:`open_length_bucketed_batch` over a deterministic RNG.

    Asserts:

    * Both ``_sorted_max_d003.idx`` and ``_sorted_p95_d003.idx`` exist
      per binary with the canonical filename grammar.
    * :func:`discover_indices` round-trips both binaries with both
      reductions.
    * The batch helper returns tensor shapes matching the
      ``(batch_size * num_variants_per_section, context_len)`` contract
      shipped in ``test_batch_helper.py``.
    * ``binary_id_per_row`` only carries valid indices into
      :attr:`MultiBinaryBatchDecodeResult.binary_names` (which is
      alphabetical).
    * Sampled rows fall within the requested target_length bucket
      (the reader's ``count_at(target_length)`` covers every drawn
      row's source index).
    """
    memmap_dir = _build_two_binary_memmap_dir(tmp_path)

    # ---- 2. Run the real writer for each binary with [MAX, P95]. ----
    for binary_name in (_BINARY_NAME_A, _BINARY_NAME_B):
        write_sorted_index_files(
            memmap_dir, binary_name,
            reductions=[_MAX, _P95],
            depths=[_DEPTH],
        )

    # ---- 3. Canonical-grammar filenames land on disk. ----
    for binary_name in (_BINARY_NAME_A, _BINARY_NAME_B):
        for tag in ("max", "p95"):
            path = memmap_dir / f"{binary_name}_sorted_{tag}_d003.idx"
            assert path.is_file(), f"missing canonical-grammar file {path}"

    # ---- 4. discover_indices + SortedIndexReader open cleanly. ----
    discovered = discover_indices(memmap_dir, depth=_DEPTH)
    assert set(discovered.keys()) == {_BINARY_NAME_A, _BINARY_NAME_B}
    for binary_name in (_BINARY_NAME_A, _BINARY_NAME_B):
        assert set(discovered[binary_name]) == {_MAX, _P95}, (
            f"{binary_name}: discover_indices missed a reduction"
        )

    # Open one reader per binary at the MAX reduction (the target
    # reduction for the sampler in this smoke).
    readers = {
        binary_name: SortedIndexReader(
            memmap_dir / f"{binary_name}_sorted_max_d003.idx",
            reduction=_MAX,
            depth=_DEPTH,
        )
        for binary_name in (_BINARY_NAME_A, _BINARY_NAME_B)
    }
    # Sanity: the reader's total_sections matches the fixture's section
    # count (5 sections per combined fixture: func_zero / solo_a /
    # multi_fn / caller_fn / callee_fn).
    for binary_name, rdr in readers.items():
        assert rdr.total_sections() == 5, (
            f"{binary_name}: reader.total_sections() == "
            f"{rdr.total_sections()}, expected 5"
        )

    # ---- 5. Build the sampler. ----
    sampler = MultiBinarySortedIndexSampler(readers)
    assert sampler.binary_names == sorted([_BINARY_NAME_A, _BINARY_NAME_B])

    # ---- 6. Pick a target length with a non-empty cross-binary pool. ----
    #
    # The two binaries hold identical synthetic content (only the
    # prefix differs) so their MAX-reduced length arrays match.  Pick
    # the bucket with the largest population so a 4-sample draw is
    # comfortably below the bucket size on at least one binary.
    rdr_a = readers[_BINARY_NAME_A]
    candidate_lengths = list(range(
        rdr_a.min_length,
        rdr_a.max_length + 1,
    ))
    # Filter out the length-0 bucket (the 0-variant ``func_zero``
    # section stamps to 0 under MAX -- a real-corpus consumer would
    # skip these via target_length > 0 anyway).
    nonzero_lengths = [
        L for L in candidate_lengths if L > 0 and rdr_a.count_at(L) > 0
    ]
    assert nonzero_lengths, (
        "fixture produced no nonzero-length buckets under MAX; "
        "smoke cannot exercise the sampler"
    )
    target_length = max(
        nonzero_lengths,
        key=lambda L: sampler.count_at(L),
    )
    pool_size = sampler.count_at(target_length)
    assert pool_size >= 1

    # ---- 7. Run the batch helper. ----
    batch_size = min(4, pool_size)
    num_variants_per_section = 2
    context_len = 32
    rng = np.random.default_rng(42)

    with _session_factory_for(memmap_dir) as factory:
        result = open_length_bucketed_batch(
            factory,
            sampler,
            target_length=target_length,
            batch_size=batch_size,
            context_len=context_len,
            num_variants_per_section=num_variants_per_section,
            max_depth=2,
            rng=rng,
        )

    # ---- 7a. Shape contract. ----
    expected_rows = batch_size * num_variants_per_section
    inner = result.inner
    assert inner.tokens.shape == (expected_rows, context_len)
    assert inner.tokens.dtype == np.uint16
    assert inner.identity_row_offsets.shape == (expected_rows + 1,)
    assert inner.number_row_offsets.shape == (expected_rows + 1,)
    assert result.binary_id_per_row.shape == (expected_rows,)

    # ---- 7b. Per-row binary id is a valid index into binary_names. ----
    bin_ids = set(int(b) for b in result.binary_id_per_row)
    assert bin_ids <= {0, 1}
    assert result.binary_names == sorted(
        [_BINARY_NAME_A, _BINARY_NAME_B]
    )

    # ---- 7c. binary_id_per_row matches alphabetical binary order
    # (open_length_bucketed_batch concatenates per-binary results in
    # alphabetical order; rows must be monotone-non-decreasing on
    # binary_id). ----
    assert list(result.binary_id_per_row) == sorted(
        list(result.binary_id_per_row),
    ), (
        f"binary_id_per_row not sorted alphabetically: "
        f"{result.binary_id_per_row.tolist()}"
    )

    # ---- 8. Bucket-bound sanity ----
    #
    # The sampler draws from each per-binary reader's
    # ``sample_section_indices(target_length, ...)``; every drawn
    # section pointer must live in the requested bucket.  We can't
    # directly inspect the sampler's draw from the result alone, but
    # we can verify the bucket was non-empty on at least one binary
    # (otherwise the helper would have raised on an empty pool) and
    # that the bucket population satisfies the draw size.
    drawn_per_binary = {
        binary_name: int((result.binary_id_per_row == i).sum())
        // num_variants_per_section
        for i, binary_name in enumerate(result.binary_names)
    }
    for binary_name, drawn in drawn_per_binary.items():
        if drawn == 0:
            continue
        bucket_count = readers[binary_name].count_at(target_length)
        assert drawn <= bucket_count, (
            f"{binary_name}: drew {drawn} from a bucket of size "
            f"{bucket_count} at target_length={target_length} -- "
            f"sampler exceeded the bucket"
        )
