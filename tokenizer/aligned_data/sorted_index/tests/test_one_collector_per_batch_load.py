"""One-collector-per-batch-load contract tests for the sorted-index
orchestrators.

Single concern: assert that the two top-level orchestrators
(:func:`open_length_bucketed_batch` and :func:`compute_reduced_lengths`)
each instantiate :class:`BucketedRunLengthCollector` EXACTLY ONCE per
call and flush it EXACTLY ONCE -- regardless of how many per-binary
``batch_decode`` calls or per-chunk ``walk_sections`` calls happen
internally. This pins the design contract that run_lengths amortises
across the whole batch_load / sorted-index build, not just per
``walk_sections`` invocation.

The assertions are wired through :func:`unittest.mock.patch` on the
orchestrator's own import binding of ``BucketedRunLengthCollector``.
A counting subclass increments construction + flush counters; the test
patches the orchestrator's binding to point at the subclass.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List
from unittest.mock import patch

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.sorted_index import (
    LengthReduction,
    MultiBinarySortedIndexSampler,
    ReductionKind,
    SortedIndexReader,
    compute_reduced_lengths,
    encode_sorted_index,
    open_length_bucketed_batch,
    read_section_variant_info,
)

from .fixtures import build_combined_fixture


_BINARY_NAME_A = "binA"
_BINARY_NAME_B = "binB"
_BINARY_NAME_C = "binC"


# ---------------------------------------------------------------------------
# Counting-subclass scaffolding
# ---------------------------------------------------------------------------


class _CountingCollectorFactory:
    """Builds counting-subclass instances that share construction +
    flush counters across every instance the factory produces.

    Pattern: build one factory per test, patch the orchestrator's
    binding to point at ``factory.cls``, then read
    ``factory.n_constructed`` / ``factory.n_flushed`` after the call.
    """

    def __init__(self) -> None:
        self.n_constructed = 0
        self.n_flushed = 0
        outer = self

        class _CountingCollector(BucketedRunLengthCollector):
            def __init__(self) -> None:
                super().__init__()
                outer.n_constructed += 1

            def flush(self) -> dict[int, np.ndarray]:
                outer.n_flushed += 1
                return super().flush()

        self.cls = _CountingCollector


# ---------------------------------------------------------------------------
# Multi-binary fixture builder (3 binaries for open_length_bucketed_batch).
# ---------------------------------------------------------------------------


def _build_three_binary_fixture(tmp_path: Path) -> Path:
    """Lay down three combined-corpus binaries under ``tmp_path``.

    Mirrors :func:`tests.test_batch_helper._build_multi_binary_fixture`
    but with three binaries instead of two so the orchestrator does
    Stage 1 across at least three internal ``batch_decode`` calls --
    making the once-per-batch_load assertion meaningful.
    """
    memmap_dir = tmp_path / "memmap"
    memmap_dir.mkdir()
    for binary_name in (_BINARY_NAME_A, _BINARY_NAME_B, _BINARY_NAME_C):
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
    return memmap_dir


def _write_synthetic_sorted_index(
    memmap_dir: Path,
    binary_name: str,
    section_indices: List[int],
    length: int,
) -> Path:
    """Lay down a sorted-index file where every entry sits at one length."""
    num_sections = max(section_indices) + 1
    lengths = np.full(num_sections, length + 100, dtype=np.uint32)
    for idx in section_indices:
        lengths[idx] = length
    path = memmap_dir / f"{binary_name}_sorted_max_d003.idx"
    path.write_bytes(encode_sorted_index(lengths))
    return path


def _make_session_factory(memmap_dir: Path):
    @contextmanager
    def session_factory(binary_name: str) -> Iterator[BinarySession]:
        dataset = BinaryDataset(memmap_dir, binary_name, vocab_manager=None)
        with dataset.open_session() as session:
            yield session
    return session_factory


def _open_three_binary_sampler(
    memmap_dir: Path, length: int,
) -> MultiBinarySortedIndexSampler:
    sampleable_indices = [1, 2, 3, 4]
    readers = {}
    for name in (_BINARY_NAME_A, _BINARY_NAME_B, _BINARY_NAME_C):
        path = _write_synthetic_sorted_index(
            memmap_dir, name, sampleable_indices, length,
        )
        readers[name] = SortedIndexReader(
            path, reduction=LengthReduction(ReductionKind.MAX), depth=3,
        )
    return MultiBinarySortedIndexSampler(readers)


# ---------------------------------------------------------------------------
# open_length_bucketed_batch: ONE collector across 3 binaries
# ---------------------------------------------------------------------------


def test_open_length_bucketed_batch_uses_one_collector_per_call(
    tmp_path: Path,
) -> None:
    """The batch-load helper must construct + flush a
    :class:`BucketedRunLengthCollector` EXACTLY ONCE regardless of how
    many per-binary ``batch_decode`` calls it makes internally.

    Forces the helper to span THREE binaries (so the internal loop
    iterates >= 3 times) and asserts each counter equals 1.
    """
    memmap_dir = _build_three_binary_fixture(tmp_path)
    target_length = 50
    sampler = _open_three_binary_sampler(memmap_dir, target_length)
    factory = _make_session_factory(memmap_dir)
    # Large enough batch_size that the sampler distributes pointers to
    # all three binaries with high probability.
    rng = np.random.default_rng(1234)
    batch_size = 9
    num_variants_per_section = 2
    context_len = 32

    counting = _CountingCollectorFactory()
    with patch(
        "tokenizer.aligned_data.sorted_index._sampler._batch."
        "BucketedRunLengthCollector",
        counting.cls,
    ):
        open_length_bucketed_batch(
            factory,
            sampler,
            target_length=target_length,
            batch_size=batch_size,
            context_len=context_len,
            num_variants_per_section=num_variants_per_section,
            max_depth=2,
            rng=rng,
        )

    assert counting.n_constructed == 1, (
        f"open_length_bucketed_batch must construct exactly one "
        f"BucketedRunLengthCollector per call; got "
        f"{counting.n_constructed}"
    )
    assert counting.n_flushed == 1, (
        f"open_length_bucketed_batch must flush its collector exactly "
        f"once per call; got {counting.n_flushed}"
    )


# ---------------------------------------------------------------------------
# compute_reduced_lengths: ONE collector across multiple chunks
# ---------------------------------------------------------------------------


def test_compute_reduced_lengths_uses_one_collector_per_call(
    tmp_path: Path,
) -> None:
    """The sorted-index build helper must construct + flush a
    :class:`BucketedRunLengthCollector` EXACTLY ONCE regardless of how
    many internal CHUNK_SIZE-sized chunks the populated-sections list
    is split into.

    Forces multiple chunks by patching :data:`CHUNK_SIZE` down to 2 so
    the combined fixture's 4 populated sections produce 2 chunks. (The
    real CHUNK_SIZE is 64; a single tiny fixture would otherwise fit
    in one chunk and not exercise the lift.)
    """
    base = build_combined_fixture(tmp_path)
    dataset = BinaryDataset(base, "sortbin", vocab_manager=None)
    section_info = read_section_variant_info(base, "sortbin")

    counting = _CountingCollectorFactory()
    with patch(
        "tokenizer.aligned_data.sorted_index._length_compute."
        "BucketedRunLengthCollector",
        counting.cls,
    ), patch(
        "tokenizer.aligned_data.sorted_index._length_compute.CHUNK_SIZE",
        2,
    ):
        with dataset.open_session() as session:
            compute_reduced_lengths(
                session,
                section_info=section_info,
                depths=[3],
                reductions=[
                    LengthReduction(kind=ReductionKind.MAX),
                    LengthReduction(
                        kind=ReductionKind.PERCENTILE, percentile=50,
                    ),
                ],
            )

    assert counting.n_constructed == 1, (
        f"compute_reduced_lengths must construct exactly one "
        f"BucketedRunLengthCollector per call; got "
        f"{counting.n_constructed}"
    )
    assert counting.n_flushed == 1, (
        f"compute_reduced_lengths must flush its collector exactly "
        f"once per call; got {counting.n_flushed}"
    )
