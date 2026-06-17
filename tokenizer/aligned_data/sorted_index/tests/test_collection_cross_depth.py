"""Cross-depth ``load_batch`` smoke on the shared real memmaps.

The collection's cross-depth path draws across EVERY configured spec at
once (the cross-(binary x spec) urn), so one batch mixes sections from
different depths, each decoded at its OWN depth via the vector_batch
engine. These tests run against the shared corpus at
``/home/.../out/build_memmap`` (p75 reduction, depths d000/d001/d003);
they SKIP when that path is absent so the suite stays hermetic on a box
without the corpus.

What is pinned:

* the cross-depth sample mixes depths and every pointer carries its spec;
* ``load_batch_cross_depth`` returns a ``[B, L]`` batch whose rows carry
  their sampled depth's geometry (the result decodes without error and
  the row count matches the sample);
* the single-spec ``load_batch(spec=...)`` path is untouched (still
  decodes per the selected depth);
* the BATCH_DECODE engine rejects a per-pointer max_depth array (cross-
  depth is vector_batch-only) -- pinned on a small synthetic collection
  so it runs even without the shared corpus.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.sorted_index import (
    DecodeEngine,
    IndexSpec,
    IndexedMemmapCollection,
    LengthReduction,
    MissingIndexPolicy,
    ReductionKind,
)
from tokenizer.aligned_data.sorted_index._sampler import decode_pointer_batch

from .fixtures import make_test_vocab_manager

_BUILD_MEMMAP = Path("/home/sirati/devel/python/asm-tokenizer/out/build_memmap")
_P75 = LengthReduction(ReductionKind.PERCENTILE, 75)
_SPECS = [IndexSpec(reduction=_P75, depth=d) for d in (0, 1, 3)]
# A band wide enough to span every depth's length universe.
_WIDE_BAND = (1, 10_000_000)

_skip_no_corpus = pytest.mark.skipif(
    not _BUILD_MEMMAP.is_dir(),
    reason=f"shared build_memmap corpus absent at {_BUILD_MEMMAP}",
)


def _open_collection() -> IndexedMemmapCollection:
    """Discover the shared corpus for the three p75 depths.

    ``SKIP_WITH_ERROR_LOG`` drops the ``*_realized`` sidecar phantoms the
    flat-dir ``<binary>_index.bin`` scan picks up (they carry no ``.idx``
    for any spec, so they fail the uniform-membership gate cleanly).
    """
    return IndexedMemmapCollection.discover(
        [_BUILD_MEMMAP],
        specs=_SPECS,
        on_missing=MissingIndexPolicy.SKIP_WITH_ERROR_LOG,
        vocab_manager=make_test_vocab_manager(),
    )


@_skip_no_corpus
def test_cross_depth_sample_mixes_depths(caplog) -> None:
    with caplog.at_level(logging.CRITICAL):
        with _open_collection() as coll:
            rng = np.random.default_rng(0)
            ptrs = coll.sample_section_pointers_cross_depth(
                0, 256, rng, band=_WIDE_BAND
            )
    assert ptrs, "cross-depth pool must be non-empty over the wide band"
    # Every pointer carries the spec it was drawn from.
    assert all(p.spec is not None for p in ptrs)
    depths = Counter(p.spec.depth for p in ptrs)
    # A genuine cross-depth draw surfaces more than one depth.
    assert len(depths) >= 2, f"expected mixed depths, got {dict(depths)}"
    assert set(depths) <= {0, 1, 3}


@_skip_no_corpus
def test_cross_depth_load_batch_returns_batch(caplog) -> None:
    with caplog.at_level(logging.CRITICAL):
        with _open_collection() as coll:
            rng = np.random.default_rng(7)
            batch_size = 32
            # Re-derive the sample (depth-agnostic) to know the row count.
            ptrs = coll.sample_section_pointers_cross_depth(
                0, batch_size, np.random.default_rng(7), band=_WIDE_BAND
            )
            assert len(ptrs) > 0
            res = coll.load_batch_cross_depth(
                0,
                batch_size,
                rng=rng,
                band=_WIDE_BAND,
                context_len=128,
                num_variants_per_section=1,
                include_fid_sidecar=True,
            )
    assert res.inner.tokens.ndim == 2
    assert res.inner.tokens.shape[1] == 128
    # One row per sampled section (num_variants_per_section=1 -> no
    # variant fan-out), in whatever per-binary concat order the decode
    # produced.
    assert res.inner.tokens.shape[0] == len(ptrs)
    assert res.binary_id_per_row.shape[0] == len(ptrs)


@_skip_no_corpus
def test_single_spec_load_batch_still_works(caplog) -> None:
    """The per-spec ``load_batch(spec=...)`` path is untouched."""
    with caplog.at_level(logging.CRITICAL):
        with _open_collection() as coll:
            rng = np.random.default_rng(3)
            res = coll.load_batch(
                0,
                16,
                rng=rng,
                band=_WIDE_BAND,
                context_len=128,
                num_variants_per_section=1,
                max_depth=3,
                spec=_SPECS[2],  # d3
                engine=DecodeEngine.VECTOR_BATCH,
            )
    assert res.inner.tokens.shape[1] == 128
    assert res.inner.tokens.shape[0] > 0


def test_batch_decode_rejects_per_pointer_depth_array() -> None:
    """The staged engine has no per-row depth seam -> NotImplementedError.

    Pinned on a tiny synthetic input (no shared corpus needed): a
    per-pointer ``max_depth`` array under ``engine=BATCH_DECODE`` must
    raise, since cross-depth decoding is vector_batch-only.
    """
    from tokenizer.aligned_data.loader.batch_decode._types import (
        SectionPointerSpec,
    )
    from tokenizer.aligned_data.loader.metadata_loader import SectionKind
    from tokenizer.aligned_data.sorted_index._types import (
        MultiBinarySectionPointer,
    )

    pointers = [
        MultiBinarySectionPointer(
            binary_name="bin",
            section_pointer=SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
        ),
        MultiBinarySectionPointer(
            binary_name="bin",
            section_pointer=SectionPointerSpec(arm=SectionKind.MATCHED, idx=1),
        ),
    ]
    with pytest.raises(NotImplementedError, match="VECTOR_BATCH-only"):
        decode_pointer_batch(
            {"bin": object()},  # never reached: the depth check fires first
            pointers,
            context_len=64,
            num_variants_per_section=1,
            max_depth=np.array([0, 3], dtype=np.int64),
            rng=np.random.default_rng(0),
            engine=DecodeEngine.BATCH_DECODE,
        )
