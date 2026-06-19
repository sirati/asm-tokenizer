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
from tokenizer.aligned_data.realized_lengths import generate_realized_geometry
from tokenizer.aligned_data.sorted_index._builder import write_sorted_index_files
from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    build_corpus_with_registry,
)
from tokenizer.aligned_data.loader.tests._corpus.specs import VariantSpec

from .fixtures import _DeterministicVariantRegistry, make_test_vocab_manager
from ._length_helpers import ensure_sidecar

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


# ---------------------------------------------------------------------------
# VC2-terminal-section decode (the ``valued_const_v2`` carrier-at-tail case).
#
# A section whose decodable body ends on a payload-less ``valued_const_v2``
# (vocab id 257, no following inline-digit run) drives the VC2 emitter's
# ``L = runlen_number[p_carrier + 1]`` ALG-8 lookahead toward the end of the
# per-segment runlength concatenation. These tests build such a section with
# a SYNTHETIC corpus (no shared build_memmap needed) and decode it through
# BOTH engines so a regression in the VC2 ``+1`` bounds guard surfaces here:
#
# * BATCH_DECODE routes the multi-section concat through ``emit_vc2_rows``
#   (the emitter under fix); the per-segment guard makes a terminal carrier
#   read ``L = 0`` instead of the neighbour segment's value / off-the-end.
# * load_batch_cross_depth (VECTOR_BATCH-only) mixes depths {0,1,3} so a
#   depth group can place the VC2-terminal section last; it must decode with
#   well-formed number sidecars.
#
# The token streams keep ONE trailing instruction-rep token after the VC2
# carrier so the per-call_target ``_promote_vc2`` tail guard (which rejects a
# carrier at the absolute last raw position) accepts the body; the VC2 still
# carries a zero-length payload, exercising the empty-run ALG-8 path.
# ---------------------------------------------------------------------------

_VC2_RAW = 257
_MAX = LengthReduction(ReductionKind.MAX)
_XDEPTHS = (0, 1, 3)


def _vc2_terminal_variant(vkey, seed_base: int, n_real: int) -> VariantSpec:
    """A variant body ending on a payload-less VC2 carrier (+ 1 real tail)."""
    base = 272 + (seed_base + 1) * 100
    body = list(range(base, base + n_real))
    tokens = np.array(body + [_VC2_RAW, base + n_real], dtype=np.uint16)
    n = int(tokens.shape[0])
    return VariantSpec(
        vkey=vkey,
        tokens=tokens,
        block_rl=np.array([n], dtype=np.uint8),
        insn_rl=np.array([2, n - 2], dtype=np.uint8),
    )


def _simple_variant(vkey, seed_base: int, n_tokens: int) -> VariantSpec:
    base = 272 + (seed_base + 1) * 100
    tokens = np.arange(base, base + n_tokens, dtype=np.uint16)
    return VariantSpec(
        vkey=vkey,
        tokens=tokens,
        block_rl=np.array([n_tokens], dtype=np.uint8),
        insn_rl=np.array([2, n_tokens - 2], dtype=np.uint8),
    )


def _build_vc2_terminal_binary(memmap_dir: Path, binary_name: str) -> None:
    """A binary with a VC2-terminal matched section + the index/geometry
    sidecars both engines read, over the three cross-depth specs."""
    memmap_dir.mkdir(parents=True, exist_ok=True)
    matched = (
        MatchedFunctionSpec(
            func_name="vc2_term",
            variants=(
                _vc2_terminal_variant(("vc2_term", 0), 0, 6),
                _vc2_terminal_variant(("vc2_term", 1), 1, 8),
            ),
            called=(),
        ),
        MatchedFunctionSpec(
            func_name="plain",
            variants=(
                _simple_variant(("plain", 0), 5, 7),
                _simple_variant(("plain", 1), 6, 9),
            ),
            called=(),
        ),
    )
    build_corpus_with_registry(
        memmap_dir,
        binary_name,
        matched=matched,
        unmatched=(),
        variants=_DeterministicVariantRegistry(),
    )
    ensure_sidecar(memmap_dir, binary_name)
    write_sorted_index_files(
        memmap_dir, binary_name, reductions=[_MAX], depths=list(_XDEPTHS)
    )
    generate_realized_geometry(memmap_dir, binary_name)


def test_vc2_terminal_section_batch_decode_decodes(tmp_path: Path) -> None:
    """BATCH_DECODE over a multi-section concat with a VC2-terminal section
    decodes (no IndexError / neighbour-misread) and emits VC2 number rows.

    This is the engine that routes through ``emit_vc2_rows``; the per-segment
    ``+1`` lookahead guard keeps a VC2 carrier at a segment tail reading a
    zero-length payload instead of running off the flat runlength array.
    """
    base = tmp_path / "vc2bin"
    _build_vc2_terminal_binary(base, "vc2bin")
    spec = IndexSpec(reduction=_MAX, depth=3)
    with IndexedMemmapCollection.discover(
        [base],
        specs=[spec],
        on_missing=MissingIndexPolicy.SKIP_WITH_ERROR_LOG,
        vocab_manager=make_test_vocab_manager(),
    ) as coll:
        # Sweep seeds: each lands a different per-binary concat order, so a
        # VC2-terminal segment surfaces as the last kept call_target.
        for seed in range(8):
            res = coll.load_batch(
                0,
                8,
                rng=np.random.default_rng(seed),
                band=_WIDE_BAND,
                context_len=64,
                num_variants_per_section=2,
                max_depth=3,
                spec=spec,
                engine=DecodeEngine.BATCH_DECODE,
                include_fid_sidecar=True,
            )
            assert res.inner.tokens.ndim == 2
            # The VC2-terminal section contributes number rows; the sidecar
            # offsets stay well-formed (monotone, last == #significand rows).
            sig = res.inner.numbers_significant
            off = res.inner.number_row_offsets
            assert sig is not None and off is not None
            assert off.ndim == 1 and off[0] == 0
            assert np.all(np.diff(off) >= 0)
            assert int(off[-1]) == int(sig.shape[0])


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_vc2_terminal_section_cross_depth_decodes(
    tmp_path: Path, seed: int
) -> None:
    """``load_batch_cross_depth`` over a VC2-terminal-section fixture mixes
    depths {0,1,3} and decodes with well-formed number sidecars -- a depth
    group can place the VC2-terminal section terminal in its concat.
    """
    base = tmp_path / "vc2bin"
    _build_vc2_terminal_binary(base, "vc2bin")
    specs = [IndexSpec(reduction=_MAX, depth=d) for d in _XDEPTHS]
    with IndexedMemmapCollection.discover(
        [base],
        specs=specs,
        on_missing=MissingIndexPolicy.SKIP_WITH_ERROR_LOG,
        vocab_manager=make_test_vocab_manager(),
    ) as coll:
        res = coll.load_batch_cross_depth(
            0,
            16,
            rng=np.random.default_rng(seed),
            band=_WIDE_BAND,
            context_len=64,
            num_variants_per_section=1,
            include_fid_sidecar=True,
        )
    assert res.inner.tokens.ndim == 2
    assert res.inner.tokens.shape[1] == 64
    sig = res.inner.numbers_significant
    off = res.inner.number_row_offsets
    assert sig is not None and off is not None
    assert off[0] == 0 and np.all(np.diff(off) >= 0)
    assert int(off[-1]) == int(sig.shape[0])
