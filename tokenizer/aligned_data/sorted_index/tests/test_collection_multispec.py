"""Multi-spec tests for :class:`IndexedMemmapCollection`.

A collection binds N ``(reduction, depth)`` :class:`IndexSpec` pairs:
one :class:`SortedIndexReader` + one sampler PER spec, sharing ONE
member/session pool. These tests cover the spec-list boundary
(``specs`` XOR ``reduction/depth``), uniform membership across specs,
per-spec pool divergence on a real call graph, the shared session
pool, spec resolution, and per-spec e2e decode.

Index construction is the production :func:`write_sorted_index_files`
builder run with several ``depths`` in one call -- the same heavy
graph traversal that emits every depth's ``.idx`` for one binary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    VariantSpec,
    build_corpus,
    make_simple_variant,
)
from tokenizer.aligned_data.sorted_index import (
    IndexSpec,
    IndexedMemmapCollection,
    LengthReduction,
    MissingIndexPolicy,
    ReductionKind,
    write_sorted_index_files,
)

from .fixtures import build_combined_fixture


_MAX = LengthReduction(ReductionKind.MAX)
_DEPTHS = [0, 1, 3]
_SPEC_D0 = IndexSpec(reduction=_MAX, depth=0)
_SPEC_D1 = IndexSpec(reduction=_MAX, depth=1)
_SPEC_D3 = IndexSpec(reduction=_MAX, depth=3)
_ALL_SPECS = [_SPEC_D0, _SPEC_D1, _SPEC_D3]

# Combined-fixture populated length (probed + pinned in test_collection.py):
# section idx 0 is the 0-variant trap; len 9 holds count 2. The
# combined fixture has no call edges that splice, so its key is depth-
# invariant -- ideal for tests where only membership / spec resolution
# matters (not depth-dependent length growth).
_COMBINED_LEN = 9


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------


def _place_combined_binary(memmap_dir: Path, binary_name: str, scratch: Path) -> None:
    """Lay one ``build_combined_fixture`` corpus into ``memmap_dir``."""
    combined_base = build_combined_fixture(scratch)
    for entry in combined_base.iterdir():
        if not entry.is_file() or not entry.name.startswith("sortbin"):
            continue
        new_name = binary_name + entry.name[len("sortbin"):]
        (memmap_dir / new_name).write_bytes(entry.read_bytes())


def _build_multidepth_binary(
    memmap_dir: Path, binary_name: str, scratch: Path, *, depths=_DEPTHS
) -> None:
    """Place a combined-corpus binary + its REAL multi-depth indexes."""
    _place_combined_binary(memmap_dir, binary_name, scratch)
    write_sorted_index_files(
        memmap_dir, binary_name, reductions=[_MAX], depths=list(depths),
    )


def _shared_variant(seed: int):
    """A variant with a fixed vkey shared across a chain's functions.

    A per-call entry's callee vkey is the caller variant's own vkey; for
    a splice edge to survive (so depth grows the spliced length) the
    callee must carry a variant with that same vkey. Sharing one vkey
    across the whole chain makes every edge land.
    """
    return make_simple_variant(("shared", 0), token_seed=seed, n_tokens=6)


def _chain_variants(seed: int, nxt):
    """Two shared-vkey variants: v0 calls the chain's next function, v1
    is quiet.

    The once-only / all-variants-equivalence walk excludes a callee
    reached by EVERY variant; a single-variant chain would splice
    nothing (FLAG-A). Pairing the caller variant with a quiet sibling
    makes each edge "some but not all" so depth still grows the spliced
    length. Both share the ``("shared", 0)`` vkey so the edge's per-call
    J lands on the callee's matching variant.
    """
    v0 = make_simple_variant(("shared", 0), token_seed=seed, n_tokens=6)
    v1 = make_simple_variant(("shared", 0), token_seed=seed + 5, n_tokens=7)
    return (
        VariantSpec(
            vkey=v0.vkey, tokens=v0.tokens, block_rl=v0.block_rl,
            insn_rl=v0.insn_rl, called=(nxt,) if nxt else (),
        ),
        VariantSpec(
            vkey=("shared", 1), tokens=v1.tokens, block_rl=v1.block_rl,
            insn_rl=v1.insn_rl, called=(),
        ),
    )


def _build_chain_binary(
    memmap_dir: Path, binary_name: str, *, depths=_DEPTHS
) -> None:
    """Place a 5-function call chain whose spliced length grows with depth.

    ``a -> b -> c -> d -> e``; each function has a caller variant 0
    (calls the next) + a quiet sibling variant 1, so depth-d3 lengths
    strictly exceed depth-d0 lengths for variant 0. Probed bucket layout
    (max reduction): a band above the d0 keys captures d3 sections but
    no d0 section.
    """
    specs = [
        MatchedFunctionSpec(
            func_name=name,
            variants=_chain_variants(10 * i + 1, nxt),
            called=(nxt,) if nxt else (),
        )
        for i, (name, nxt) in enumerate(
            [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", None)]
        )
    ]
    build_corpus(memmap_dir, binary_name, matched=specs)
    write_sorted_index_files(
        memmap_dir, binary_name, reductions=[_MAX], depths=list(depths),
    )


# ---------------------------------------------------------------------------
# 1. Multi-spec discovery: specs property + member parity with single-spec
# ---------------------------------------------------------------------------


def test_discover_multispec_specs_and_members(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_b = tmp_path / "pkgB"
    dir_a.mkdir()
    dir_b.mkdir()
    _build_multidepth_binary(dir_a, "alpha", tmp_path / "scratch_a")
    _build_multidepth_binary(dir_b, "beta", tmp_path / "scratch_b")

    with IndexedMemmapCollection.discover(
        [dir_a, dir_b], specs=[_SPEC_D3, _SPEC_D0, _SPEC_D1]
    ) as coll:
        # Stable order: sorted by (filename_tag, depth) regardless of
        # the order specs were passed in.
        assert coll.specs == [_SPEC_D0, _SPEC_D1, _SPEC_D3]
        multi_members = [m.qualified_name for m in coll.members]
        assert multi_members == ["alpha", "beta"]

    # Members identical to a single-spec discovery of the same dirs.
    with IndexedMemmapCollection.discover(
        [dir_a, dir_b], reduction=_MAX, depth=0
    ) as single:
        single_members = [m.qualified_name for m in single.members]
    assert multi_members == single_members
    assert coll.binary_names == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# 2. Boundary validation
# ---------------------------------------------------------------------------


def test_boundary_both_forms_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "alpha", tmp_path / "scratch_a")
    with pytest.raises(ValueError, match="not both"):
        IndexedMemmapCollection.discover(
            [dir_a], specs=[_SPEC_D0], reduction=_MAX, depth=0
        )


def test_boundary_neither_form_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "alpha", tmp_path / "scratch_a")
    with pytest.raises(ValueError, match="provide specs"):
        IndexedMemmapCollection.discover([dir_a])


def test_boundary_duplicate_specs_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "alpha", tmp_path / "scratch_a")
    with pytest.raises(ValueError, match="duplicate spec"):
        IndexedMemmapCollection.discover(
            [dir_a], specs=[_SPEC_D0, _SPEC_D0]
        )


def test_convenience_form_equivalent_to_single_spec_list(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "alpha", tmp_path / "scratch_a")

    with IndexedMemmapCollection.discover(
        [dir_a], reduction=_MAX, depth=0
    ) as conv, IndexedMemmapCollection.discover(
        [dir_a], specs=[_SPEC_D0]
    ) as listed:
        assert conv.specs == listed.specs == [_SPEC_D0]
        # spec=None resolves to the single spec for both forms.
        assert conv.count_at(_COMBINED_LEN) == listed.count_at(_COMBINED_LEN)
        assert conv.count_at(_COMBINED_LEN) > 0


# ---------------------------------------------------------------------------
# 3. Per-spec pools differ on a real call graph (d3 lengths exceed d0)
# ---------------------------------------------------------------------------


def test_per_spec_pools_differ_on_call_graph(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_chain_binary(dir_a, "chain")

    with IndexedMemmapCollection.discover(
        [dir_a], specs=[_SPEC_D0, _SPEC_D3]
    ) as coll:
        # Band above the d0 universe (everything sits at key 7) but
        # inside the d3 universe (keys 21 + 28).
        band = (15, 100)
        d0_count = coll.count_in_band(*band, spec=_SPEC_D0)
        d3_count = coll.count_in_band(*band, spec=_SPEC_D3)
        assert d0_count == 0
        assert d3_count == 3  # keys 21 (1) + 28 (2)
        assert d0_count != d3_count

        # Sampling spec=d3 draws from the d3 universe (the band is
        # empty for d0, so a d0 sample would be empty).
        rng = np.random.default_rng(7)
        d3_ptrs = coll.sample_section_pointers(
            0, 8, rng, band=band, spec=_SPEC_D3
        )
        assert len(d3_ptrs) == 3
        d0_ptrs = coll.sample_section_pointers(
            0, 8, rng, band=band, spec=_SPEC_D0
        )
        assert d0_ptrs == []


# ---------------------------------------------------------------------------
# 4. Uniform membership: a binary missing only its d3 idx
# ---------------------------------------------------------------------------


def _drop_idx(memmap_dir: Path, binary_name: str, spec: IndexSpec) -> None:
    path = (
        memmap_dir
        / f"{binary_name}_sorted_{spec.reduction.filename_tag()}"
        f"_d{spec.depth:03d}.idx"
    )
    path.unlink()


def test_uniform_membership_missing_d3_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "good", tmp_path / "scratch_g")
    _build_multidepth_binary(dir_a, "halfblind", tmp_path / "scratch_h")
    _drop_idx(dir_a, "halfblind", _SPEC_D3)

    with pytest.raises(ValueError) as exc:
        IndexedMemmapCollection.discover([dir_a], specs=_ALL_SPECS)
    msg = str(exc.value)
    # Names the (dir, binary, d3-tag) triple; never the present specs.
    assert "halfblind" in msg
    assert "max_d003" in msg
    assert "good" not in msg
    assert "max_d000" not in msg


def test_uniform_membership_missing_d3_skip_drops_from_all_specs(
    tmp_path: Path, caplog
) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "good", tmp_path / "scratch_g")
    _build_multidepth_binary(dir_a, "halfblind", tmp_path / "scratch_h")
    _drop_idx(dir_a, "halfblind", _SPEC_D3)

    with caplog.at_level(logging.ERROR):
        with IndexedMemmapCollection.discover(
            [dir_a],
            specs=_ALL_SPECS,
            on_missing=MissingIndexPolicy.SKIP_WITH_ERROR_LOG,
        ) as coll:
            # Excluded from the WHOLE collection, including d0 where its
            # idx is present.
            assert [m.qualified_name for m in coll.members] == ["good"]
            assert coll.count_at(_COMBINED_LEN, spec=_SPEC_D0) > 0
            # halfblind contributes nothing at any spec.
            good_only = coll.count_at(_COMBINED_LEN, spec=_SPEC_D0)
            assert good_only == coll.count_at(_COMBINED_LEN, spec=_SPEC_D1)

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    joined = " ".join(r.getMessage() for r in error_records)
    assert "halfblind" in joined
    assert "max_d003" in joined  # names exactly the missing spec
    assert "max_d000" not in joined  # not the present ones


# ---------------------------------------------------------------------------
# 5. Spec resolution
# ---------------------------------------------------------------------------


def test_spec_none_with_one_spec_works(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "alpha", tmp_path / "scratch_a")
    with IndexedMemmapCollection.discover(
        [dir_a], specs=[_SPEC_D0]
    ) as coll:
        assert coll.count_at(_COMBINED_LEN) > 0  # spec=None resolves to the lone spec


def test_spec_none_with_several_specs_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "alpha", tmp_path / "scratch_a")
    with IndexedMemmapCollection.discover(
        [dir_a], specs=_ALL_SPECS
    ) as coll:
        with pytest.raises(ValueError) as exc:
            coll.count_at(_COMBINED_LEN)
        msg = str(exc.value)
        assert "ambiguous" in msg
        # Lists every configured spec, in stable order.
        assert "max_d000" in msg
        assert "max_d001" in msg
        assert "max_d003" in msg


def test_unknown_spec_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_multidepth_binary(dir_a, "alpha", tmp_path / "scratch_a")
    with IndexedMemmapCollection.discover(
        [dir_a], specs=[_SPEC_D0, _SPEC_D1]
    ) as coll:
        with pytest.raises(ValueError) as exc:
            coll.count_at(_COMBINED_LEN, spec=_SPEC_D3)
        msg = str(exc.value)
        assert "not configured" in msg
        assert "max_d003" in msg


# ---------------------------------------------------------------------------
# 6. Shared sessions across specs
# ---------------------------------------------------------------------------


def test_sessions_shared_across_specs(tmp_path: Path, monkeypatch) -> None:
    """load_batch(spec=d0) then load_batch(spec=d3) opens each binary once.

    The session pool is collection-level: a binary sampled by the d0
    batch AND the d3 batch reuses the SAME warm session.
    """
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_chain_binary(dir_a, "alpha")
    _build_chain_binary(dir_a, "beta")

    from collections import Counter

    open_counts: Counter = Counter()
    real_open = BinaryDataset.open_session

    def counting_open(self):
        open_counts[self.binary_name] += 1
        return real_open(self)

    monkeypatch.setattr(BinaryDataset, "open_session", counting_open)

    with IndexedMemmapCollection.discover(
        [dir_a], specs=[_SPEC_D0, _SPEC_D3]
    ) as coll:
        rng = np.random.default_rng(1)
        # batch_size 10 == both binaries' full key-8 pool (5 each) so
        # the d0 draw saturates and samples every section of both
        # binaries; the d3 band draw re-samples the d3-spliced sections.
        # (d0 MAX key is 8 = 1 self-token + the quiet sibling's 7-token
        # body, the larger of the two variants.)
        coll.load_batch(
            8, 10, rng=rng, spec=_SPEC_D0,
            context_len=16, num_variants_per_section=1, max_depth=1,
        )
        coll.load_batch(
            0, 10, rng=rng, band=(15, 100), spec=_SPEC_D3,
            context_len=16, num_variants_per_section=1, max_depth=3,
        )
        # Both binaries were sampled across the two specs ...
        assert set(open_counts) == {"alpha", "beta"}
        # ... and each opened exactly once (one shared session per member).
        for name, count in open_counts.items():
            assert count == 1, (
                f"{name} opened {count} times; one shared session must "
                "serve every spec"
            )


# ---------------------------------------------------------------------------
# 7. load_batch e2e per spec: d0 and d3 both decode, differ in length char
# ---------------------------------------------------------------------------


def test_load_batch_e2e_differs_per_spec(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_chain_binary(dir_a, "chain")

    with IndexedMemmapCollection.discover(
        [dir_a], specs=[_SPEC_D0, _SPEC_D3]
    ) as coll:
        # d0: exact bucket at key 8 (every section) decodes.
        rng = np.random.default_rng(11)
        res_d0 = coll.load_batch(
            8, 5, rng=rng, spec=_SPEC_D0,
            context_len=64, num_variants_per_section=1, max_depth=1,
        )
        assert res_d0.inner.tokens.shape[0] == 5

        # d3: the high band (keys 21/28) draws ONLY the deeply-spliced
        # sections -- a length character the d0 pool at key 7 cannot
        # produce. Assert via the sampled index keys, not decode internals.
        d3_band = (15, 100)
        d3_ptrs = coll.sample_section_pointers(
            0, 8, rng, band=d3_band, spec=_SPEC_D3
        )
        d0_band_ptrs = coll.sample_section_pointers(
            0, 8, rng, band=d3_band, spec=_SPEC_D0
        )
        assert len(d3_ptrs) > 0
        assert d0_band_ptrs == []  # the d0 universe has nothing in this band

        res_d3 = coll.load_batch(
            0, 8, rng=rng, band=d3_band, spec=_SPEC_D3,
            context_len=64, num_variants_per_section=1, max_depth=3,
        )
        assert res_d3.inner.tokens.shape[0] == len(d3_ptrs)
