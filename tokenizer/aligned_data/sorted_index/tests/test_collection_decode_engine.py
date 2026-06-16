"""Selectable decode-engine tests for :class:`IndexedMemmapCollection`.

The collection's ``load_batch`` routes the per-binary decode through a
selectable :class:`DecodeEngine`. Two properties are pinned here:

* **Default-unchanged**: ``load_batch`` with no ``engine`` arg takes the
  :attr:`DecodeEngine.BATCH_DECODE` path -- the historical staged
  collector/flush/finalise decode. Asserted by spying the engine that
  reaches :func:`decode_pointer_batch`.
* **Engine byte-identity**: ``engine=VECTOR_BATCH`` produces output
  ``np.array_equal`` to ``engine=BATCH_DECODE`` for EVERY array the
  cross-binary result carries -- the token tensor, the
  ``(section_idx, variant_idx)`` mapping, the per-row ``binary_id``
  sidecar + ``binary_names`` reverse map, and every dense identity /
  number sidecar (data + offsets), plus the optional FID sidecars. This
  mirrors the single-binary ``bench_decode`` byte-identity gate one level
  up at the cross-binary collection assembly.

Both run over a tiny multi-dir synthetic corpus built via the production
fixture machinery (``_build_decodable_binary`` -- combined fixture + the
real ``write_sorted_index_files`` builder + the ``ensure_sidecar``
realized-geometry sidecars that the vector_batch engine reads), so the
decode actually exercises the geometry -> scatter -> dense pipeline, not
a stub.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.realized_lengths import generate_realized_geometry
from tokenizer.aligned_data.sorted_index import IndexedMemmapCollection
from tokenizer.aligned_data.sorted_index._sampler import DecodeEngine

from .test_collection import _BAND, _EMPTY_LENGTH, _build_decodable_binary
from .fixtures import make_test_vocab_manager
from . import test_collection as _collection_tests


# Arrays compared for byte-identity: the cross-binary sidecars + every
# inner BatchDecodeResult dense array. fid arrays are exercised by passing
# include_fid_sidecar=True below.
_INNER_DENSE_FIELDS = (
    "tokens",
    "batch_idx_to_section_variant",
    "identities",
    "identity_row_offsets",
    "numbers_significant",
    "numbers_sign_exponent",
    "number_row_offsets",
    "fid_sidecar",
    "fid_row_offsets",
    "fid_per_category_counts",
)


def _build_vector_batch_binary(
    memmap_dir: Path, binary_name: str, scratch: Path
) -> None:
    """A decodable binary PLUS both arms' realized-geometry sidecars.

    ``_build_decodable_binary`` lays down the combined fixture + real
    sorted index + the realized-LENGTH sidecars the index build needs;
    the vector_batch engine additionally reads the realized-GEOMETRY
    (RLG3) pair per arm, so this generates those too.
    """
    _build_decodable_binary(memmap_dir, binary_name, scratch)
    generate_realized_geometry(memmap_dir, binary_name)


def _build_two_dir_corpus(tmp_path: Path) -> tuple[Path, Path]:
    dir_a = tmp_path / "pkgA"
    dir_b = tmp_path / "pkgB"
    dir_a.mkdir()
    dir_b.mkdir()
    _build_vector_batch_binary(dir_a, "alpha", tmp_path / "scratch_a")
    _build_vector_batch_binary(dir_b, "beta", tmp_path / "scratch_b")
    return dir_a, dir_b


def _load(coll: IndexedMemmapCollection, engine: DecodeEngine, seed: int):
    """One seeded band-draw load_batch on ``engine`` (fid sidecar ON)."""
    return coll.load_batch(
        _EMPTY_LENGTH,
        6,
        rng=np.random.default_rng(seed),
        band=_BAND,
        context_len=32,
        num_variants_per_section=2,
        max_depth=_collection_tests._DEPTH,
        include_fid_sidecar=True,
        engine=engine,
    )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_vector_batch_engine_byte_identical_to_batch_decode(
    tmp_path: Path, seed: int
) -> None:
    """VECTOR_BATCH == BATCH_DECODE across every cross-binary array."""
    dir_a, dir_b = _build_two_dir_corpus(tmp_path)
    with IndexedMemmapCollection.discover(
        [dir_a, dir_b],
        reduction=_collection_tests._MAX,
        depth=_collection_tests._DEPTH,
        vocab_manager=make_test_vocab_manager(),
    ) as coll:
        bd = _load(coll, DecodeEngine.BATCH_DECODE, seed)
        vb = _load(coll, DecodeEngine.VECTOR_BATCH, seed)

    # Cross-binary identity sidecars.
    assert bd.binary_names == vb.binary_names
    assert np.array_equal(bd.binary_id_per_row, vb.binary_id_per_row)

    # Every inner dense array (incl. fid sidecars, which we requested).
    for field in _INNER_DENSE_FIELDS:
        a = getattr(bd.inner, field)
        b = getattr(vb.inner, field)
        assert (a is None) == (b is None), (
            f"{field}: one engine produced None, the other did not"
        )
        if a is None:
            continue
        assert a.shape == b.shape, (
            f"{field}: shape bd={a.shape} vb={b.shape}"
        )
        assert a.dtype == b.dtype, (
            f"{field}: dtype bd={a.dtype} vb={b.dtype}"
        )
        assert np.array_equal(a, b), f"{field}: arrays diverge between engines"

    # The fid sidecars were genuinely populated (guards against a vacuous
    # both-None pass that would hide a real divergence).
    assert bd.inner.fid_sidecar is not None


def test_load_batch_default_engine_is_batch_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``engine`` arg routes through DecodeEngine.BATCH_DECODE."""
    dir_a, dir_b = _build_two_dir_corpus(tmp_path)

    seen: list[DecodeEngine] = []
    import tokenizer.aligned_data.sorted_index._collection._collection as mod

    real = mod.decode_pointer_batch

    def _spy(*args, **kwargs):
        seen.append(kwargs["engine"])
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "decode_pointer_batch", _spy)

    with IndexedMemmapCollection.discover(
        [dir_a, dir_b],
        reduction=_collection_tests._MAX,
        depth=_collection_tests._DEPTH,
        vocab_manager=make_test_vocab_manager(),
    ) as coll:
        coll.load_batch(
            _EMPTY_LENGTH,
            6,
            rng=np.random.default_rng(0),
            band=_BAND,
            context_len=32,
            num_variants_per_section=2,
            max_depth=_collection_tests._DEPTH,
        )

    assert seen == [DecodeEngine.BATCH_DECODE]
