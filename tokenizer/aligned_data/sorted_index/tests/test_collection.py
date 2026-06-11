"""End-to-end tests for :class:`IndexedMemmapCollection` (collection layer).

The collection layer assembles a corpus of per-binary indexed memmap
directories into ONE unbiased length-bucketed batch source. These tests
build tiny synthetic corpora via the production fixture machinery
(``build_combined_fixture`` + the real ``write_sorted_index_files``
builder), lay binaries across multiple memmap directories, and exercise
discovery, naming, missing-index policy, unbiasedness, persistent
sessions, the full decode, and close semantics.

Two index-construction styles appear:

* Real ``.idx`` via :func:`write_sorted_index_files` -- used wherever the
  underlying ``_data.bin`` must actually decode (the e2e path).
* Synthetic single-bucket ``.idx`` via :func:`encode_sorted_index` --
  used where only the SAMPLER pool sizes matter (the unbiasedness
  frequency test needs controlled A:B pool ratios; no decode).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pytest

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.sorted_index import (
    IndexedMemmapCollection,
    LengthReduction,
    MissingIndexPolicy,
    ReductionKind,
    encode_sorted_index,
    write_sorted_index_files,
)

from .fixtures import build_combined_fixture


_MAX = LengthReduction(ReductionKind.MAX)
_DEPTH = 3

# Combined-fixture sorted-index bucket layout (max reduction, depth 3),
# probed once and pinned: section idx 0 is the 0-variant section (len 0);
# the sampleable buckets are len 9 (count 2), len 10 (count 1),
# len 12 (count 1). The exact-bucket and band targets below use these.
_POPULATED_LENGTH = 9
_EMPTY_LENGTH = 50  # no exact bucket here; the band around it is populated.
# Band spans every NON-ZERO-variant populated length (9/10/12) while
# excluding the len-0 zero-variant section (which decode cannot sample).
_BAND = (5, 100)


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------


def _place_combined_binary(memmap_dir: Path, binary_name: str, scratch: Path) -> None:
    """Lay one ``build_combined_fixture`` corpus into ``memmap_dir``.

    ``build_combined_fixture`` hardcodes binary_name ``sortbin`` and
    writes to ``scratch/combined``; every artefact's prefix is renamed
    into ``memmap_dir`` so binaries with distinct names co-exist in one
    directory.
    """
    combined_base = build_combined_fixture(scratch)
    for entry in combined_base.iterdir():
        if not entry.is_file() or not entry.name.startswith("sortbin"):
            continue
        new_name = binary_name + entry.name[len("sortbin"):]
        (memmap_dir / new_name).write_bytes(entry.read_bytes())


def _build_decodable_binary(
    memmap_dir: Path, binary_name: str, scratch: Path
) -> None:
    """Place a combined-corpus binary + its REAL sorted index in ``memmap_dir``."""
    _place_combined_binary(memmap_dir, binary_name, scratch)
    write_sorted_index_files(
        memmap_dir, binary_name, reductions=[_MAX], depths=[_DEPTH],
    )


def _overlay_synthetic_index(
    memmap_dir: Path, binary_name: str, length: int, pool_size: int
) -> None:
    """Overwrite ``binary_name``'s ``.idx`` with a single-bucket pool.

    Puts exactly ``pool_size`` sampleable sections in the one bucket at
    ``length``; every other section is stamped at a disjoint length so it
    never lands in the band. The sampleable section indices (1..pool_size)
    are real combined-fixture sections, so the resulting pointers decode --
    but the unbiasedness frequency test only consults the SAMPLER pool
    size, never decoding. This overlay lets the pool size be controlled
    while the binary keeps its real corpus sidecars (so ``BinaryDataset``
    constructs).
    """
    # Combined fixture has 5 sections (idx 0 is the 0-variant trap); pool
    # sections are 1..pool_size, the rest park at a disjoint length.
    num_sections = max(5, pool_size + 1)
    lengths = np.full(num_sections, length + 1000, dtype=np.uint32)
    for idx in range(1, pool_size + 1):
        lengths[idx] = length
    path = memmap_dir / f"{binary_name}_sorted_{_MAX.filename_tag()}_d{_DEPTH:03d}.idx"
    path.write_bytes(encode_sorted_index(lengths))


def _touch_index_bin(memmap_dir: Path, binary_name: str) -> None:
    """Lay down a bare ``<binary>_index.bin`` (binary-exists signal only).

    Used by tests that need a binary to be DISCOVERED but whose ``.idx``
    presence is the variable under test; the synthetic ``.idx`` (if any)
    is the only sampler input, so the catalog sidecars are unnecessary.
    """
    (memmap_dir / f"{binary_name}_index.bin").write_bytes(b"")


# ---------------------------------------------------------------------------
# 1. Multi-dir discovery
# ---------------------------------------------------------------------------


def test_discovery_two_dirs_distinct_binaries(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_b = tmp_path / "pkgB"
    dir_a.mkdir()
    dir_b.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")
    _build_decodable_binary(dir_b, "beta", tmp_path / "scratch_b")

    with IndexedMemmapCollection.discover(
        [dir_a, dir_b], reduction=_MAX, depth=_DEPTH
    ) as coll:
        names = [m.qualified_name for m in coll.members]
        assert names == ["alpha", "beta"]  # alphabetical, bare
        assert coll.binary_names == ["alpha", "beta"]
        # Members carry the structured triple.
        by_name = {m.qualified_name: m for m in coll.members}
        assert by_name["alpha"].memmap_dir == dir_a
        assert by_name["alpha"].binary_name == "alpha"
        assert by_name["beta"].memmap_dir == dir_b
        assert by_name["beta"].binary_name == "beta"


# ---------------------------------------------------------------------------
# 2. Collision qualification
# ---------------------------------------------------------------------------


def test_collision_qualified_by_dirname(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_b = tmp_path / "pkgB"
    dir_a.mkdir()
    dir_b.mkdir()
    # Same binary name "shared" in BOTH dirs -> both qualified.
    _build_decodable_binary(dir_a, "shared", tmp_path / "scratch_a")
    _build_decodable_binary(dir_b, "shared", tmp_path / "scratch_b")
    # A unique one stays bare even amid a collision.
    _build_decodable_binary(dir_a, "lonely", tmp_path / "scratch_l")

    with IndexedMemmapCollection.discover(
        [dir_a, dir_b], reduction=_MAX, depth=_DEPTH
    ) as coll:
        names = [m.qualified_name for m in coll.members]
        assert names == ["lonely", "pkgA/shared", "pkgB/shared"]
        # Distinct ids, both resolvable to their dir.
        by_name = {m.qualified_name: m for m in coll.members}
        assert by_name["pkgA/shared"].memmap_dir == dir_a
        assert by_name["pkgB/shared"].memmap_dir == dir_b
        assert by_name["lonely"].qualified_name == "lonely"


def test_collision_same_dirname_raises(tmp_path: Path) -> None:
    # Two DIFFERENT parent dirs but the SAME dir.name -> qualification
    # cannot disambiguate -> refuse.
    dir_a = tmp_path / "x" / "pkg"
    dir_b = tmp_path / "y" / "pkg"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    assert dir_a.name == dir_b.name == "pkg"
    _build_decodable_binary(dir_a, "shared", tmp_path / "scratch_a")
    _build_decodable_binary(dir_b, "shared", tmp_path / "scratch_b")

    with pytest.raises(ValueError, match="ambiguous qualified name"):
        IndexedMemmapCollection.discover(
            [dir_a, dir_b], reduction=_MAX, depth=_DEPTH
        )


# ---------------------------------------------------------------------------
# 3. Missing index policy
# ---------------------------------------------------------------------------


def test_missing_index_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    # "good" gets its real index; "orphan" exists but has NO .idx.
    _build_decodable_binary(dir_a, "good", tmp_path / "scratch_g")
    _touch_index_bin(dir_a, "orphan")

    with pytest.raises(ValueError, match="missing sorted-index"):
        IndexedMemmapCollection.discover(
            [dir_a], reduction=_MAX, depth=_DEPTH
        )


def test_missing_index_skip_logs_error(tmp_path: Path, caplog) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "good", tmp_path / "scratch_g")
    _touch_index_bin(dir_a, "orphan")

    with caplog.at_level(logging.ERROR):
        with IndexedMemmapCollection.discover(
            [dir_a],
            reduction=_MAX,
            depth=_DEPTH,
            on_missing=MissingIndexPolicy.SKIP_WITH_ERROR_LOG,
        ) as coll:
            assert [m.qualified_name for m in coll.members] == ["good"]

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("orphan" in r.getMessage() for r in error_records), (
        "excluding a binary for a missing index must emit an ERROR-level "
        "record naming it"
    )


def test_missing_index_wrong_depth_raises(tmp_path: Path) -> None:
    # The index exists for depth 3 but the request is depth 7 -> missing.
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "good", tmp_path / "scratch_g")

    with pytest.raises(ValueError, match="depth=7"):
        IndexedMemmapCollection.discover(
            [dir_a], reduction=_MAX, depth=7
        )


# ---------------------------------------------------------------------------
# 4. Unbiased across dirs
# ---------------------------------------------------------------------------


def _frequency_setup(tmp_path: Path, pool_a: int, pool_b: int) -> IndexedMemmapCollection:
    """Two decodable binaries in two dirs with controlled pool sizes.

    Each is a real combined-fixture corpus (so ``BinaryDataset``
    constructs from real sidecars) whose ``.idx`` is overlaid with a
    single-bucket pool of ``pool_*`` sampleable sections. Pools stay
    <= 4 (the combined fixture's sampleable section count) so the
    referenced sections are real.
    """
    dir_a = tmp_path / "pkgA"
    dir_b = tmp_path / "pkgB"
    dir_a.mkdir()
    dir_b.mkdir()
    for d, name, pool, scratch in (
        (dir_a, "alpha", pool_a, tmp_path / "scratch_a"),
        (dir_b, "beta", pool_b, tmp_path / "scratch_b"),
    ):
        _build_decodable_binary(d, name, scratch)
        _overlay_synthetic_index(d, name, _POPULATED_LENGTH, pool)
    return IndexedMemmapCollection.discover(
        [dir_a, dir_b], reduction=_MAX, depth=_DEPTH
    )


def _draw_frequencies(coll, n_draws, target_length, *, band):
    rng = np.random.default_rng(20240101)
    picks: Counter = Counter()
    for _ in range(n_draws):
        ptrs = coll.sample_section_pointers(
            target_length, 1, rng, band=band,
        )
        assert len(ptrs) == 1
        picks[ptrs[0].binary_name] += 1
    return picks


def test_unbiased_exact_length(tmp_path: Path) -> None:
    pool_a, pool_b = 3, 1   # expected alpha share = 0.75
    with _frequency_setup(tmp_path, pool_a, pool_b) as coll:
        assert coll.count_at(_POPULATED_LENGTH) == pool_a + pool_b
        picks = _draw_frequencies(
            coll, 4000, _POPULATED_LENGTH, band=None
        )
    share_alpha = picks["alpha"] / sum(picks.values())
    assert abs(share_alpha - pool_a / (pool_a + pool_b)) < 0.05


def test_unbiased_band(tmp_path: Path) -> None:
    pool_a, pool_b = 4, 1   # expected alpha share = 0.8
    with _frequency_setup(tmp_path, pool_a, pool_b) as coll:
        # The overlaid pool sits at _POPULATED_LENGTH, inside _BAND; the
        # band count therefore equals the combined pool.
        assert coll.count_in_band(*_BAND) == pool_a + pool_b
        picks = _draw_frequencies(
            coll, 4000, _EMPTY_LENGTH, band=_BAND
        )
    share_alpha = picks["alpha"] / sum(picks.values())
    assert abs(share_alpha - pool_a / (pool_a + pool_b)) < 0.05


# ---------------------------------------------------------------------------
# 5. Persistent sessions
# ---------------------------------------------------------------------------


def test_sessions_persist_and_lazy(tmp_path: Path, monkeypatch) -> None:
    """Two load_batch calls reuse ONE session per sampled binary.

    Counts ``BinaryDataset.open_session`` calls (the only handle a member
    can reach without breaking encapsulation): a persistent session means
    a binary sampled in BOTH batches still opens exactly once.
    """
    dir_a = tmp_path / "pkgA"
    dir_b = tmp_path / "pkgB"
    dir_a.mkdir()
    dir_b.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")
    _build_decodable_binary(dir_b, "beta", tmp_path / "scratch_b")

    open_counts: Counter = Counter()
    real_open = BinaryDataset.open_session

    def counting_open(self):
        open_counts[self.binary_name] += 1
        return real_open(self)

    monkeypatch.setattr(BinaryDataset, "open_session", counting_open)

    with IndexedMemmapCollection.discover(
        [dir_a, dir_b], reduction=_MAX, depth=_DEPTH
    ) as coll:
        # batch_size 8 == the band pool (4 sampleable sections per binary
        # x 2 binaries), so the without-replacement urn draw saturates and
        # samples EVERY section of BOTH binaries each call -- making the
        # second batch a guaranteed re-sample of both binaries.
        rng = np.random.default_rng(1)
        for _ in range(2):
            coll.load_batch(
                _EMPTY_LENGTH,
                8,
                rng=rng,
                band=_BAND,
                context_len=16,
                num_variants_per_section=2,
                max_depth=2,
            )
        # Both binaries sampled in both batches -> both opened.
        assert set(open_counts) == {"alpha", "beta"}
        # Every binary that was opened was opened exactly once (reuse).
        for name, count in open_counts.items():
            assert count == 1, (
                f"{name} opened {count} times; sessions must persist across "
                "load_batch calls"
            )


def test_never_sampled_member_opens_no_session(tmp_path: Path, monkeypatch) -> None:
    """A member whose pool is empty for every draw never opens a session."""
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    # alpha is decodable + has the combined pool; idle is decodable too
    # but its index is overlaid at a DISJOINT length so it is discovered
    # yet never sampled when we draw at the combined-fixture's populated
    # length.
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")
    _build_decodable_binary(dir_a, "idle", tmp_path / "scratch_idle")
    _overlay_synthetic_index(dir_a, "idle", 7777, pool_size=4)

    opened: List[str] = []
    real_open = BinaryDataset.open_session

    def recording_open(self):
        opened.append(self.binary_name)
        return real_open(self)

    monkeypatch.setattr(BinaryDataset, "open_session", recording_open)

    with IndexedMemmapCollection.discover(
        [dir_a], reduction=_MAX, depth=_DEPTH
    ) as coll:
        assert {m.qualified_name for m in coll.members} == {"alpha", "idle"}
        rng = np.random.default_rng(0)
        coll.load_batch(
            _POPULATED_LENGTH,
            2,
            rng=rng,
            context_len=16,
            num_variants_per_section=2,
            max_depth=2,
        )
    assert "idle" not in opened, "an unsampled member must not open a session"
    assert "alpha" in opened


# ---------------------------------------------------------------------------
# 6. load_batch e2e across two dirs
# ---------------------------------------------------------------------------


def test_load_batch_e2e_two_dirs(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_b = tmp_path / "pkgB"
    dir_a.mkdir()
    dir_b.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")
    _build_decodable_binary(dir_b, "beta", tmp_path / "scratch_b")

    with IndexedMemmapCollection.discover(
        [dir_a, dir_b], reduction=_MAX, depth=_DEPTH
    ) as coll:
        rng = np.random.default_rng(42)
        num_variants = 2
        batch_size = 6
        result = coll.load_batch(
            _EMPTY_LENGTH,
            batch_size,
            rng=rng,
            band=_BAND,
            context_len=32,
            num_variants_per_section=num_variants,
            max_depth=2,
        )
        inner = result.inner
        expected_rows = batch_size * num_variants
        assert inner.tokens.shape == (expected_rows, 32)
        assert result.binary_id_per_row.shape == (expected_rows,)
        # binary_names is the qualified, alphabetical reverse map.
        assert result.binary_names == ["alpha", "beta"]
        # Every binary_id resolves into binary_names.
        assert set(int(b) for b in result.binary_id_per_row) <= {0, 1}
        # With enough draws across a 6*2 batch both dirs should appear
        # (seeded); at minimum the ids are a valid mapping. Assert the
        # mapping is internally consistent: row ids index binary_names.
        assert all(
            0 <= int(b) < len(result.binary_names)
            for b in result.binary_id_per_row
        )


def test_load_batch_band_when_exact_bucket_empty(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")

    with IndexedMemmapCollection.discover(
        [dir_a], reduction=_MAX, depth=_DEPTH
    ) as coll:
        # The exact bucket at _EMPTY_LENGTH is empty ...
        assert coll.count_at(_EMPTY_LENGTH) == 0
        # ... but the band around it is populated, so the band draw works.
        rng = np.random.default_rng(3)
        result = coll.load_batch(
            _EMPTY_LENGTH,
            3,
            rng=rng,
            band=_BAND,
            context_len=16,
            num_variants_per_section=2,
            max_depth=2,
        )
        assert result.inner.tokens.shape[0] == 3 * 2


def test_load_batch_empty_pool_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")

    with IndexedMemmapCollection.discover(
        [dir_a], reduction=_MAX, depth=_DEPTH
    ) as coll:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="empty sampler pool"):
            coll.load_batch(
                999_999,
                2,
                rng=rng,
                context_len=16,
                num_variants_per_section=2,
                max_depth=2,
            )


# ---------------------------------------------------------------------------
# 7. close()
# ---------------------------------------------------------------------------


def test_close_then_load_batch_raises(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")

    coll = IndexedMemmapCollection.discover(
        [dir_a], reduction=_MAX, depth=_DEPTH
    )
    coll.close()
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="closed"):
        coll.load_batch(
            _POPULATED_LENGTH,
            2,
            rng=rng,
            context_len=16,
            num_variants_per_section=2,
            max_depth=2,
        )
    # Idempotent.
    coll.close()


def test_context_manager_closes(tmp_path: Path) -> None:
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")

    with IndexedMemmapCollection.discover(
        [dir_a], reduction=_MAX, depth=_DEPTH
    ) as coll:
        rng = np.random.default_rng(0)
        coll.load_batch(
            _POPULATED_LENGTH,
            2,
            rng=rng,
            context_len=16,
            num_variants_per_section=2,
            max_depth=2,
        )
    # After the with-block the collection is closed.
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="closed"):
        coll.sample_section_pointers(_POPULATED_LENGTH, 1, rng)
