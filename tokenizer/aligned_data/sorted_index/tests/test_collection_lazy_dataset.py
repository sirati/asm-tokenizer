"""Laziness proof for the collection's per-binary :class:`BinaryDataset`.

The collection must NOT parse any binary's section arms at
``discover()`` time -- that eager parse over the whole corpus (every
``BinaryDataset.__init__`` -> ``load_section_arm`` -> ``parse_section_bin``)
is what wedged training at init. These tests pin the deferral:

* ``BinaryDataset.__init__`` is called 0 times by ``discover(...)`` and
  only once a ``load_batch`` (via ``_session_for``) touches a binary --
  the mutation that reverts to eager discovery makes the first assertion
  fire and the test fail.
* ``_dataset_for(name)`` returns the SAME instance across calls (one
  parse per binary), and an unsampled member is never built.

Reuses the same synthetic decodable-corpus fixture as
``test_collection`` so the parse path is the real one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.sorted_index import (
    IndexedMemmapCollection,
    LengthReduction,
    ReductionKind,
)

from ._length_helpers import ensure_sidecar
from .fixtures import build_combined_fixture, make_test_vocab_manager


_MAX = LengthReduction(ReductionKind.MAX)
_DEPTH = 3
_POPULATED_LENGTH = 9


def _build_decodable_binary(
    memmap_dir: Path, binary_name: str, scratch: Path
) -> None:
    """Place a combined-corpus binary + its REAL sorted index (mirrors
    ``test_collection._build_decodable_binary``)."""
    from tokenizer.aligned_data.sorted_index import write_sorted_index_files

    combined_base = build_combined_fixture(scratch)
    for entry in combined_base.iterdir():
        if not entry.is_file() or not entry.name.startswith("sortbin"):
            continue
        new_name = binary_name + entry.name[len("sortbin"):]
        (memmap_dir / new_name).write_bytes(entry.read_bytes())
    ensure_sidecar(memmap_dir, binary_name)
    write_sorted_index_files(
        memmap_dir, binary_name, reductions=[_MAX], depths=[_DEPTH],
    )


def _count_init(monkeypatch) -> dict:
    """Monkeypatch a per-binary call-counter onto ``BinaryDataset.__init__``."""
    counts: dict = {}
    real_init = BinaryDataset.__init__

    def counting_init(self, base_path, binary_name, *args, **kwargs):
        counts[binary_name] = counts.get(binary_name, 0) + 1
        real_init(self, base_path, binary_name, *args, **kwargs)

    monkeypatch.setattr(BinaryDataset, "__init__", counting_init)
    return counts


def test_discover_does_not_parse_sections(tmp_path: Path, monkeypatch) -> None:
    """discover() builds 0 datasets; the first load_batch builds exactly one.

    MUTATION: reverting ``discover_members`` to eagerly construct a
    ``BinaryDataset`` per member makes the post-discover assertion
    (``counts == {}``) fire -> this test fails. That is the proof the
    init-time wedge is gone.
    """
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")

    counts = _count_init(monkeypatch)

    coll = IndexedMemmapCollection.discover(
        [dir_a],
        reduction=_MAX,
        depth=_DEPTH,
        vocab_manager=make_test_vocab_manager(),
    )
    # Lazy: nothing parsed at discovery.
    assert counts == {}, (
        "discover() must not construct any BinaryDataset (no section parse "
        f"up front); got {dict(counts)}"
    )

    with coll:
        rng = np.random.default_rng(0)
        coll.load_batch(
            _POPULATED_LENGTH,
            2,
            rng=rng,
            context_len=16,
            num_variants_per_section=2,
            max_depth=2,
        )
    # First sample of alpha built its dataset exactly once.
    assert counts == {"alpha": 1}, (
        "the sampled binary's BinaryDataset must be built exactly once on "
        f"first sample; got {dict(counts)}"
    )


def test_dataset_for_memoizes(tmp_path: Path, monkeypatch) -> None:
    """``_dataset_for`` returns the SAME instance twice (one parse/binary)."""
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")

    counts = _count_init(monkeypatch)

    with IndexedMemmapCollection.discover(
        [dir_a],
        reduction=_MAX,
        depth=_DEPTH,
        vocab_manager=make_test_vocab_manager(),
    ) as coll:
        first = coll._dataset_for("alpha")
        second = coll._dataset_for("alpha")
        assert first is second, "dataset must be cached, not rebuilt per call"
        assert counts == {"alpha": 1}, (
            f"BinaryDataset must be built exactly once; got {dict(counts)}"
        )


def test_unsampled_member_never_built(tmp_path: Path, monkeypatch) -> None:
    """A discovered-but-unsampled member never constructs its dataset."""
    dir_a = tmp_path / "pkgA"
    dir_a.mkdir()
    _build_decodable_binary(dir_a, "alpha", tmp_path / "scratch_a")
    _build_decodable_binary(dir_a, "beta", tmp_path / "scratch_b")

    counts = _count_init(monkeypatch)

    with IndexedMemmapCollection.discover(
        [dir_a],
        reduction=_MAX,
        depth=_DEPTH,
        vocab_manager=make_test_vocab_manager(),
    ) as coll:
        assert {m.qualified_name for m in coll.members} == {"alpha", "beta"}
        # Touch only alpha.
        coll._dataset_for("alpha")
        assert counts == {"alpha": 1}, (
            "only the touched binary may be built; an unsampled member must "
            f"stay unparsed; got {dict(counts)}"
        )
