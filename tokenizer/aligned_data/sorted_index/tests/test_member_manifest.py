"""The reader's additive metadata plan owns eligibility and consumed files."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.sorted_index._collection import _discovery as owner
from tokenizer.aligned_data.sorted_index._collection._member import MissingIndexPolicy
from tokenizer.aligned_data.sorted_index._types import IndexSpec, LengthReduction, ReductionKind
from tokenizer.aligned_data.realized_lengths._geometry_reader import RealizedGeometryReader
from tokenizer.aligned_data.realized_lengths._geometry_format import GEOMETRY_ARMS, write_geometry_pair


def spec(depth=0):
    return IndexSpec(LengthReduction(ReductionKind.PERCENTILE, percentile=75), depth)


def member(root, name='one', depth=0):
    root.mkdir(parents=True, exist_ok=True)
    (root / f'{name}_index.bin').touch()
    (root / f'{name}_sorted_p75_d{depth:03d}.idx').write_bytes(b'not an index: planning must not read it')


def test_manifest_is_metadata_only_and_contains_both_geometry_pairs(tmp_path, monkeypatch):
    member(tmp_path)
    def forbidden_reader(path, *, reduction, depth):
        raise AssertionError('metadata planning constructed an index reader')
    def forbidden_read(path):
        raise AssertionError('metadata planning read corpus bytes')
    monkeypatch.setattr(owner, 'SortedIndexReader', forbidden_reader)
    monkeypatch.setattr(Path, 'read_bytes', forbidden_read)
    plan = owner.plan_members([tmp_path], specs=(spec(),))
    assert [item.binary_name for item in plan.members] == ['one']
    assert {path.name for path in plan.files} == {
        'one_sections.bin', 'one_data.bin', 'one_index.bin', 'one_unmatched_data.bin',
        'one_unmatched_index.bin', 'one_variants.bin', 'one_variants.csv',
        'one_function_names.txt', 'one_extern_providers.txt', 'one_realized.bin',
        'one_realized_index.bin', 'one_unmatched_realized.bin',
        'one_unmatched_realized_index.bin', 'one_sorted_p75_d000.idx'}
    assert plan.indices[0].path == tmp_path / 'one_sorted_p75_d000.idx'


def test_discovery_consumes_the_one_membership_plan(tmp_path, monkeypatch):
    member(tmp_path, 'a')
    member(tmp_path, 'b')
    original = owner.plan_members([tmp_path], specs=(spec(),))
    selected = replace(original, members=original.members[:1], indices=original.indices[:1])
    planned = []
    def supplied(directories, *, specs, on_missing):
        planned.append(tuple(directories))
        return selected
    opened = []
    def reader(path, *, reduction, depth):
        opened.append(path)
        return path
    monkeypatch.setattr(owner, 'plan_members', supplied)
    monkeypatch.setattr(owner, 'SortedIndexReader', reader)
    members, readers = owner.discover_members([tmp_path], specs=(spec(),), on_missing=MissingIndexPolicy.RAISE)
    assert planned == [(tmp_path,)]
    assert members == list(selected.members)
    assert opened == [tmp_path / 'a_sorted_p75_d000.idx']
    assert readers == {spec(): {'a': opened[0]}}


def test_plan_and_reader_share_the_canonical_depth_eligibility(tmp_path):
    member(tmp_path, depth=1000)
    for operation in (owner.plan_members, owner.discover_members):
        with pytest.raises(ValueError, match='missing sorted-index.*p75_d1000'):
            operation([tmp_path], specs=(spec(1000),), on_missing=MissingIndexPolicy.RAISE)


def test_manifest_excludes_deep_backups_and_unrequested_indices(tmp_path):
    member(tmp_path)
    member(tmp_path / 'backup' / 'deep', 'unused')
    (tmp_path / 'one_sorted_p75_d003.idx').touch()
    (tmp_path / 'one_sections.csv').touch()
    plan = owner.plan_members([tmp_path], specs=(spec(),))
    assert [item.binary_name for item in plan.members] == ['one']
    assert all('unused' not in path.name and 'd003' not in path.name and not path.name.endswith('sections.csv')
               for path in plan.files)


@pytest.mark.parametrize('arm', GEOMETRY_ARMS, ids=lambda arm: arm.name)
def test_geometry_open_uses_its_published_input_paths(tmp_path, monkeypatch, arm):
    paths = RealizedGeometryReader.input_paths(tmp_path, 'one', arm)
    write_geometry_pair(paths.geometry, paths.index, body_lengths=np.array([7], dtype=np.uint32),
        id_counts=np.array([2], dtype=np.uint32), value_counts=np.array([1], dtype=np.uint32),
        csr_offsets=np.array([0, 1], dtype=np.uint32))
    observed = []
    original = RealizedGeometryReader.input_paths
    def traced(cls, base, name, selected):
        result = original(base, name, selected)
        observed.append(result)
        return result
    monkeypatch.setattr(RealizedGeometryReader, 'input_paths', classmethod(traced))
    reader = RealizedGeometryReader.open(tmp_path, 'one', arm)
    try:
        assert reader.body_lengths.tolist() == [7]
        assert observed == [paths]
    finally:
        reader.close()
