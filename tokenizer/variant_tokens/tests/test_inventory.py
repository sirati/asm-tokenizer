"""Tests for ``tokenizer.variant_tokens.inventory.VariantInventory``.

Covers dedup, deterministic iteration, and the ``":" not in key``
hard-assert.
"""

from __future__ import annotations

import pytest

from tokenizer.variant_tokens.inventory import VariantInventory

from ._fakes import FakeVersionInfo


def test_dedup_across_identical_versions():
    inv = VariantInventory()
    vi = FakeVersionInfo(extra_metadata={"hardening": "full"})
    inv.add(vi)
    inv.add(vi)
    # 4 positional + 1 metadata = 5 distinct strings, regardless of
    # how many times the same version is added.
    assert len(inv) == 5


def test_dedup_partial_overlap():
    """Two versions sharing arch/compiler but differing in opt should
    yield distinct opt tokens but the same arch/compiler tokens."""
    a = FakeVersionInfo(opt="-O2", extra_metadata={})
    b = FakeVersionInfo(opt="-Os", extra_metadata={})
    inv = VariantInventory()
    inv.add(a)
    inv.add(b)
    # 3 shared (arch, comp, cver) + 2 distinct opt = 5.
    assert len(inv) == 5


def test_iter_tokens_alphabetical():
    inv = VariantInventory()
    inv.add(FakeVersionInfo(extra_metadata={"hardening": "full"}))
    out = list(inv.iter_tokens())
    assert out == sorted(out)


def test_metadata_key_with_colon_rejected():
    """The ``inventory.add()`` invariant is the only guard against
    decoder corruption — a metadata key containing ``:`` would split
    into a different (key, value) pair than the encoder built."""
    inv = VariantInventory()
    bad = FakeVersionInfo(extra_metadata={"comp:fortify": "yes"})
    with pytest.raises(AssertionError, match=r"contains ':'"):
        inv.add(bad)


def test_update_iterable_helper():
    inv = VariantInventory()
    versions = [
        FakeVersionInfo(opt="-O0"),
        FakeVersionInfo(opt="-O1"),
        FakeVersionInfo(opt="-O2"),
    ]
    inv.update(versions)
    # 3 shared positional + 3 distinct opt = 6.
    assert len(inv) == 6


def test_contains_membership():
    inv = VariantInventory()
    inv.add(FakeVersionInfo(opt="-O2"))
    assert "opt:O2" in inv
    assert "opt:Os" not in inv


def test_empty_extra_metadata_does_not_assert():
    inv = VariantInventory()
    inv.add(FakeVersionInfo(extra_metadata={}))
    assert len(inv) == 4  # only positional axes


def test_none_extra_metadata_fails_loudly():
    """``BinaryVersionInfo`` guarantees ``extra_metadata`` is a dict
    via ``field(default_factory=dict)``. A ``None`` here means the
    constructor was bypassed (raw kwarg construction with a bad
    value); inventory must surface that as a clear ``TypeError``
    rather than silently treating ``None`` as empty — silent coercion
    would hide an upstream bug."""
    inv = VariantInventory()

    class BadMeta:
        arch = "x86_64"
        compiler = "gcc"
        compilerversion = "13.2.0"
        opt = "-O2"
        extra_metadata = None

    with pytest.raises(TypeError):
        inv.add(BadMeta())
