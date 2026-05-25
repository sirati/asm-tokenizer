"""Unit tests for :class:`tokenizer.variant_info.VariantIdentity` and
the :meth:`VariantInfo.from_function_data_metadata` factory.

Covers:

* :class:`VariantIdentity` equality / hash semantics (frozen tuple of
  ``(arch, compiler, compiler_version, opt, pkg, variant_id)``),
* :meth:`VariantInfo.identity` returns the same tuple shape,
* round-trip via :meth:`VariantInfo.from_function_data_metadata` on
  the matched-arm shape (axis values + arbitrary metakeys),
* the unmatched-arm sentinel (``"unknown"`` axis values) flows
  through unchanged so per-arm bucketing stays consistent,
* structural-key strip list — every key the loader injects
  (``variant_ref`` / ``call_targets`` / ``data_offset`` /
  ``category_counts`` / ``filename`` / ``variant_tokens`` / etc.) is
  stripped from the extra-metadata residue,
* alphabetical key order (stable iteration for grouping bucket
  consistency),
* list-valued metakeys comma-join to single strings.
"""

from __future__ import annotations

import pytest

from tokenizer.variant_info import VariantIdentity, VariantInfo


# ----------------------------------------------------------------- identity


def test_variant_identity_equality_is_field_wise():
    """Two :class:`VariantIdentity` with identical fields compare equal
    (frozen dataclass default ``__eq__``)."""
    a = VariantIdentity(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="hello", variant_id=0,
    )
    b = VariantIdentity(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="hello", variant_id=0,
    )
    assert a == b
    assert hash(a) == hash(b)


def test_variant_identity_disambiguates_via_variant_id():
    """Two variants of the same function sharing the canonical-4 axes
    but differing in :attr:`variant_id` (the dedup-disambiguator at
    ``tokenizer.aligned_data.io.write_matched_section_csv``) MUST NOT
    compare equal. Cluster #9 — using a hand-rolled canonical-4 tuple
    would collapse them silently."""
    a = VariantIdentity(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="hello", variant_id=0,
    )
    b = VariantIdentity(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="hello", variant_id=1,
    )
    assert a != b
    assert hash(a) != hash(b)


def test_variant_identity_distinguishes_pkg():
    """Two same-arch / same-toolchain variants of different packages
    must hash to distinct buckets."""
    a = VariantIdentity(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="hello", variant_id=0,
    )
    b = VariantIdentity(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="busybox", variant_id=0,
    )
    assert a != b


def test_variant_info_identity_property():
    """:meth:`VariantInfo.identity` projects onto the same six fields."""
    info = VariantInfo(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="hello", variant_id=7,
        extra_metadata={"hardening": "full"},
    )
    expected = VariantIdentity(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="hello", variant_id=7,
    )
    assert info.identity == expected


def test_variant_info_identity_matches_hash():
    """``VariantInfo.__hash__`` and ``hash(info.identity)`` agree —
    same six-field shape, same canonical-identity contract."""
    info = VariantInfo(
        arch="arm64", compiler="clang", compiler_version="15.0.0",
        opt="O3", pkg="curl", variant_id=3,
    )
    # The two hashes hash distinct tuples (VariantInfo's hash hashes
    # the bare tuple; VariantIdentity hashes a frozen dataclass) but
    # both identities are equal under the canonical-identity definition.
    info_b = VariantInfo(
        arch="arm64", compiler="clang", compiler_version="15.0.0",
        opt="O3", pkg="curl", variant_id=3,
        extra_metadata={"sanitizer": "addr"},  # differs in opaque metadata
    )
    assert info == info_b
    assert info.identity == info_b.identity


# ------------------------------------------------- from_function_data_metadata


def test_factory_extracts_canonical_axes_from_matched_metadata():
    """The matched-arm metadata shape: ``arch`` / ``compiler`` /
    ``compilerversion`` / ``opt`` come from the variant-resolver via
    :func:`tokenizer.variant_tokens.encoder.decode_record`."""
    metadata = {
        "arch": "x64",
        "compiler": "gcc",
        "compilerversion": "13.2.0",
        "opt": "O2",
        # Loader-managed structural keys:
        "variant_ref": "abc",
        "call_targets": [],
        "data_offset": 0,
        "category_counts": [],
        "filename": "x64-gcc-13.2.0-O2_hello",
    }
    identity, extra = VariantInfo.from_function_data_metadata(
        metadata, pkg="hello", variant_id=5,
    )
    assert identity == VariantIdentity(
        arch="x64", compiler="gcc", compiler_version="13.2.0",
        opt="O2", pkg="hello", variant_id=5,
    )
    # All structural keys stripped; no user-facing metakeys here.
    assert dict(extra) == {}


def test_factory_strips_every_structural_key():
    """The loader injects a known set of structural keys; none of them
    may leak into ``extra_metadata``. Drift here would silently
    expand the inspector's EXTRA_META grouping surface with internal
    plumbing."""
    metadata = {
        "arch": "x64", "compiler": "gcc",
        "compilerversion": "13.2.0", "opt": "O2",
        # Per ``tokenizer.aligned_data.loader._session_parsers``:
        "variant_ref": "abc",
        "variant_refs": ["a", "b"],
        "variants": [],
        "called": [],
        "call_targets": [],
        "data_offset": 16,
        "category_counts": [1, 2, 3],
        "filename": "x64-gcc-13.2.0-O2_hello",
        "variant_tokens": None,
        # ``pkg`` and ``variant_id`` are NEVER user-facing extras
        # either — they're canonical-identity fields:
        "pkg": "hello",
        "variant_id": 5,
        # One genuine user-facing metakey:
        "hardening": ["full"],
    }
    _, extra = VariantInfo.from_function_data_metadata(metadata)
    assert dict(extra) == {"hardening": "full"}


def test_factory_joins_list_values_with_commas():
    """Tail metakeys come back as sorted lists from
    :func:`decode_record`; comma-join projects them onto a single
    string-per-key so the inspector's EXTRA_META grouping has one
    bucket per unique value-set."""
    metadata = {
        "arch": "arm64", "compiler": "clang",
        "compilerversion": "15.0.0", "opt": "O3",
        "flag_set": ["-fpic", "-fno-plt"],
        "hardening": ["full"],
    }
    _, extra = VariantInfo.from_function_data_metadata(metadata)
    assert dict(extra) == {
        "flag_set": "-fpic,-fno-plt",
        "hardening": "full",
    }


def test_factory_emits_keys_in_alphabetical_order():
    """Stable iteration is required: the inspector's BITWIDTH /
    POSITIONAL / EXTRA_META axis dispatcher iterates this Mapping when
    building the per-axis group set, and a non-deterministic order
    would surface as flickering bucket order across runs."""
    metadata = {
        "arch": "x64", "compiler": "gcc",
        "compilerversion": "13.2.0", "opt": "O2",
        "zeta": "1", "alpha": "2", "mu": "3",
    }
    _, extra = VariantInfo.from_function_data_metadata(metadata)
    assert list(extra.keys()) == ["alpha", "mu", "zeta"]


def test_factory_handles_unmatched_arm_unknown_sentinel():
    """Unmatched-arm metadata carries ``"unknown"`` for every axis
    (per ``_session_parsers.build_unmatched_function_data``). The
    factory passes it through unchanged so per-arm bucketing remains
    stable."""
    metadata = {
        "arch": "unknown", "compiler": "unknown",
        "compilerversion": "unknown", "opt": "unknown",
        "variant_refs": ["a"], "variants": [], "called": [],
        "call_targets": [], "data_offset": 0,
        "category_counts": [],
    }
    identity, extra = VariantInfo.from_function_data_metadata(metadata)
    assert identity.arch == "unknown"
    assert identity.compiler == "unknown"
    assert identity.compiler_version == "unknown"
    assert identity.opt == "unknown"
    assert dict(extra) == {}


def test_factory_missing_axis_keys_collapse_to_unknown():
    """A metadata dict with no ``arch`` key (defensive — should not
    happen in production paths but the loader doesn't enforce key
    presence) gives the ``"unknown"`` sentinel rather than ``None`` so
    the :class:`VariantIdentity` field stays string-typed."""
    metadata = {}
    identity, extra = VariantInfo.from_function_data_metadata(metadata)
    assert identity.arch == "unknown"
    assert identity.compiler == "unknown"
    assert identity.compiler_version == "unknown"
    assert identity.opt == "unknown"
    assert dict(extra) == {}


def test_factory_returns_mappingproxytype():
    """The returned ``extra_metadata`` MUST be immutable (per plan
    decision 21 — frozen-dataclass field cannot be mutated by callers).
    :class:`types.MappingProxyType` enforces this at the Python level."""
    metadata = {"arch": "x64", "compiler": "gcc", "compilerversion": "13", "opt": "O2", "key": "val"}
    _, extra = VariantInfo.from_function_data_metadata(metadata)
    with pytest.raises(TypeError):
        extra["new_key"] = "x"  # type: ignore[index]


def test_factory_default_pkg_and_variant_id():
    """``pkg`` and ``variant_id`` are NOT recoverable from the runtime
    metadata dict (the variant record encodes axes + sorted metakey
    tail). Defaults are ``""`` and ``0`` — the BatchDecode backend's
    pre-Wave-5 contract."""
    metadata = {"arch": "x64", "compiler": "gcc", "compilerversion": "13", "opt": "O2"}
    identity, _ = VariantInfo.from_function_data_metadata(metadata)
    assert identity.pkg == ""
    assert identity.variant_id == 0
