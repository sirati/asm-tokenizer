"""Shared on-disk fixture builders for the loader integration tests.

Single concern: lay down a synthetic per-binary memmap output tree on
disk that mirrors the post-restructuring v1 wire format the production
pipeline emits, with function-name lengths that produce section CSV
byte offsets covering every mod-4 residue (``0..3``).

Submodules:

* :mod:`.names`    -- ``make_variable_length_names`` +
                       ``assert_mod4_residues_covered``
* :mod:`.specs`    -- ``MatchedFunctionSpec`` / ``UnmatchedFunctionSpec`` /
                       ``VariantSpec`` + ``matched_spec`` / ``unmatched_spec``
                       / ``make_simple_variant``
* :mod:`.builder`  -- ``build_corpus`` (stub registry) +
                       ``build_corpus_with_registry`` (test-supplied registry)
                       + ``CorpusPaths``

Re-exported here so tests need a single import line.
"""

from __future__ import annotations

from .builder import (
    CorpusPaths,
    build_corpus,
    build_corpus_with_registry,
)
from .names import (
    assert_starts_4_byte_aligned,
    make_variable_length_names,
)
from .specs import (
    MatchedFunctionSpec,
    UnmatchedFunctionSpec,
    VariantSpec,
    make_simple_variant,
    matched_spec,
    unmatched_spec,
)

__all__ = (
    "CorpusPaths",
    "MatchedFunctionSpec",
    "UnmatchedFunctionSpec",
    "VariantSpec",
    "assert_starts_4_byte_aligned",
    "build_corpus",
    "build_corpus_with_registry",
    "make_simple_variant",
    "make_variable_length_names",
    "matched_spec",
    "unmatched_spec",
)
