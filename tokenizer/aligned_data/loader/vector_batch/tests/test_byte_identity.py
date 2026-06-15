"""Byte-identity harness: the vectorized path vs ``batch_decode``.

THE GATE for the vectorized dataloader. Runs the NEW path
(:func:`...vector_batch._entry.vector_batch_tokens`) and the CURRENT
:func:`...batch_decode.batch_decode` with IDENTICAL settings (same RNG
seed, same section pointers, same depth, BACKFILL OFF) on a real small
binary and asserts FULL byte-identity of the token tensor + the
``batch_idx_to_section_variant`` mapping.

Both paths now do sampled-subset outline detection (T0 merged), so the
included set + emission order converge by construction; this test pins
that the geometry + scatter assemble the SAME ``tokens[B, L]`` the
staged ``batch_decode`` pipeline produces.

Highest-risk fixture (TC1 author's flag): ``build_combined_fixture``
carries a MULTI-VARIANT section (``multi_fn``, 4 variants) AND an
ASYMMETRIC call graph (``caller_fn`` -> ``callee_fn`` with disjoint vkey
sets -> a MISSING_VARIANT_INDEX splice edge), so subset-vs-full FLAG-A
inclusion differs across rows. The straddler cut is exercised by a tight
``context_len``; a multi-section batch by pointing at several sections.

REUSE: the on-disk corpus + RLG3 sidecars + unified-vocab session come
from the shared sorted-index fixtures + realized-geometry generators (the
same wiring the realized-geometry oracle test uses); the new path's
handles come from :func:`...vector_batch.session_handles.
open_vector_batch_handles`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tokenizer.aligned_data.loader.batch_decode import (
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.vector_batch._entry import (
    vector_batch_tokens,
)
from tokenizer.aligned_data.loader.vector_batch.session_handles import (
    open_vector_batch_handles,
)
from tokenizer.aligned_data.realized_lengths import (
    generate_realized_geometry,
    generate_realized_lengths,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture,
    build_many_variant_section_fixture,
    make_test_vocab_manager,
)


_BINARY_NAME = "sortbin"


def _prepare(builder, tmp_path: Path):
    """Build the corpus + RLG3 sidecars; return ``base_path``."""
    base = builder(tmp_path)
    generate_realized_lengths(base, _BINARY_NAME)
    generate_realized_geometry(base, _BINARY_NAME)
    return base


def _nonempty_matched_idxs(base: Path) -> list[int]:
    """Matched section indices with at least one variant.

    ``batch_decode`` rejects a 0-variant section at sampling time (the
    combined fixture deliberately carries one as an edge case), so the
    harness batches only the sample-able sections.
    """
    dataset = BinaryDataset(base, _BINARY_NAME, vocab_manager=None)
    out: list[int] = []
    n = 0
    with dataset.open_session() as session:
        while True:
            try:
                matched = session.load_matched(n)
            except IndexError:
                break
            if len(matched.variants) > 0:
                out.append(n)
            n += 1
    return out


def _run_both(
    base: Path,
    *,
    section_idxs,
    num_variants_per_section: int,
    context_len: int,
    max_depth: int,
    seed: int,
    variant_padding: VariantPadding = VariantPadding.PAD_NULL,
):
    """Run ``batch_decode`` + the vectorized path with the SAME draw."""
    pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=int(i))
        for i in section_idxs
    ]
    dataset = BinaryDataset(
        base, _BINARY_NAME, vocab_manager=make_test_vocab_manager()
    )
    with dataset.open_session() as session:
        # Reference: the current staged pipeline.
        ref = batch_decode(
            session,
            pointers,
            num_variants_per_section=num_variants_per_section,
            context_len=context_len,
            max_depth=max_depth,
            variant_padding=variant_padding,
            rng=np.random.default_rng(seed),
        )
        # New path: same inputs, fresh rng with the SAME seed -> same draw.
        with open_vector_batch_handles(base, _BINARY_NAME) as handles:
            new = vector_batch_tokens(
                session,
                pointers,
                handles=handles,
                num_variants_per_section=num_variants_per_section,
                context_len=context_len,
                max_depth=max_depth,
                variant_padding=variant_padding,
                rng=np.random.default_rng(seed),
            )
    return ref, new


def _assert_token_identity(ref, new) -> None:
    assert new.tokens.dtype == ref.tokens.dtype == np.uint16
    assert new.tokens.shape == ref.tokens.shape, (
        f"shape mismatch: new {new.tokens.shape} vs ref {ref.tokens.shape}"
    )
    assert np.array_equal(
        new.batch_idx_to_section_variant, ref.batch_idx_to_section_variant
    ), "batch_idx_to_section_variant differs"
    if not np.array_equal(new.tokens, ref.tokens):
        diff_rows = np.nonzero(
            (new.tokens != ref.tokens).any(axis=1)
        )[0]
        r = int(diff_rows[0])
        raise AssertionError(
            f"tokens differ in {diff_rows.size} row(s); first row {r}:\n"
            f"  ref={ref.tokens[r].tolist()}\n"
            f"  new={new.tokens[r].tolist()}"
        )


# ---------------------------------------------------------------------------
# Multi-variant + asymmetric call graph (the highest-risk case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_byte_identity_combined_fixture_full_depth(seed, tmp_path):
    """Whole combined corpus (multi-variant + asymmetric splice), a wide
    context so no straddler -- every row's full body must match."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=4096,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_byte_identity_combined_straddler_cut(seed, tmp_path):
    """A tight context_len forces a mid-function straddler cut on the
    longer rows -- the cut column must land identically."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=12,
        max_depth=3,
        seed=seed,
    )
    _assert_token_identity(ref, new)


@pytest.mark.parametrize("seed", [0, 5])
def test_byte_identity_many_variant_section(seed, tmp_path):
    """The many-variant single section -- subset sampling of <4 of the
    variants exercises FLAG-A subset-vs-full divergence per row."""
    base = _prepare(build_many_variant_section_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=2,
        context_len=512,
        max_depth=2,
        seed=seed,
    )
    _assert_token_identity(ref, new)


def test_byte_identity_depth_zero_roots_only(tmp_path):
    """depth 0: every row is root-only (no splice) -- the prefix + root
    body + self-token must match with no callee assembly."""
    base = _prepare(build_combined_fixture, tmp_path)
    idxs = _nonempty_matched_idxs(base)
    ref, new = _run_both(
        base,
        section_idxs=idxs,
        num_variants_per_section=4,
        context_len=256,
        max_depth=0,
        seed=0,
    )
    _assert_token_identity(ref, new)
