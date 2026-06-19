"""Row<->variant-body identity test for the validation decode path.

The sibling ``test_validation_decode.py`` proves the WHOLE
``open_validation_batches`` wiring (shapes + coherence + that explicit
indices reach ``resolve_section_pointers``'s ``sampled_variant_indices``).
It does NOT, however, pin ROW<->BODY identity end-to-end: that decoded
output row ``k`` actually carries the body of the variant the selection
named at explicit-position ``k``. A body-loader reorder regression -- one
that loaded the right variant SET but scattered the bodies into the wrong
rows -- would slip through every gate there.

This file closes that gap with a deterministic, RNG-independent identity
proof at the SAME decode seam ``open_validation_batches`` drives
(``decode_pointer_batch`` over one ``MultiBinarySectionPointer`` carrying
an ``ExplicitIndicesSelection``, ``variant_padding=RAGGED``). The proof is
a permutation invariant: decoding matched section idx 2 (4 variants) under
an explicit order and under its REVERSE must permute the OUTPUT ROWS
correspondingly. row ``k`` of ``decode(reverse)`` is byte-identical to row
``(n-1-k)`` of ``decode(forward)`` -- across the token grid AND each row's
identity / number sidecar sub-slices. Because the body is the only thing
the explicit index chooses, this proves each row's body tracks its chosen
variant index rather than a fixed/positional decode.

A vacuity guard asserts the 4 variant rows are not all token-identical
(otherwise the permutation is trivially satisfied and proves nothing).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.batch_decode._variant_selection import (
    ExplicitIndicesSelection,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.sorted_index._sampler import decode_pointer_batch
from tokenizer.aligned_data.sorted_index._types import (
    MultiBinarySectionPointer,
)
from tokenizer.aligned_data.sorted_index.tests.fixtures import (
    build_combined_fixture_with_variants,
    make_test_vocab_manager,
)


_BINARY_NAME = "sortbin"
# Fixture matched variant counts are [0, 1, 4, 1, 2]; idx 2 is the only
# section with enough variants for a non-trivial 4-way permutation.
_SECTION_IDX = 2
_N_VARIANTS = 4


def _decode_order(
    session: BinarySession, order: tuple[int, ...]
) -> object:
    """Decode matched section ``_SECTION_IDX`` under a pinned variant order.

    Mirrors ``open_validation_batches``' decode seam exactly: one
    ``MultiBinarySectionPointer`` whose ``SectionPointerSpec`` carries an
    ``ExplicitIndicesSelection``, decoded through the unchanged
    ``decode_pointer_batch`` with ``RAGGED`` padding so the output has
    exactly ``len(order)`` rows in explicit-index order. A FRESH rng is
    passed so neither order's decode can drift via shared RNG state -- the
    explicit selection never consumes rng, so the two decodes differ ONLY
    in the pinned variant order.
    """
    pointer = MultiBinarySectionPointer(
        binary_name=_BINARY_NAME,
        section_pointer=SectionPointerSpec(
            arm=SectionKind.MATCHED,
            idx=_SECTION_IDX,
            variant_selection=ExplicitIndicesSelection(order),
        ),
    )
    return decode_pointer_batch(
        {_BINARY_NAME: session},
        [pointer],
        context_len=32,
        num_variants_per_section=len(order),
        max_depth=2,
        rng=np.random.default_rng(0),
        variant_padding=VariantPadding.RAGGED,
    ).inner


def _row_signature(inner: object, k: int) -> tuple[bytes, bytes, bytes]:
    """Full byte signature of decoded row ``k``: tokens + id + number slices."""
    tokens = np.asarray(inner.tokens)[k].tobytes()
    i0, i1 = (int(x) for x in inner.identity_row_offsets[k : k + 2])
    identities = np.asarray(inner.identities)[i0:i1].tobytes()
    n0, n1 = (int(x) for x in inner.number_row_offsets[k : k + 2])
    numbers = (
        np.asarray(inner.numbers_significant)[n0:n1].tobytes()
        + np.asarray(inner.numbers_sign_exponent)[n0:n1].tobytes()
    )
    return (tokens, identities, numbers)


def test_explicit_order_permutes_decoded_rows(tmp_path: Path) -> None:
    vocab_manager = make_test_vocab_manager()
    base = build_combined_fixture_with_variants(tmp_path, vocab_manager)

    @contextmanager
    def session_factory() -> Iterator[BinarySession]:
        dataset = BinaryDataset(base, _BINARY_NAME, vocab_manager=vocab_manager)
        with dataset.open_session() as session:
            yield session

    forward = (0, 1, 2, 3)
    reverse = (3, 2, 1, 0)

    with session_factory() as session:
        fwd = _decode_order(session, forward)
        rev = _decode_order(session, reverse)

        assert np.asarray(fwd.tokens).shape == (_N_VARIANTS, 32)
        assert np.asarray(rev.tokens).shape == (_N_VARIANTS, 32)

        fwd_rows = [_row_signature(fwd, k) for k in range(_N_VARIANTS)]
        rev_rows = [_row_signature(rev, k) for k in range(_N_VARIANTS)]

        # Vacuity guard: the 4 variant bodies must be distinguishable at the
        # decoded-row level, else the permutation invariant is trivially met.
        assert (
            len(set(fwd_rows)) > 1
        ), "fixture's 4 variants decoded to identical rows; permutation vacuous"

        # The identity contract: reversing the explicit variant order
        # reverses the output rows. Row k of reverse == row (n-1-k) of
        # forward, byte-for-byte across tokens + id + number sub-slices --
        # proving each row's BODY tracks its chosen variant index.
        for k in range(_N_VARIANTS):
            assert rev_rows[k] == fwd_rows[_N_VARIANTS - 1 - k], (
                f"row {k} of reverse-order decode did not match row "
                f"{_N_VARIANTS - 1 - k} of forward-order decode; a body "
                "loader reorder would leak through here"
            )
