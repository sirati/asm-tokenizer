"""Byte-equivalence test for the deferred collector dispatch shape.

Single concern: assert that :func:`batch_decode` produces byte-
identical :class:`BatchDecodeResult` output between the synchronous
``collector=None`` path and the deferred ``collector=<shared>`` path
(when the caller flushes the shared collector + calls
:meth:`PendingBatchDecode.finalise`).

The deferred path is the hook the orchestrator
(``open_length_bucketed_batch`` / ``compute_reduced_lengths``) uses to
share a collector across many ``batch_decode`` / ``walk_sections``
calls. The contract: the only thing the deferred path defers is the
``run_lengths`` dispatch + the Stage-1 finalisation; every Stage-2/3/4
output array must match the synchronous path bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from tokenizer.aligned_data.loader.batch_decode import (
    PendingBatchDecode,
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.loader.tests._session_fixture import (
    build_synthetic_binary,
)


def test_batch_decode_deferred_collector_byte_equiv(tmp_path) -> None:
    """``batch_decode(collector=shared)`` + ``.finalise(runlen_results)``
    must produce the same :class:`BatchDecodeResult` arrays as the
    synchronous ``collector=None`` path.

    Runs the synchronous path with one seeded RNG; runs the deferred
    path with an EQUIVALENT seeded RNG (same seed -> same variant
    sampling) and asserts every output array compares byte-equal.
    """
    fb = build_synthetic_binary(tmp_path)

    section_pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
    ]
    kwargs = dict(
        section_pointers=section_pointers,
        num_variants_per_section=3,
        context_len=64,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
    )

    # Synchronous path.
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as session:
        sync_result = batch_decode(
            session,
            **kwargs,
            rng=np.random.default_rng(seed=42),
        )

    # Deferred path: caller owns the collector, calls batch_decode with
    # collector=shared, then flushes + finalises.
    collector = BucketedRunLengthCollector()
    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as session:
        pending = batch_decode(
            session,
            **kwargs,
            rng=np.random.default_rng(seed=42),
            collector=collector,
        )
        assert isinstance(pending, PendingBatchDecode)
        runlen_results = collector.flush()
        deferred_result = pending.finalise(runlen_results)

    # Cross-array byte-equivalence. Every output array on the sync
    # result must compare bit-identical to the deferred result.
    np.testing.assert_array_equal(sync_result.tokens, deferred_result.tokens)
    np.testing.assert_array_equal(
        sync_result.identities, deferred_result.identities,
    )
    np.testing.assert_array_equal(
        sync_result.identity_row_offsets,
        deferred_result.identity_row_offsets,
    )
    np.testing.assert_array_equal(
        sync_result.numbers_significant, deferred_result.numbers_significant,
    )
    np.testing.assert_array_equal(
        sync_result.numbers_sign_exponent,
        deferred_result.numbers_sign_exponent,
    )
    np.testing.assert_array_equal(
        sync_result.number_row_offsets, deferred_result.number_row_offsets,
    )
    np.testing.assert_array_equal(
        sync_result.batch_idx_to_section_variant,
        deferred_result.batch_idx_to_section_variant,
    )
    # Sidecars / intermediate are None under the kwargs we used.
    assert sync_result.fid_sidecar is None
    assert deferred_result.fid_sidecar is None
    assert sync_result.fid_row_offsets is None
    assert deferred_result.fid_row_offsets is None
    assert sync_result.intermediate is None
    assert deferred_result.intermediate is None


def test_batch_decode_deferred_collector_handles_multi_call(tmp_path) -> None:
    """ONE shared collector across TWO sequential ``batch_decode``
    calls must produce the SAME outputs as two separate synchronous
    ``batch_decode`` calls.

    This is the smallest pinning of the orchestrator pattern: each
    deferred decode stages onto the shared collector; one flush at the
    end materialises every pending result.
    """
    fb = build_synthetic_binary(tmp_path)

    section_pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
    ]
    kwargs = dict(
        section_pointers=section_pointers,
        num_variants_per_section=2,
        context_len=32,
        max_depth=2,
        variant_padding=VariantPadding.PAD_NULL,
    )

    # Two synchronous calls with two distinct seeds.
    sync_results = []
    for seed in (11, 22):
        with BinarySession(
            fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
        ) as session:
            sync_results.append(batch_decode(
                session,
                **kwargs,
                rng=np.random.default_rng(seed=seed),
            ))

    # Two deferred calls onto ONE shared collector.
    collector = BucketedRunLengthCollector()
    pendings = []
    sessions_cm = []  # keep sessions alive across the whole staging phase
    try:
        for seed in (11, 22):
            session_cm = BinarySession(
                fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
            )
            session = session_cm.__enter__()
            sessions_cm.append(session_cm)
            pendings.append(batch_decode(
                session,
                **kwargs,
                rng=np.random.default_rng(seed=seed),
                collector=collector,
            ))
        runlen_results = collector.flush()
        deferred_results = [p.finalise(runlen_results) for p in pendings]
    finally:
        for session_cm in reversed(sessions_cm):
            session_cm.__exit__(None, None, None)

    # Each pair must be byte-equal.
    for sync, deferred in zip(sync_results, deferred_results):
        np.testing.assert_array_equal(sync.tokens, deferred.tokens)
        np.testing.assert_array_equal(sync.identities, deferred.identities)
        np.testing.assert_array_equal(
            sync.identity_row_offsets, deferred.identity_row_offsets,
        )
        np.testing.assert_array_equal(
            sync.numbers_significant, deferred.numbers_significant,
        )
        np.testing.assert_array_equal(
            sync.numbers_sign_exponent, deferred.numbers_sign_exponent,
        )
