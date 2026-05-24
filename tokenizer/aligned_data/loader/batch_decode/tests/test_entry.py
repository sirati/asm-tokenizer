"""Integration tests for :func:`batch_decode` -- the end-to-end pipeline
entry point.

Single concern of this file: pin the public-API surface of
:func:`batch_decode` and the wiring contract between the four stages
(:func:`walk_sections` -> :func:`predict_lengths` ->
:func:`build_bulk_bytes` -> :func:`assemble_batch`).

The four stages themselves are tested by their own modules' tests. This
file focuses on:

1. Default-argument plumbing (RNG defaulting,
   ``variant_padding=VariantPadding.PAD_NULL`` default, optional sidecar
   defaults).
2. Linear composition order -- each stage receives the previous
   stage's output verbatim, plus the relevant kwargs.
3. End-to-end smoke against a synthetic :class:`BinarySession` built
   from the shared :mod:`_session_fixture` corpus -- exercises the
   real stage-1 -> stage-4 wiring against a tiny on-disk binary.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import tokenizer.aligned_data.loader.batch_decode._entry as _entry_module
from tokenizer.aligned_data.loader.batch_decode import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
    batch_decode,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.loader.tests._session_fixture import (
    build_synthetic_binary,
)


# ---------------------------------------------------------------------------
# Wiring tests via monkeypatch
# ---------------------------------------------------------------------------


def test_batch_decode_default_rng_initialised(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing ``rng=None`` makes :func:`batch_decode` allocate a fresh
    :func:`numpy.random.default_rng`. The RNG is threaded through to
    stage 1 (the only stage that uses it).
    """

    captured: dict[str, Any] = {}

    def fake_walk_sections(session, section_pointers, **kwargs):
        captured["rng"] = kwargs["rng"]
        return "stage1_marker"

    def fake_predict_lengths(stage1, **kwargs):
        captured["stage1_seen"] = stage1
        return "stage2_marker"

    def fake_build_bulk_bytes(stage2):
        captured["stage2_seen"] = stage2
        return "stage3_marker"

    def fake_assemble_batch(stage3, **kwargs):
        captured["stage3_seen"] = stage3
        captured["assemble_kwargs"] = kwargs
        return "result_marker"

    monkeypatch.setattr(_entry_module, "walk_sections", fake_walk_sections)
    monkeypatch.setattr(_entry_module, "predict_lengths", fake_predict_lengths)
    monkeypatch.setattr(_entry_module, "build_bulk_bytes", fake_build_bulk_bytes)
    monkeypatch.setattr(_entry_module, "assemble_batch", fake_assemble_batch)

    result = batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        context_len=4,
        max_depth=1,
    )

    assert result == "result_marker"
    # Default RNG: a Generator instance was constructed.
    assert isinstance(captured["rng"], np.random.Generator)
    # Stage-to-stage threading: each stage saw the prior stage's output.
    assert captured["stage1_seen"] == "stage1_marker"
    assert captured["stage2_seen"] == "stage2_marker"
    assert captured["stage3_seen"] == "stage3_marker"


def test_batch_decode_default_variant_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``variant_padding`` is :attr:`VariantPadding.PAD_NULL`."""

    captured: dict[str, Any] = {}

    def fake_walk_sections(session, section_pointers, **kwargs):
        captured["variant_padding"] = kwargs["variant_padding"]
        return "stage1"

    monkeypatch.setattr(_entry_module, "walk_sections", fake_walk_sections)
    monkeypatch.setattr(
        _entry_module, "predict_lengths", lambda stage1, **kw: "stage2"
    )
    monkeypatch.setattr(
        _entry_module, "build_bulk_bytes", lambda stage2: "stage3"
    )
    monkeypatch.setattr(
        _entry_module, "assemble_batch", lambda stage3, **kw: "result"
    )

    batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        context_len=4,
        max_depth=1,
    )

    assert captured["variant_padding"] is VariantPadding.PAD_NULL


@pytest.mark.parametrize(
    "padding",
    [
        VariantPadding.PAD_NULL,
        VariantPadding.RESAMPLE_WITHIN_SECTION,
        VariantPadding.RAGGED,
        VariantPadding.REDISTRIBUTE,
    ],
)
def test_batch_decode_variant_padding_threaded(
    monkeypatch: pytest.MonkeyPatch, padding: VariantPadding
) -> None:
    """Every :class:`VariantPadding` policy is threaded through to
    :func:`walk_sections` verbatim."""

    captured: dict[str, Any] = {}

    def fake_walk_sections(session, section_pointers, **kwargs):
        captured["variant_padding"] = kwargs["variant_padding"]
        return "stage1"

    monkeypatch.setattr(_entry_module, "walk_sections", fake_walk_sections)
    monkeypatch.setattr(
        _entry_module, "predict_lengths", lambda stage1, **kw: "stage2"
    )
    monkeypatch.setattr(
        _entry_module, "build_bulk_bytes", lambda stage2: "stage3"
    )
    monkeypatch.setattr(
        _entry_module, "assemble_batch", lambda stage3, **kw: "result"
    )

    batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        context_len=4,
        max_depth=1,
        variant_padding=padding,
    )

    assert captured["variant_padding"] is padding


def test_batch_decode_kwargs_threaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """All optional kwargs reach the correct stage:

    * ``inlined_equivalent_call_targets_only`` -> stage 1.
    * ``context_len`` -> stage 2 + stage 4 (used by both).
    * ``include_fid_sidecar``, ``keep_intermediate`` -> stage 4.
    """

    captured_walk: dict[str, Any] = {}
    captured_predict: dict[str, Any] = {}
    captured_assemble: dict[str, Any] = {}

    def fake_walk_sections(session, section_pointers, **kwargs):
        captured_walk.update(kwargs)
        return "stage1"

    def fake_predict_lengths(stage1, **kwargs):
        captured_predict.update(kwargs)
        return "stage2"

    def fake_assemble_batch(stage3, **kwargs):
        captured_assemble.update(kwargs)
        return "result"

    monkeypatch.setattr(_entry_module, "walk_sections", fake_walk_sections)
    monkeypatch.setattr(_entry_module, "predict_lengths", fake_predict_lengths)
    monkeypatch.setattr(
        _entry_module, "build_bulk_bytes", lambda stage2: "stage3"
    )
    monkeypatch.setattr(_entry_module, "assemble_batch", fake_assemble_batch)

    explicit_rng = np.random.default_rng(seed=42)
    batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=3,
        context_len=16,
        max_depth=2,
        variant_padding=VariantPadding.RAGGED,
        inlined_equivalent_call_targets_only=True,
        include_fid_sidecar=True,
        keep_intermediate=True,
        rng=explicit_rng,
    )

    # Stage 1 kwargs.
    assert captured_walk["num_variants_per_section"] == 3
    assert captured_walk["max_depth"] == 2
    assert captured_walk["variant_padding"] is VariantPadding.RAGGED
    assert captured_walk["inlined_equivalent_call_targets_only"] is True
    assert captured_walk["rng"] is explicit_rng

    # Stage 2 kwargs.
    assert captured_predict["context_len"] == 16

    # Stage 4 kwargs.
    assert captured_assemble["context_len"] == 16
    assert captured_assemble["include_fid_sidecar"] is True
    assert captured_assemble["keep_intermediate"] is True


def test_batch_decode_returns_assemble_batch_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:func:`batch_decode` returns exactly what :func:`assemble_batch`
    returned -- no post-processing layer."""

    sentinel = object()

    monkeypatch.setattr(
        _entry_module, "walk_sections", lambda *a, **kw: "stage1"
    )
    monkeypatch.setattr(
        _entry_module, "predict_lengths", lambda stage1, **kw: "stage2"
    )
    monkeypatch.setattr(
        _entry_module, "build_bulk_bytes", lambda stage2: "stage3"
    )
    monkeypatch.setattr(
        _entry_module, "assemble_batch", lambda stage3, **kw: sentinel
    )

    result = batch_decode(
        session=None,  # type: ignore[arg-type]
        section_pointers=[],
        num_variants_per_section=1,
        context_len=4,
        max_depth=1,
    )
    assert result is sentinel


# ---------------------------------------------------------------------------
# End-to-end smoke against a real BinarySession
# ---------------------------------------------------------------------------


def test_batch_decode_end_to_end_synthetic(tmp_path) -> None:
    """End-to-end smoke against a tiny synthetic :class:`BinarySession`.

    Builds the shared on-disk corpus from :mod:`_session_fixture`,
    targets the matched section twice (so the variant-padding policy
    has both real and null-content slots to lay out), runs the full
    :func:`batch_decode` pipeline (stage 1 -> 4) through the real
    session API, and asserts the cross-array invariants on the
    resulting :class:`BatchDecodeResult`:

    * ``tokens`` shape + ``uint16`` dtype.
    * ``identity_row_offsets`` / ``number_row_offsets`` length =
      ``batch_size + 1``, monotone non-decreasing, terminal value
      equals the corresponding flat-array length.
    * ``batch_idx_to_section_variant`` shape + ``uint32`` dtype.
    * Some ``tokens`` cells are non-zero (real content is present).
    * The first row's identity slice is ``uint16``.
    """

    fb = build_synthetic_binary(tmp_path)

    # Two pointers at the matched section (idx=0, 2 variants each).
    # ``BinarySession`` opens one data arm per session by design, so a
    # single ``batch_decode`` call only spans one ``SectionKind``;
    # cross-arm batching is a higher-level orchestration concern.
    # With PAD_NULL + ``num_variants_per_section=3``, each pointer
    # fills 2 real slots + 1 null-content padding slot, so
    # ``batch_size = 2 * 3 = 6``.
    section_pointers = [
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
        SectionPointerSpec(arm=SectionKind.MATCHED, idx=0),
    ]
    num_variants_per_section = 3
    context_len = 64
    batch_size = len(section_pointers) * num_variants_per_section

    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as session:
        result = batch_decode(
            session,
            section_pointers=section_pointers,
            num_variants_per_section=num_variants_per_section,
            context_len=context_len,
            max_depth=2,
            variant_padding=VariantPadding.PAD_NULL,
            rng=np.random.default_rng(seed=42),
        )

    assert isinstance(result, BatchDecodeResult)

    # tokens: shape + dtype.
    assert result.tokens.shape == (batch_size, context_len)
    assert result.tokens.dtype == np.uint16

    # Identity / number row offsets: length, monotone, terminal-equals-length.
    assert result.identity_row_offsets.shape == (batch_size + 1,)
    assert np.all(np.diff(result.identity_row_offsets) >= 0)
    assert int(result.identity_row_offsets[-1]) == result.identities.shape[0]

    assert result.number_row_offsets.shape == (batch_size + 1,)
    assert np.all(np.diff(result.number_row_offsets) >= 0)
    assert (
        int(result.number_row_offsets[-1])
        == result.numbers_significant.shape[0]
    )
    # The two number arrays are parallel.
    assert (
        result.numbers_sign_exponent.shape == result.numbers_significant.shape
    )

    # batch_idx_to_section_variant shape + dtype.
    assert result.batch_idx_to_section_variant.shape == (batch_size, 2)
    assert result.batch_idx_to_section_variant.dtype == np.uint32

    # Cells in ``tokens`` are either 0 (null-content padding) or >= 1
    # (the post-shift smallest id). u16 is unsigned so the lower bound
    # is trivial; the meaningful check is that real content is present
    # somewhere in the batch -- the synthetic corpus has 2 real
    # variants per matched section pointer, so some cells must be
    # non-padding.
    assert int((result.tokens >= 1).sum()) > 0

    # First batch row exists and its identity slice (possibly empty)
    # is uint16.
    row0_start = int(result.identity_row_offsets[0])
    row0_stop = int(result.identity_row_offsets[1])
    row0_identities = result.identities[row0_start:row0_stop]
    assert row0_identities.dtype == np.uint16


def test_batch_decode_prepends_variant_tokens_once_per_row(tmp_path) -> None:
    """Each row's variant-axis prefix is contributed ONCE, at row start,
    as a row-level identity prefix BEFORE any call_target body.

    Every function in a single splice tree shares the same compilation
    variant axis (same binary -> same arch/compiler/opt), so the
    variant-axis prefix is a row-level property, not a per-function one.
    Stage 1 (:mod:`._callee_walk`) feeds ``function_data.tokens`` (body
    only) into :func:`build_inline_decode_state` for every call_target
    -- root AND inlined callees alike, no special case. The
    row-level :attr:`Stage1Variant.variant_tokens` carries the prefix
    separately and Stage 4 emits it at row column 0 before any
    call_target body, with the per-call-target LOCAL_FUNC self-token
    landing AT root body start (slot ``n_axis``).

    Row layout (plan D3 + ALG-9):

        row[0..n_axis]            = variant_tokens, post-shift (id - 256)
        row[n_axis]               = LOCAL_FUNC self-token at root body start
        row[n_axis + 1 ..]        = root body tokens, post-shift

    The walker-level proof that callees do NOT also carry the
    variant_tokens prefix lives in
    :func:`test_variant_tokens_prepended_only_at_root_not_at_callees`
    in ``test_callee_walk.py``; the synthetic ``BinarySession`` fixture
    here has no real callees in any of its rows.
    """
    from tokenizer.token_manager import VocabularyManager

    reserved_digit_count = VocabularyManager._V2_RESERVED_DIGIT_COUNT

    fb = build_synthetic_binary(tmp_path)

    # One pointer + two real variants exhausts the matched section
    # without padding rows, so EVERY batch row has real content whose
    # first n_axis + 1 tokens are determined: variant_tokens then the
    # LOCAL_FUNC self-token marking root body start.
    section_pointers = [SectionPointerSpec(arm=SectionKind.MATCHED, idx=0)]
    num_variants_per_section = 2
    context_len = 32
    batch_size = len(section_pointers) * num_variants_per_section

    with BinarySession(
        fb["base_path"], fb["binary_name"], fb["vocab"], fb["metadata"]
    ) as session:
        # Snapshot the per-variant variant_tokens BEFORE running
        # batch_decode so we can compare them against the row tensor
        # without coupling to internal stage-1 wiring.
        mf = session.load_matched(0)
        variant_tokens_per_slot = [v.variant_tokens for v in mf.variants]

        result = batch_decode(
            session,
            section_pointers=section_pointers,
            num_variants_per_section=num_variants_per_section,
            context_len=context_len,
            max_depth=1,
            variant_padding=VariantPadding.PAD_NULL,
            rng=np.random.default_rng(seed=42),
        )

    # Sanity: every variant in the fixture resolves the same single
    # _variants.bin record, so variant_tokens is identical across slots
    # AND non-empty (the resolver populates it from the encoded record).
    assert len(variant_tokens_per_slot) == num_variants_per_section
    n_axis = int(variant_tokens_per_slot[0].shape[0])
    assert n_axis > 0
    for vt in variant_tokens_per_slot[1:]:
        np.testing.assert_array_equal(vt, variant_tokens_per_slot[0])

    # Each row's layout (plan D3 + ALG-9 + Stage 4 row assembly):
    #   row[0..n_axis]      = variant_tokens, post-shift (id - 256)
    #   row[n_axis]         = LOCAL_FUNC self-token at root body start
    #   row[n_axis + 1..]   = root body tokens, post-shift
    #
    # ``LOCAL_FUNC`` is at IDENTITY block offset 1 (the matched fixture
    # has only LOCAL call sites; PLT/EXT do not arise here). Computing
    # the shifted id from :class:`VocabularyManager` keeps the test
    # decoupled from the literal constant 9.
    local_func_shifted = (
        VocabularyManager._V2_IDENTITY_BLOCK_START
        + 1
        - reserved_digit_count
    )
    expected_axis_shifted = (
        variant_tokens_per_slot[0].astype(np.int32) - reserved_digit_count
    ).astype(np.uint16)

    assert result.tokens.shape == (batch_size, context_len)
    for row_idx in range(batch_size):
        row = result.tokens[row_idx]
        np.testing.assert_array_equal(
            row[:n_axis],
            expected_axis_shifted,
            err_msg=(
                f"row {row_idx}: variant_tokens did not land at slots "
                f"0..{n_axis} of the model-facing token stream"
            ),
        )
        assert int(row[n_axis]) == local_func_shifted, (
            f"row {row_idx}: expected LOCAL_FUNC prepend "
            f"({local_func_shifted}) at slot {n_axis} (root body start), "
            f"got {int(row[n_axis])}"
        )
