"""Saver + loader hard-cutover tests.

Covers:

* Unified-vocab format_version=1 round-trip (in-scope memmap-chain
  version per plan memoized-booping-wren.md).
* Per-binary-CSV format_version=2 round-trip (out-of-scope tokenize
  output; the saver/loader still serve it so this test guards against
  collateral damage from the cleanup).
* Hard-cutover rejections: legacy v1-no-trailer rows; trailers
  declaring format_version=3 or =4; the saver-side ValueError on a
  programmer-supplied unsupported version.

Fixtures bypass the constructor by calling ``VocabularyManager.from_vocab``
with explicit pre-baked arrays so the tests do not depend on
constructor-side digit-prefill behavior (which lives in a sibling
file outside this subtask's scope).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pytest

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType
from tokenizer.vocab_unifier.loader import (
    is_vocab_def,
    load_vocab_manager_csv_row_bytes,
)
from tokenizer.vocab_unifier.saver import save_vocabulary


_DIGIT_COUNT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_RESERVED = VocabularyManager._V2_RESERVED_TOKEN_COUNT  # 257 (digits + value_negative)
_EAGER_BLOCK_END = VocabularyManager._V2_EAGER_BLOCK_END  # 272 (reserved + canonical blocks)

# Canonical number + identity blocks pre-registered by every unified VM
# produced by the post-Phase-4 unifier (see
# ``VocabularyManager._register_v2_canonical_blocks``). The wire-format
# strings and their TokenTypes are the source-declaration order of the
# Inner-class registration sequence; mirroring them here keeps test
# fixtures aligned with the head-of-vocab invariant asserted by
# ``VocabularyManager.from_vocab``.
_CANONICAL_BLOCK_NAMES = (
    # Number block 257..263
    "valued_const_v2",
    "float16", "bfloat16", "float32", "float64", "float80", "float128",
    # Identity block 264..271 (user-canonical first 5 then alphabetical)
    "block_v2",
    "local_func", "plt_func", "ext_func", "string_ptr",
    "jump_table", "ro_data_ptr", "rw_data_ptr",
)
_CANONICAL_BLOCK_TYPES = (
    TokenType.VALUED_CONST_V2,
    TokenType.FLOAT16, TokenType.BFLOAT16, TokenType.FLOAT32,
    TokenType.FLOAT64, TokenType.FLOAT80, TokenType.FLOAT128,
    TokenType.BLOCK_V2,
    TokenType.LOCAL_FUNC, TokenType.PLT_FUNC, TokenType.EXT_FUNC,
    TokenType.STRING_PTR, TokenType.JUMP_TABLE,
    TokenType.RO_DATA_PTR, TokenType.RW_DATA_PTR,
)
assert len(_CANONICAL_BLOCK_NAMES) == _EAGER_BLOCK_END - _RESERVED
assert len(_CANONICAL_BLOCK_TYPES) == _EAGER_BLOCK_END - _RESERVED


def _make_vm(
    *,
    platform: str | None,
    real_tokens: list[str],
    format_version: int,
) -> VocabularyManager:
    """Build a VocabularyManager with the protocol-reserved prefix
    (256 digit slots + ``value_negative`` at slot 256) up front, followed
    by — for unified VMs only — the canonical number+identity blocks at
    slots 257..271, then ``real_tokens``. Uses ``from_vocab`` so the call
    is independent of constructor-side prefill logic but mirrors the same
    wire shape the post-Phase-4 unifier produces (so the head-of-vocab
    invariant asserted inside ``from_vocab`` for unified VMs holds).

    Per-binary VMs keep the legacy shape — they never carry the canonical
    blocks (those are registered lazily by tokenization, not eagerly)."""
    digit_names = [f"digit_{i:02X}" for i in range(_DIGIT_COUNT)]
    canonical_block = list(_CANONICAL_BLOCK_NAMES) if platform is None else []
    vocab_list = digit_names + ["value_negative"] + canonical_block + real_tokens
    total = len(vocab_list)

    id_to_token_type = np.full(total, TokenType.UNRESOLVED, dtype=np.int8)
    # The `value_negative` marker carries its own token type; the saver/
    # loader round-trip must preserve it.
    id_to_token_type[_DIGIT_COUNT] = TokenType.VALUE_NEGATIVE
    # Canonical block tokens carry the source-declaration token types
    # the unifier's `_register_v2_canonical_blocks` writes them with.
    real_token_start = _RESERVED + len(canonical_block)
    for offset, ttype in enumerate(_CANONICAL_BLOCK_TYPES[: len(canonical_block)]):
        id_to_token_type[_RESERVED + offset] = ttype
    # Mark real tokens with a non-reserved type so the round-trip surfaces
    # any normalization bugs in the saver's `- TokenType.UNRESOLVED` step.
    for i in range(real_token_start, total):
        id_to_token_type[i] = TokenType.PLATFORM

    platform_instruction_type_cache = np.full(
        total, PlatformInstructionTypes.AGNOSTIC, dtype=np.int8
    )

    # Tiny lit caches — content irrelevant to format-version branching.
    lit_start_cache = np.array([], dtype=np.int_)
    lit_end_cache = np.array([], dtype=np.int_)

    platform_list = None
    token_to_platform = None
    if platform is None:
        platform_list = ["x64", "arm64"]
        token_to_platform = np.full(total, -1, dtype=np.int8)
        for i in range(real_token_start, total):
            token_to_platform[i] = (i - real_token_start) % len(platform_list)

    return VocabularyManager.from_vocab(
        platform=platform,
        vocab_list=vocab_list,
        id_to_token_type=id_to_token_type,
        platform_instruction_type_cache=platform_instruction_type_cache,
        lit_start_cache=lit_start_cache,
        lit_end_cache=lit_end_cache,
        platform_list=platform_list,
        token_to_platform=token_to_platform,
        format_version=format_version,
    )


def _save_to_bytes(vm: VocabularyManager) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='\n')
    save_vocabulary(vm, writer)
    return buf.getvalue().encode("ascii")


def _craft_unsupported_row(
    *,
    platform_label: str,
    declared_version: str | None,
    omit_trailer: bool = False,
) -> bytes:
    """Build a wire-format vocab row whose trailer declares
    ``declared_version`` (or has no trailer if ``omit_trailer``).

    The base cells are filled with the minimum valid shape — empty
    base64 payloads, a single real token. This is enough to pass shape
    asserts; the test asserts the loader still rejects.
    """
    is_unified = platform_label == "unified"
    real_tokens = ["real_token_a"]
    digit_names = [f"digit_{i:02X}" for i in range(_DIGIT_COUNT)]
    vocab_list = digit_names + ["value_negative"] + real_tokens
    # Saver normally strips the protocol-reserved prefix; mirror that.
    serialized_vocab = ",".join(vocab_list[_RESERVED:])

    # Minimal valid base64 payloads (1 entry per array; matches len(real_tokens)).
    empty_int8 = ndarray_to_base64(np.zeros(1, dtype=np.int8))
    empty_int = ndarray_to_base64(np.array([], dtype=np.int_))

    row = [
        "vocabulary",
        serialized_vocab,
        f"_id_to_token_type norm:{0 + TokenType.UNRESOLVED}",
        empty_int8,
        f"_platform_instruction_type_cache norm:{0 + PlatformInstructionTypes.UNRESOLVED}",
        empty_int8,
        "_lit_start_cache",
        empty_int,
        "_lit_end_cache",
        empty_int,
    ]
    if is_unified:
        row += [
            "platforms norm:-1",
            "x64,arm64",
            ndarray_to_base64(np.zeros(1, dtype=np.int8)),
        ]
    if not omit_trailer:
        assert declared_version is not None
        row += ["format_version", declared_version]

    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow(row)
    return buf.getvalue().encode("ascii")


# ---------------------------------------------------------------------------
# Happy-path round-trips
# ---------------------------------------------------------------------------


def test_unified_v1_round_trip() -> None:
    """Unified vocab (platform=None, format_version=1) survives a
    save/load round-trip with id_to_token + reserved-digit
    reconstruction intact."""
    vm = _make_vm(
        platform=None,
        real_tokens=["arch:x64", "arch:arm64", "Block_00"],
        format_version=1,
    )
    raw = _save_to_bytes(vm)
    loaded = load_vocab_manager_csv_row_bytes(raw, "unified")

    assert loaded is not None
    assert loaded.format_version == 1
    assert len(loaded.id_to_token) == len(vm.id_to_token)
    # Reserved-digit slots reconstituted.
    for i in range(_DIGIT_COUNT):
        assert loaded.id_to_token[i] == f"digit_{i:02X}"
    # value_negative marker reconstituted at slot 256 (not serialised on
    # the wire — protocol-reserved like the digits).
    assert loaded.id_to_token[_DIGIT_COUNT] == "value_negative"
    # Real tokens round-trip byte-identically.
    assert loaded.id_to_token[_RESERVED:] == vm.id_to_token[_RESERVED:]
    # Token-to-platform array round-trips for the unified layout.
    assert loaded.platform_list == vm.platform_list


def test_per_binary_v2_round_trip() -> None:
    """Per-binary CSV (platform='x64', format_version=2) survives the
    same round-trip. The legacy-purge MUST NOT break this out-of-scope
    path."""
    vm = _make_vm(
        platform="x64",
        real_tokens=["x64_mov", "x64_add", "Block_00"],
        format_version=2,
    )
    raw = _save_to_bytes(vm)
    loaded = load_vocab_manager_csv_row_bytes(raw, "x64")

    assert loaded is not None
    assert loaded.format_version == 2
    assert len(loaded.id_to_token) == len(vm.id_to_token)
    for i in range(_DIGIT_COUNT):
        assert loaded.id_to_token[i] == f"digit_{i:02X}"
    # value_negative marker reconstituted at slot 256 on per-binary too.
    assert loaded.id_to_token[_DIGIT_COUNT] == "value_negative"
    assert loaded.id_to_token[_RESERVED:] == vm.id_to_token[_RESERVED:]


def test_unified_v1_trailer_cells_present() -> None:
    """Saver always stamps the 2-cell trailer for v1 unified vocabs.
    Wire layout: 13 base cells + 2 trailer cells = 15 total."""
    vm = _make_vm(
        platform=None,
        real_tokens=["arch:x64"],
        format_version=1,
    )
    raw = _save_to_bytes(vm)
    rows = list(csv.reader(io.StringIO(raw.decode("ascii"))))
    assert len(rows) == 1
    row = rows[0]
    assert len(row) == 15
    assert row[13] == "format_version"
    assert row[14] == "1"


# ---------------------------------------------------------------------------
# Hard-cutover rejections (loader)
# ---------------------------------------------------------------------------


def test_legacy_v1_no_trailer_unified_rejected() -> None:
    """A unified row with the legacy 13-cell shape (no trailer) is
    rejected: is_vocab_def returns False and the load wrapper returns
    None (the shape assert fires inside is_vocab_def's try/except)."""
    raw = _craft_unsupported_row(
        platform_label="unified",
        declared_version=None,
        omit_trailer=True,
    )
    ok, _ = is_vocab_def(raw, "unified")
    assert ok is False
    # Public entry point translates the failed shape check to None.
    assert load_vocab_manager_csv_row_bytes(raw, "unified") is None


def test_legacy_v1_no_trailer_per_binary_rejected() -> None:
    """Same shape rejection for the 10-cell per-binary layout."""
    raw = _craft_unsupported_row(
        platform_label="x64",
        declared_version=None,
        omit_trailer=True,
    )
    ok, _ = is_vocab_def(raw, "x64")
    assert ok is False
    assert load_vocab_manager_csv_row_bytes(raw, "x64") is None


def test_trailer_format_version_3_rejected() -> None:
    """A trailer declaring format_version=3 raises ValueError mentioning
    the unsupported value. The reader has no knowledge of "v3 was a
    thing once" — it just enforces membership in {1, 2}."""
    raw = _craft_unsupported_row(
        platform_label="unified",
        declared_version="3",
    )
    with pytest.raises(ValueError, match="got 3"):
        load_vocab_manager_csv_row_bytes(raw, "unified")


def test_trailer_format_version_4_rejected() -> None:
    """Same membership check for a hypothetical future version that
    hasn't been wired through this writer/reader yet."""
    raw = _craft_unsupported_row(
        platform_label="unified",
        declared_version="4",
    )
    with pytest.raises(ValueError, match="got 4"):
        load_vocab_manager_csv_row_bytes(raw, "unified")


# ---------------------------------------------------------------------------
# Hard-cutover rejection (saver)
# ---------------------------------------------------------------------------


def test_saver_rejects_unsupported_format_version() -> None:
    """Saver guards its contract: any format_version outside {1, 2}
    raises ValueError before any bytes are written."""
    vm = _make_vm(
        platform=None,
        real_tokens=["arch:x64"],
        format_version=1,
    )
    # Bypass the from_vocab constructor's defaulting by mutating
    # post-hoc — the saver's check is the only thing exercised here.
    vm.format_version = 99

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator='\n')
    with pytest.raises(ValueError, match="got 99"):
        save_vocabulary(vm, writer)


# ---------------------------------------------------------------------------
# value_negative strip + reconstruction
# ---------------------------------------------------------------------------


def _vocab_cell(raw: bytes) -> list[str]:
    """Extract the comma-separated vocabulary cell from a serialised
    vocab row. The cell sits at row index 1 (per ``save_vocabulary``)."""
    row = next(csv.reader(io.StringIO(raw.decode("ascii")), quotechar='"'))
    return row[1].split(",")


def test_saver_strips_value_negative_per_binary() -> None:
    """``value_negative`` (slot 256) MUST NOT appear in the serialised
    per-binary vocab cell. The marker is protocol-reserved (same
    treatment as digit slots 0..255) and reconstituted by the loader."""
    vm = VocabularyManager(platform="x64", format_version=2)
    # Register a handful of representative tokens to exercise the strip
    # boundary — the marker still must not surface in the wire vocab.
    vm.Block_V2(0)
    vm.Block_V2(1)

    buf = io.StringIO()
    save_vocabulary(vm, csv.writer(buf, lineterminator='\n'))
    raw = buf.getvalue().encode("ascii")

    assert "value_negative" not in _vocab_cell(raw), (
        "value_negative must not appear in serialised per-binary vocab cell"
    )

    # Round-trip — reload and assert slot 256 still pins value_negative
    # with the correct token type.
    vm2 = load_vocab_manager_csv_row_bytes(raw, "x64")
    assert vm2 is not None
    assert vm2.id_to_token[256] == "value_negative"
    assert vm2.get_token_id("value_negative") == 256
    assert vm2.id_to_token_type[256] == TokenType.VALUE_NEGATIVE


def test_saver_strips_value_negative_unified() -> None:
    """Same protocol-reserved treatment for the unified-vocab format
    (``format_version=1``, ``platform=None``)."""
    vm = VocabularyManager(platform=None, format_version=1)
    # Pre-register the canonical NUMBER+IDENTITY blocks at slots 257..271
    # so the round-trip lands on a vocab whose head matches the
    # ``from_vocab`` head-of-vocab invariant; without this, the loader
    # would assert "expected 'valued_const_v2' at slot 257".
    vm._register_v2_canonical_blocks()
    # Register one Variant_Axis token so the wire vocab carries something
    # past the canonical tail; the strip boundary is what's under test,
    # not the cell contents.
    vm.Variant_Axis("arch:x64")

    buf = io.StringIO()
    save_vocabulary(vm, csv.writer(buf, lineterminator='\n'))
    raw = buf.getvalue().encode("ascii")

    assert "value_negative" not in _vocab_cell(raw), (
        "value_negative must not appear in serialised unified vocab cell"
    )

    vm2 = load_vocab_manager_csv_row_bytes(raw, "unified")
    assert vm2 is not None
    assert vm2.id_to_token[256] == "value_negative"
    assert vm2.get_token_id("value_negative") == 256
    assert vm2.id_to_token_type[256] == TokenType.VALUE_NEGATIVE
