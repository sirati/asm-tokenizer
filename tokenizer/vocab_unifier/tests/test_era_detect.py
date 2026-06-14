"""Per-CSV vocab-era detection tests.

Covers :func:`tokenizer.vocab_unifier.era_detect.detect_legacy_no_value_negative`:

* MODERN (era #3, offset 257) CSVs are POSITIVELY detected → returns
  ``False`` (``legacy_no_value_negative=False``) regardless of the
  caller default.
* LEGACY (era #1, offset 256) CSVs are carrier-blind → the detector
  returns the caller default (it cannot positively confirm legacy).
* A MIXED set detects each file's era correctly.
* A UNIFORM modern set detects every file as modern (no regression vs
  today's correct single-flag behaviour).
* A degenerate / too-short CSV (no disambiguating carriers) falls back
  to the caller default.

The fixtures CONSTRUCT both eras at the wire level. A modern CSV's
vocab-def is produced by the real saver (which strips the 257-slot
reserved prefix). A legacy CSV's vocab-def is hand-built so the
serialized vocab/types begin at per-binary id 256 with NO
``value_negative`` — exactly what ``load_vocab_manager(...,
legacy_no_value_negative=True)`` reconstitutes.
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
from tokenizer.vocab_unifier.era_detect import detect_legacy_no_value_negative

_DIGIT = VocabularyManager._V2_RESERVED_DIGIT_COUNT  # 256
_SIGN = VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID  # 256

# Per-binary content tokens past the reserved band. names[0]=block_v2 in
# BOTH eras (the IDENTITY anchor). A spread of distinct instructions plus
# a valued_const_v2 so the carrier id range is exercised densely.
_CONTENT_NAMES = (
    ["block_v2"]
    + [f"x64_insn{i}" for i in range(40)]
    + ["valued_const_v2"]
)
_CONTENT_TYPES = (
    [TokenType.BLOCK_V2]
    + [TokenType.PLATFORM] * 40
    + [TokenType.VALUED_CONST_V2]
)
assert len(_CONTENT_NAMES) == len(_CONTENT_TYPES)
_N_CONTENT = len(_CONTENT_NAMES)


def _vocab_def_row_bytes() -> bytes:
    """Wire-format per-binary vocab-def row whose serialized vocab/types
    span is exactly ``_CONTENT_NAMES`` / ``_CONTENT_TYPES``.

    Identical bytes serve BOTH eras: the serialized span is the
    post-reserved content in either case. The era is decided purely by
    which reserved prefix the LOADER prepends (256 vs 257), so a single
    serialized row faithfully models both — exactly the structural
    indistinguishability this detector must work around.
    """
    type_arr = np.array([int(t) for t in _CONTENT_TYPES], dtype=np.int8)
    pit = np.full(_N_CONTENT, PlatformInstructionTypes.AGNOSTIC, dtype=np.int8)
    row = [
        "vocabulary",
        ",".join(_CONTENT_NAMES),
        f"_id_to_token_type norm:{0 + TokenType.UNRESOLVED}",
        ndarray_to_base64(type_arr - TokenType.UNRESOLVED),
        f"_platform_instruction_type_cache norm:{0 + PlatformInstructionTypes.UNRESOLVED}",
        ndarray_to_base64(pit - PlatformInstructionTypes.UNRESOLVED),
        "_lit_start_cache",
        ndarray_to_base64(np.array([], dtype=np.int_)),
        "_lit_end_cache",
        ndarray_to_base64(np.array([], dtype=np.int_)),
        "format_version",
        "2",
    ]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(row)
    return buf.getvalue().encode("ascii")


def _record_stream(era: int, n_blocks: int) -> np.ndarray:
    """A plausible token stream of ``n_blocks`` basic blocks under ``era``.

    era=3 (modern): block_v2@257, instructions@258.., valued_const_v2@298
    (the TOP carrier id) with a ``value_negative``(256) postfix on the
    negative literal. era=1 (legacy): the same content shifted down one
    (block_v2@256, top carrier@297) and NO ``value_negative`` marker.
    Each block touches the FULL carrier id range so the top-id
    out-of-range signal is dense under a wrong-offset load.
    """
    base = _DIGIT + (1 if era == 3 else 0)
    ids: list[int] = []
    for _ in range(n_blocks):
        ids.append(base + 0)  # block_v2 (frequent opener)
        for k in range(1, _N_CONTENT - 1):  # every instruction
            ids += [base + k, k % _DIGIT]  # marker + one inline digit
        # valued_const_v2 (TOP carrier id) with a (negative) inline value
        ids += [base + (_N_CONTENT - 1), 0x05]
        if era == 3:
            ids.append(_SIGN)  # value_negative postfix (modern only)
    return np.array(ids, dtype=np.uint16)


def _write_csv(path: Path, era: int, *, n_blocks: int = 20) -> Path:
    """Write a full per-binary tokenize CSV (v2 prelude + header + record
    rows + trailing vocab-def line) for ``era`` at ``path``. Filename
    carries the ``x64-`` arch prefix the loader's platform auto-detect
    keys on.
    """
    stream_b64 = ndarray_to_base64(_record_stream(era, n_blocks))
    block_rl_b64 = ndarray_to_base64(np.array([1, 2], dtype=np.uint16))
    insn_rl_b64 = ndarray_to_base64(np.array([1, 2], dtype=np.uint16))

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["version=2"])
    writer.writerow(
        [
            "function_name",
            "occurrence",
            "tokens_base64",
            "block_runlength_base64",
            "instruction_runlength_base64",
            "metadata",
        ]
    )
    # A couple of record rows carrying the era-appropriate token stream.
    for fn in range(2):
        writer.writerow(
            [f"func_{fn}", 0, stream_b64, block_rl_b64, insn_rl_b64, "{}"]
        )
    text = buf.getvalue().encode("ascii")
    # Append the trailing vocab-def line.
    text += _vocab_def_row_bytes()

    path.write_bytes(text)
    return path


def _write_degenerate_csv(path: Path) -> Path:
    """A valid-vocab CSV with ZERO record rows (header + vocab-def only) —
    no token stream, hence no disambiguating carriers."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["version=2"])
    writer.writerow(
        [
            "function_name",
            "occurrence",
            "tokens_base64",
            "block_runlength_base64",
            "instruction_runlength_base64",
            "metadata",
        ]
    )
    text = buf.getvalue().encode("ascii") + _vocab_def_row_bytes()
    path.write_bytes(text)
    return path


# ---------------------------------------------------------------------------
# Positive modern detection (overrides the default both ways)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("default", [True, False])
def test_modern_csv_detected_as_257(tmp_path: Path, default: bool) -> None:
    """A MODERN (era #3) CSV is positively detected as offset-257
    (``legacy_no_value_negative=False``) REGARDLESS of the caller default
    — the carrier coherence is decisive."""
    csv_path = _write_csv(tmp_path / "x64-modern_output.csv", era=3)
    detected = detect_legacy_no_value_negative(csv_path, default=default)
    assert detected is False


# ---------------------------------------------------------------------------
# Legacy is carrier-blind → defers to the default
# ---------------------------------------------------------------------------


def test_legacy_csv_defers_to_legacy_default(tmp_path: Path) -> None:
    """A LEGACY (era #1) CSV resolves cleanly under BOTH offsets, so the
    detector cannot positively confirm it and returns the caller default.
    With the legacy default (True) it lands on the correct era."""
    csv_path = _write_csv(tmp_path / "x64-legacy_output.csv", era=1)
    assert detect_legacy_no_value_negative(csv_path, default=True) is True


def test_legacy_csv_with_modern_default_returns_modern(tmp_path: Path) -> None:
    """The carrier-blindness is honest: a legacy CSV under a MODERN
    default returns modern (False). Documents the one-directional limit —
    a mixed corpus MUST pass the legacy default so legacy files resolve
    correctly while modern files self-upgrade via positive detection."""
    csv_path = _write_csv(tmp_path / "x64-legacy_output.csv", era=1)
    assert detect_legacy_no_value_negative(csv_path, default=False) is False


# ---------------------------------------------------------------------------
# Mixed corpus
# ---------------------------------------------------------------------------


def test_mixed_corpus_each_detected(tmp_path: Path) -> None:
    """A MIXED set (legacy untouched + modern re-tokenized) under the
    legacy default: every modern file self-upgrades to 257 and every
    legacy file keeps 256 — the exact mixed-era workflow."""
    legacy_a = _write_csv(tmp_path / "x64-legacy_a_output.csv", era=1)
    legacy_b = _write_csv(tmp_path / "x64-legacy_b_output.csv", era=1)
    modern_a = _write_csv(tmp_path / "x64-modern_a_output.csv", era=3)
    modern_b = _write_csv(tmp_path / "x64-modern_b_output.csv", era=3)

    default = True  # legacy default (the mixed-corpus operator setting)
    assert detect_legacy_no_value_negative(legacy_a, default=default) is True
    assert detect_legacy_no_value_negative(legacy_b, default=default) is True
    assert detect_legacy_no_value_negative(modern_a, default=default) is False
    assert detect_legacy_no_value_negative(modern_b, default=default) is False


# ---------------------------------------------------------------------------
# Uniform modern — no regression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("default", [True, False])
def test_uniform_modern_all_detected_257(tmp_path: Path, default: bool) -> None:
    """Every CSV in a UNIFORM modern corpus detects 257, regardless of
    the default. This is the no-regression guarantee: today's correct
    single-flag (modern) behaviour is reproduced file-by-file."""
    paths = [
        _write_csv(tmp_path / f"x64-mod{i}_output.csv", era=3) for i in range(5)
    ]
    for p in paths:
        assert detect_legacy_no_value_negative(p, default=default) is False


# ---------------------------------------------------------------------------
# Degenerate / too-short → default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("default", [True, False])
def test_degenerate_csv_falls_back_to_default(
    tmp_path: Path, default: bool
) -> None:
    """A CSV with no record rows (no disambiguating carriers) returns the
    caller default unchanged in both directions."""
    csv_path = _write_degenerate_csv(tmp_path / "x64-empty_output.csv")
    assert detect_legacy_no_value_negative(csv_path, default=default) is default
