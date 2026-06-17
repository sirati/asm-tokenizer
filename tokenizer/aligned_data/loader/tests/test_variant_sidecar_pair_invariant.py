"""``BinaryDataset`` hard-fails on a half-present variant-axis sidecar pair.

The builder writes ``<bin>_variants.bin`` (axis token records) and
``<bin>_variants.csv`` (the byte-offset->filename refs) in lockstep.
Either both exist (consistent, has variant axes) or neither does
(legacy / no-variant-axis corpus). An INCONSISTENT corpus -- one side
present, the other missing -- would otherwise decode SILENTLY without
the axis prefix: the resolver reads the present record but has no refs
to map it, the session swallows the resulting ``KeyError`` to ``None``,
and the parser substitutes an empty variant-token stream. That corrupts
every training row with NO signal, so ``BinaryDataset`` construction
must RAISE on the half-present state.

These tests pin all three states:
  * both present + consistent  -> constructs + resolves the axis prefix
  * both absent (legacy)       -> constructs, no prefix, no error
  * exactly one present        -> raises ValueError naming the missing file
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset

from ._session_fixture import build_synthetic_binary


def _build(tmp_path: Path):
    """On-disk synthetic corpus + its sidecar paths.

    ``build_synthetic_binary`` lays down BOTH variant sidecars (the
    consistent state); the tests below remove one or both to exercise
    the legacy + inconsistent states.
    """
    fb = build_synthetic_binary(tmp_path)
    base = Path(fb["base_path"])
    name = fb["binary_name"]
    return (
        fb,
        base / f"{name}_variants.bin",
        base / f"{name}_variants.csv",
    )


def test_both_sidecars_present_resolves_prefix(tmp_path) -> None:
    """Consistent corpus: construction succeeds and the axis prefix
    resolves (non-empty ``variant_tokens``)."""
    fb, var_bin, var_csv = _build(tmp_path)
    assert var_bin.exists() and var_csv.exists()

    ds = BinaryDataset(fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"])
    with ds.open_session() as sess:
        mf = sess.load_matched(0)
    # A resolved variant carries the 4 positional axis tokens.
    assert mf.variants[0].variant_tokens.shape[0] >= 4


def test_both_sidecars_absent_is_legacy_no_raise(tmp_path) -> None:
    """Legacy / no-variant-axis corpus: BOTH sidecars absent constructs
    fine and emits NO axis prefix (the historical tolerant path)."""
    fb, var_bin, var_csv = _build(tmp_path)
    var_bin.unlink()
    var_csv.unlink()

    ds = BinaryDataset(fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"])
    with ds.open_session() as sess:
        mf = sess.load_matched(0)
    # No variant sidecar -> resolver short-circuits -> empty prefix.
    assert mf.variants[0].variant_tokens.shape == (0,)


def test_bin_present_csv_absent_raises(tmp_path) -> None:
    """Inconsistent: ``_variants.bin`` present, ``_variants.csv`` missing
    -> RAISE naming the missing CSV (the silent-corruption guard)."""
    fb, var_bin, var_csv = _build(tmp_path)
    var_csv.unlink()
    assert var_bin.exists() and not var_csv.exists()

    with pytest.raises(ValueError) as exc:
        BinaryDataset(fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"])
    msg = str(exc.value)
    assert fb["binary_name"] in msg
    assert var_csv.name in msg


def test_csv_present_bin_absent_raises(tmp_path) -> None:
    """Inconsistent the other way: ``_variants.csv`` present,
    ``_variants.bin`` missing -> RAISE naming the missing bin."""
    fb, var_bin, var_csv = _build(tmp_path)
    var_bin.unlink()
    assert var_csv.exists() and not var_bin.exists()

    with pytest.raises(ValueError) as exc:
        BinaryDataset(fb["base_path"], fb["binary_name"], vocab_manager=fb["vocab"])
    msg = str(exc.value)
    assert fb["binary_name"] in msg
    assert var_bin.name in msg
