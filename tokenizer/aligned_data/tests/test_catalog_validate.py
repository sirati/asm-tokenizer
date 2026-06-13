"""Unit tests for the standalone per-call-J corruption validator.

Single concern: pin :func:`validate_per_call_js` + the
:class:`CorruptionReport` shape + the ``--json`` CLI on a clean corpus
and on a deliberately-corrupt corpus (per-call ``J`` byte-patched on
disk to inject the validator's two corruption shapes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.catalog_validate import (
    CorruptCatalogError,
    CorruptionKind,
    CorruptionReport,
    require_clean,
    validate_per_call_js,
    main,
)
from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.loader.tests._corpus import (
    MatchedFunctionSpec,
    VariantSpec,
    build_corpus,
    build_corrupt_per_call_j_corpus,
    make_simple_variant,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)


def _cols(base: Path, name: str):
    starts, lengths = read_csv_section_index_arrays(base / f"{name}_index.bin")
    blob = np.fromfile(base / f"{name}_sections.bin", dtype=np.uint8)
    return parse_sections_columnar(blob, starts, lengths), starts


def _callset(seed: int, per_variant_called):
    out = []
    for i, called in enumerate(per_variant_called):
        b = make_simple_variant(("V", i), token_seed=seed + i, n_tokens=6 + i)
        out.append(
            VariantSpec(
                vkey=b.vkey, tokens=b.tokens, block_rl=b.block_rl,
                insn_rl=b.insn_rl, called=tuple(called),
            )
        )
    return tuple(out)


def _shared(n, seed):
    return tuple(
        make_simple_variant(("V", i), token_seed=seed + i, n_tokens=6 + i)
        for i in range(n)
    )


def test_clean_corpus_reports_no_corruption(tmp_path: Path) -> None:
    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_callset(1, [("leaf",), ()]),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf", variants=_shared(2, 21), called=(),
        ),
    ]
    build_corpus(tmp_path, "clean", matched=specs)
    cols, starts = _cols(tmp_path, "clean")
    report = validate_per_call_js(cols, starts)
    assert isinstance(report, CorruptionReport)
    assert not report.is_corrupt
    assert report.total_corrupt == 0


def test_corrupt_corpus_reports_both_shapes(tmp_path: Path) -> None:
    paths = build_corrupt_per_call_j_corpus(tmp_path)
    cols, starts = _cols(paths.base_path, paths.binary_name)
    report = validate_per_call_js(cols, starts)
    assert report.is_corrupt
    assert set(report.counts) == {
        CorruptionKind.OUT_OF_RANGE_J,
        CorruptionKind.VKEY_MISMATCH,
    }
    assert report.first_kind in report.counts
    # Samples carry locating triples for each present kind.
    for kind in report.counts:
        assert report.samples[kind]
        assert all(len(t) == 3 for t in report.samples[kind])


def test_report_to_dict_is_json_serialisable(tmp_path: Path) -> None:
    paths = build_corrupt_per_call_j_corpus(tmp_path)
    cols, starts = _cols(paths.base_path, paths.binary_name)
    report = validate_per_call_js(cols, starts)
    payload = report.to_dict()
    text = json.dumps(payload)  # must not raise
    back = json.loads(text)
    assert back["is_corrupt"] is True
    assert back["total_corrupt"] == report.total_corrupt
    assert set(back["counts"]) == {
        CorruptionKind.OUT_OF_RANGE_J.value,
        CorruptionKind.VKEY_MISMATCH.value,
    }


def test_require_clean_raises_on_corrupt_passes_on_clean(tmp_path: Path) -> None:
    """``require_clean`` is the validator's own opt-in RAISING gate.

    It raises :class:`CorruptCatalogError` (carrying the typed kind +
    counts + first-slot locator) on a corrupt catalog and returns
    silently on a clean one. Consumers never call it.
    """
    import pytest

    paths = build_corrupt_per_call_j_corpus(tmp_path)
    cols, starts = _cols(paths.base_path, paths.binary_name)
    report = validate_per_call_js(cols, starts)
    with pytest.raises(CorruptCatalogError) as exc:
        require_clean(cols, starts)
    assert exc.value.counts == report.counts
    assert exc.value.kind is report.first_kind
    assert exc.value.slot == report.samples[report.first_kind][0]

    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_callset(1, [("leaf",), ()]),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf", variants=_shared(2, 21), called=(),
        ),
    ]
    build_corpus(tmp_path, "clean", matched=specs)
    ccols, cstarts = _cols(tmp_path, "clean")
    require_clean(ccols, cstarts)  # must not raise


def test_report_samples_relocate_the_real_corrupt_slots(tmp_path: Path) -> None:
    """Each reported sample triple re-derives off the columns.

    Corruption validation runs ONCE, standalone -- the consumers carry NO
    corruption check (a concrete out-of-range ``J`` would IndexError in
    them, which is why the validator is meant to run BEFORE consumption).
    This pins the validator's slot locators: every ``(owning_var,
    called_idx, J)`` it reports must be the actual on-disk entry, so its
    enumeration can't silently mis-locate a slot.
    """
    paths = build_corrupt_per_call_j_corpus(tmp_path)
    cols, starts = _cols(paths.base_path, paths.binary_name)
    report = validate_per_call_js(cols, starts)
    assert report.is_corrupt
    for kind, triples in report.samples.items():
        for owning_var, called_idx, J in triples:
            p0 = int(cols.pce_offsets[owning_var])
            p1 = int(cols.pce_offsets[owning_var + 1])
            slot_js = {
                int(cols.pce_called_idx[i]): int(cols.pce_section_variant_index[i])
                for i in range(p0, p1)
            }
            assert slot_js[called_idx] == J


def test_cli_exit_codes(tmp_path: Path, capsys) -> None:
    # Clean corpus -> exit 0; corrupt corpus -> exit 1; --json prints JSON.
    specs = [
        MatchedFunctionSpec(
            func_name="root", variants=_callset(1, [("leaf",), ()]),
            called=("leaf",),
        ),
        MatchedFunctionSpec(
            func_name="leaf", variants=_shared(2, 21), called=(),
        ),
    ]
    build_corpus(tmp_path, "clean", matched=specs)
    assert main([str(tmp_path), "clean"]) == 0
    assert "OK" in capsys.readouterr().out

    paths = build_corrupt_per_call_j_corpus(tmp_path)
    assert main([str(paths.base_path), paths.binary_name]) == 1
    assert "CORRUPT" in capsys.readouterr().out

    rc = main([str(paths.base_path), paths.binary_name, "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["is_corrupt"] is True
