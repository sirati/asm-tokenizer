"""Hard-cutover smokes: legacy on-disk formats must raise with a
clear migration message rather than silently producing wrong arrays.

Pins the raise paths the post-restructuring loader / validator now
own:

* Stale / missing BIN prelude on ``<binary>_sections.bin`` →
  ``_matched_arm_loader`` raises (the prelude assertion at the BIN
  walker's entry point).
* Missing ``<binary>_function_names.txt`` → ``BinaryDataset`` raises
  (the sidecar reader's ``ValueError`` is rethrown from the ctor).
* Bad sidecar prelude → ``load_function_names`` raises.
* Legacy 6-cell unmatched section row → validator's
  ``build_unmatched_index_lookup`` raises (the validator still
  cross-checks the CSV catalog).

The raise paths exist in the production code already; these tests
keep the migration messages from silently becoming stale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.function_names_loader import (
    load_function_names,
)
from tokenizer.aligned_data.loader.metadata_loader import (
    BinaryArmPaths,
    SectionKind,
    load_section_arm,
)
from tokenizer.memmap_validation._unmatched_lookup import (
    build_unmatched_index_lookup,
)

from ._corpus import build_corpus, matched_spec, unmatched_spec


def _matched_paths(corpus) -> BinaryArmPaths:
    return BinaryArmPaths(
        sections_csv=corpus.matched_sections_csv,
        sections_bin=corpus.sections_bin,
        index_bin=corpus.matched_index_bin,
        data_bin=corpus.matched_data_bin,
    )


def _line_to_name(corpus):
    _, line_to_name = load_function_names(corpus.function_names_sidecar)
    return line_to_name


def test_stale_sections_bin_prelude_raises(tmp_path: Path) -> None:
    """Overwrite the BIN's prelude with the wrong magic and confirm the
    matched-arm walker raises with a migration message at open time
    rather than silently producing wrong sections.
    """
    corpus = build_corpus(tmp_path, "bin", matched=[matched_spec("zeta_fn")])
    raw = corpus.sections_bin.read_bytes()
    # Stamp a wrong magic in bytes 0..3; rest of the file stays intact.
    bad = b"BAD!" + raw[4:]
    corpus.sections_bin.write_bytes(bad)

    with pytest.raises(ValueError, match="magic"):
        load_section_arm(
            SectionKind.MATCHED, _matched_paths(corpus), _line_to_name(corpus),
            matched_index=corpus.matched_index_bin,
        )


def test_missing_function_names_sidecar_raises(tmp_path: Path) -> None:
    """BinaryDataset construction reads the sidecar in __init__; deleting
    the sidecar after the build surfaces as ValueError at open time
    rather than a confusing KeyError later in iteration.
    """
    corpus = build_corpus(
        tmp_path, "bin",
        matched=[matched_spec("zeta_fn")],
        unmatched=[unmatched_spec("alpha_fn")],
    )
    corpus.function_names_sidecar.unlink()
    with pytest.raises(ValueError, match="re-run memmap_builder"):
        BinaryDataset(corpus.base_path, corpus.binary_name)


def test_bad_sidecar_prelude_raises(tmp_path: Path) -> None:
    """Sidecar with a wrong-version prelude must raise with the
    migration-pointing message; otherwise a future version mismatch
    would silently truncate the name table.
    """
    corpus = build_corpus(tmp_path, "bin", matched=[matched_spec("zeta_fn")])
    sidecar = corpus.function_names_sidecar
    text = sidecar.read_text("utf-8")
    # Replace the # format=N line with a wrong version.
    body = text.split("\n", 1)[1]
    sidecar.write_text("# format=99\n" + body, encoding="utf-8")

    with pytest.raises(ValueError, match="re-run memmap_builder"):
        load_function_names(sidecar)


def test_legacy_6_cell_unmatched_row_raises(tmp_path: Path) -> None:
    """Validator's unmatched-row walker must reject the pre-restructuring
    6-cell row rather than silently producing an empty index lookup
    (which would surface as a wave of csv-only-unmatched mis-flags).
    """
    corpus = build_corpus(
        tmp_path, "bin", unmatched=[unmatched_spec("alpha_fn")]
    )
    # Take the first variant row of the unmatched sections CSV and pad
    # it to 6 cells so it looks like the legacy layout.
    text = corpus.unmatched_sections_csv.read_text("utf-8")
    lines = text.splitlines(keepends=True)
    row_idx = next(
        i for i, line in enumerate(lines)
        if i >= 1 and line.strip() and line.count(",") == 4
    )
    cells = lines[row_idx].rstrip("\r\n").split(",")
    assert len(cells) == 5, f"sanity: expected 5-cell row, got {cells!r}"
    legacy = ",".join(cells + ["00000004"])
    lines[row_idx] = legacy + "\n"
    corpus.unmatched_sections_csv.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match="re-run memmap_builder"):
        build_unmatched_index_lookup(
            corpus.unmatched_sections_csv,
            corpus.base_path / f"{corpus.binary_name}_variants.csv",
            version_keys=[],
            line_to_name=_line_to_name(corpus),
        )
