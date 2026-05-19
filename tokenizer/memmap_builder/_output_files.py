"""Per-group section + index output-file lifecycle helpers.

Each pass needs a sections CSV (with the ``# format=N`` prelude already
written), an ``_index.bin``, and a tidy ``close`` step. The two arms
diverge only on the index-file layout:

* **matched** -- ``matched_index.bin`` is the function-to-CSV-section
  locator in pre-v1 layout (no prelude, no alignment shift); CSV byte
  offsets are not 4-aligned so the v1 writer's assertion would trip
  on every entry. The dedicated opener leaves the file empty so the
  caller's first byte is the first entry.
* **unmatched** -- ``unmatched_index.bin`` keeps the v1 layout with the
  16-byte file-level prelude already written by the opener.

Splitting the openers (rather than gating one helper on a boolean
flag) keeps the prelude policy local to each arm and prevents a
caller from accidentally mixing v1 entries with the pre-v1 file shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.index_format import write_index_prelude

logger = logging.getLogger(__name__)


@dataclass
class SectionOutputs:
    """Open sections CSV + index bin handles for one pass.

    ``sections_file`` already carries the ``# format=N`` prelude line.
    The ``index_file`` prelude policy is opener-specific -- see
    :func:`open_matched_section_outputs` and
    :func:`open_unmatched_section_outputs`.
    """

    sections_file: "object"
    index_file: "object"
    sections_path: Path
    index_path: Path

    def close(self) -> None:
        self.sections_file.close()
        logger.info(f"  Closed: {self.sections_path}")
        self.index_file.close()
        logger.info(f"  Closed: {self.index_path}")


def _open_section_csv_and_index(output_dir: Path, prefix: str) -> SectionOutputs:
    """Open the section CSV + index bin handles + emit the CSV prelude.

    The index file is left empty -- arm-specific openers below add the
    index prelude (or skip it) to match their layout.
    """
    sections_path = output_dir / f"{prefix}_sections.csv"
    index_path = output_dir / f"{prefix}_index.bin"
    logger.info(f"  Creating: {sections_path}")
    logger.info(f"  Creating: {index_path}")
    sections_file = open(sections_path, "w", newline="", encoding="ascii")
    index_file = open(index_path, "wb")
    write_csv_prelude(sections_file)
    return SectionOutputs(
        sections_file=sections_file,
        index_file=index_file,
        sections_path=sections_path,
        index_path=index_path,
    )


def open_matched_section_outputs(
    output_dir: Path, prefix: str
) -> SectionOutputs:
    """Open matched-arm outputs; index file is pre-v1 layout (no prelude).

    ``matched_index.bin`` is the CSV-section locator written via
    :func:`tokenizer.aligned_data.csv_section_index.write_csv_section_index_entry`
    -- 8-byte entries with no file-level header. The reader infers the
    entry count from file size; no prelude bytes are consumed.
    """
    return _open_section_csv_and_index(output_dir, prefix)


def open_unmatched_section_outputs(
    output_dir: Path, prefix: str
) -> SectionOutputs:
    """Open unmatched-arm outputs; index file carries the v1 prelude.

    ``unmatched_index.bin`` stays on the v1 wire format (one entry per
    version's data-bin record, 4-byte aligned). The opener writes the
    16-byte ``IDX1`` prelude before returning so the caller's first
    ``write_index_entry`` lands at the correct offset.
    """
    outputs = _open_section_csv_and_index(output_dir, prefix)
    write_index_prelude(outputs.index_file)
    return outputs
