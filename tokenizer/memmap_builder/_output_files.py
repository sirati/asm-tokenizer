"""Per-group section + index output-file lifecycle helpers.

Each (matched / unmatched) pass needs the same three things from the
output directory: a sections CSV (with the ``# format=N`` prelude
already written), an ``_index.bin`` (with the 16-byte file-level
prelude already written), and a tidy ``close`` step. Factoring the
file-open + prelude-write into :class:`SectionOutputs` keeps
``build_memmap_files`` focused on the pipeline shape rather than the
per-file boilerplate, and prevents the two passes from drifting in
which prelude bytes they emit.
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

    ``sections_file`` already carries the ``# format=N`` prelude line;
    ``index_file`` already carries the 16-byte file-level prelude.
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


def open_section_outputs(
    output_dir: Path, prefix: str
) -> SectionOutputs:
    """Open the section CSV + index bin for ``prefix`` and emit both preludes."""
    sections_path = output_dir / f"{prefix}_sections.csv"
    index_path = output_dir / f"{prefix}_index.bin"
    logger.info(f"  Creating: {sections_path}")
    logger.info(f"  Creating: {index_path}")
    sections_file = open(sections_path, "w", newline="", encoding="ascii")
    index_file = open(index_path, "wb")
    write_csv_prelude(sections_file)
    write_index_prelude(index_file)
    return SectionOutputs(
        sections_file=sections_file,
        index_file=index_file,
        sections_path=sections_path,
        index_path=index_path,
    )
