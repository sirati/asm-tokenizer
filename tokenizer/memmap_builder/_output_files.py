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

Per-binary BIN catalog: :func:`open_sections_bin_outputs` opens the
``<binary>_sections.bin`` writer (a :class:`SectionWriter`) AND a
fresh :class:`ExternProviderRegistry`. Both per-binary artefacts —
the BIN itself and the sidecar — share this lifecycle so a single
helper is the chokepoint for opening AND closing them. The BIN
writer's :meth:`SectionWriter.finalize` runs the back-patch +
sentinel-sweep checks; the sidecar is serialised to disk via
:meth:`ExternProviderRegistry.write_sidecar`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tokenizer.aligned_data.csv_format import write_csv_prelude
from tokenizer.aligned_data.extern_providers import ExternProviderRegistry
from tokenizer.aligned_data.index_format import write_index_prelude
from tokenizer.aligned_data.matched_sections_bin import SectionWriter

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


@dataclass
class SectionsBinOutputs:
    """Per-binary BIN-catalog handles: writer + extern-provider registry.

    Owns the :class:`SectionWriter` for ``<binary>_sections.bin`` and
    a fresh :class:`ExternProviderRegistry` that the matched + unmatched
    pass-2 walkers both share. The BIN holds matched + unmatched
    sections (per Phase 3 layout decision), so a SINGLE writer is
    threaded through both arms — there is no per-arm BIN.

    ``finalize`` runs the writer's structural checks + closes the
    underlying mmap AND writes the ``<binary>_extern_providers.txt``
    sidecar in encounter order. Idempotent on the writer (a second
    call is a no-op via :meth:`SectionWriter.close`), but the
    sidecar is rewritten — that's still correct because
    :meth:`ExternProviderRegistry.write_sidecar` is itself idempotent
    on the registry state.
    """

    section_writer: SectionWriter
    extern_providers: ExternProviderRegistry
    sections_bin_path: Path
    output_dir: Path
    binary_name: str

    def finalize(self) -> None:
        """Finalize the BIN + write the extern-provider sidecar.

        Order matters: the BIN's :meth:`SectionWriter.finalize` runs
        the structural ``pending_holes`` + ``0xFFFF`` sweep assertions
        first; if either trips, the sidecar is not written and the
        exception surfaces. The sidecar is a TEXT artefact whose
        content is fully known from the in-memory registry, so it can
        be re-emitted on retry without coordinating with the BIN.
        """
        self.section_writer.finalize()
        sidecar_path = self.extern_providers.write_sidecar(
            self.output_dir, self.binary_name
        )
        logger.info(f"  Wrote: {sidecar_path}")

    def close(self) -> None:
        """Always-runs cleanup: drop the BIN mmap without running checks.

        Mirrors :meth:`SectionWriter.close`. Used as an ExitStack
        callback so an exception mid-build still releases the mmap.
        """
        self.section_writer.close()


def open_sections_bin_outputs(
    output_dir: Path, binary_name: str
) -> SectionsBinOutputs:
    """Open ``<binary>_sections.bin`` + a fresh extern-provider registry.

    The BIN's 16-byte ``MSEC`` prelude is stamped lazily by
    :class:`SectionWriter`'s constructor; the sidecar is text and is
    not opened until :meth:`SectionsBinOutputs.finalize` runs, so the
    only on-disk side effect of this opener is creating the BIN
    mapping (which :class:`SectionWriter` truncates / unmaps on
    :meth:`close` if the build fails mid-flight).
    """
    sections_bin_path = output_dir / f"{binary_name}_sections.bin"
    logger.info(f"  Creating: {sections_bin_path}")
    return SectionsBinOutputs(
        section_writer=SectionWriter(sections_bin_path),
        extern_providers=ExternProviderRegistry(),
        sections_bin_path=sections_bin_path,
        output_dir=output_dir,
        binary_name=binary_name,
    )


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
