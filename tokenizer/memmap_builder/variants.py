"""Per-binary-group variant registry.

A ``VariantRegistry`` is the single authority over the
``vkey -> byte_offset`` mapping inside one ``build_memmap_files``
invocation. It owns:

  * the on-disk record placement in ``<binary>_variants.bin`` (each
    variant's record is appended in registration order; the byte
    offset the writer assigns is the registry's canonical reference),
  * the hex-string encoding section CSVs cite as ``variant_ref``
    (bare lowercase hex of the byte offset; same shape every other
    byte-offset cell in the section CSV uses — see
    ``aligned_data/io.py``'s ``f"{data_offset:x}"`` precedent),
  * the slim back-reference CSV ``<binary>_variants.csv`` mapping
    ``filename -> offset`` for human / filename-recovery tooling.

Section writers and the warn-log writer take an opaque
``variant_ref: str`` from this registry; they don't know about the
bin layout or the unified vocab. The hex shape of the cell is
unchanged from the previous (row-index) regime — only the integer's
meaning flipped from "row index in verbose CSV" to "byte offset
into ``_variants.bin``", so existing section-CSV consumers parse
the cell the same way and dispatch to the new memmap path.

Design intent: the byte offset is the *only* cross-reference between
``<binary>_sections.csv`` / ``<binary>_unmatched_sections.csv`` /
``<binary>.warn.log`` rows and the per-variant token record in the
bin. The dataloader's hot path memmaps the bin once and slices at
``offset`` — no CSV consultation. The slim CSV is purely a
human-readable filename lookup; the dataloader never reads it on
the hot path.

Two variants that share the canonical-4 axes (arch/compiler/version/
opt) but differ in flag_set / hardening / sanitizer / march therefore
get distinct ``vkey``s (via ``variant_id``) and distinct bin records
and remain distinguishable.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tokenizer.token_manager import VocabularyManager
from tokenizer.variant_tokens.record import write_record


class VariantRegistry:
    """Assigns a stable byte offset to each variant and emits the bin + slim CSV.

    The registry hard-depends on the unified ``VocabularyManager`` —
    encoding each record requires looking up every axis-token string
    in the unified vocab. Passing the vocab in the constructor (rather
    than per call to ``write_sidecar``) makes that dependency explicit
    at construction time and matches the "registry owns the encoding
    concern" boundary.

    Construction takes the same ordered sequence of ``BinaryVersionInfo``
    that ``build_memmap_files`` will iterate. ``write_sidecar`` MUST
    run before any caller invokes ``ref(vkey)``; the orchestration in
    ``build_memmap_files`` already enforces this ordering (sidecar
    write precedes the matched / unmatched section passes).
    """

    def __init__(
        self,
        versions: Iterable[Any],
        unified_vocab: VocabularyManager,
    ) -> None:
        # Dedup vkeys in encounter order so two passes (registry build
        # + bin write) walk the variants in the same sequence and the
        # offsets land deterministically.
        self._unified_vocab = unified_vocab
        self._ordered_versions: List[Any] = []
        self._vkey_index: Dict[Any, int] = {}
        for version in versions:
            vkey = _vkey_for_version(version)
            if vkey in self._vkey_index:
                continue
            self._vkey_index[vkey] = len(self._ordered_versions)
            self._ordered_versions.append(version)
        # Populated by ``write_sidecar``. Empty until then; ``ref``
        # raises KeyError on lookup so a caller-ordering bug surfaces
        # loudly rather than emitting bogus 0-offset refs.
        self._offsets: Dict[Any, int] = {}

    @classmethod
    def from_versions(
        cls,
        versions: Iterable[Any],
        unified_vocab: VocabularyManager,
    ) -> "VariantRegistry":
        """Convenience factory — symmetric naming with the old API.

        Equivalent to calling the constructor directly; kept so
        ``build_memmap_files`` and tests retain the previously-typed
        ``VariantRegistry.from_versions(...)`` call shape.
        """
        return cls(versions, unified_vocab)

    def ref(self, vkey: Any) -> str:
        """Return ``vkey``'s byte offset into ``_variants.bin`` as hex.

        Format: bare lowercase hex (no ``0x`` prefix) to match the
        ``f"{data_offset:x}"`` convention every other byte-offset
        cell in the section CSV already uses (see
        ``aligned_data/io.py:write_function_section_csv`` and
        ``write_unmatched_section_csv``).

        Raises ``KeyError`` if ``write_sidecar`` has not yet run OR
        if the vkey was never registered. Both are programming
        errors — ``build_memmap_files`` calls ``write_sidecar`` before
        the section passes, and every vkey a section writer uses must
        come from the same ``versions`` list the registry was built
        from.
        """
        return f"{self._offsets[vkey]:x}"

    def write_sidecar(self, output_dir: Path, binary_name: str) -> Path:
        """Emit ``<binary>_variants.bin`` and slim ``<binary>_variants.csv``.

        The bin file holds one record per unique variant in
        registration order (uint16 little-endian via
        ``variant_tokens.record.write_record``). Each record's
        starting byte offset is captured into ``self._offsets`` so
        ``ref(vkey)`` can return it.

        The slim CSV is two columns — ``filename,offset`` — one row
        per variant in the same order. ``offset`` is bare lowercase
        hex, matching the section-CSV convention so a human can
        cross-reference cells without mental base conversion. Length
        is intentionally omitted: the bin record's first u16
        (``n_tokens``) self-describes the slice length, so a back-
        reference table need not duplicate it.

        Returns the bin path (the slim CSV path is derivable as a
        sibling). The bin is the primary artefact; the CSV is a
        back-reference.
        """
        bin_path = output_dir / f"{binary_name}_variants.bin"
        csv_path = output_dir / f"{binary_name}_variants.csv"

        # Single pass — for each unique variant in registration order,
        # write its record to the bin and capture the offset back into
        # the registry, then emit the matching slim CSV row.
        with open(bin_path, "wb") as bin_handle:
            for version in self._ordered_versions:
                vkey = _vkey_for_version(version)
                offset = write_record(bin_handle, version, self._unified_vocab)
                self._offsets[vkey] = offset

        with open(csv_path, "w", newline="", encoding="ascii") as csv_handle:
            writer = csv.writer(csv_handle)
            writer.writerow(["filename", "offset"])
            for version in self._ordered_versions:
                vkey = _vkey_for_version(version)
                filename = getattr(version, "filename", "") or ""
                writer.writerow([filename, f"{self._offsets[vkey]:x}"])

        return bin_path


def _vkey_for_version(version: Any) -> Any:
    """Build the ``VersionKey`` used for dedup + lookup.

    Imported lazily from ``builder`` to keep the registry's import
    graph one-way (builder imports variants; variants does not
    import builder at module load).
    """
    from .builder import VersionKey

    return VersionKey(
        arch=version.arch,
        compiler=version.compiler,
        compilerversion=version.compilerversion,
        opt=version.opt,
        variant_id=version.variant_id,
    )


def write_warn_log_entry(warn_log, func_name: str, variant_ref: str, called_func: str) -> None:
    """Single chokepoint for the warn-log row format.

    Centralising the write here keeps the format change (4-axis
    cluster -> single ``0x<hex>`` ref) from leaking back into the
    helpers / writers / passes call sites: each of those just needs
    a ``variant_ref`` string and the called-function name.
    """
    warn_log.write(f"{func_name},{variant_ref},{called_func}\n")
