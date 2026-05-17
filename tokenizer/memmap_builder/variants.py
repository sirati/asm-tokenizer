"""Per-binary-group variant registry.

A ``VariantRegistry`` is the single authority over the
``vkey -> 0-based row index`` mapping inside one
``build_memmap_files`` invocation. It owns:

  * the index assignment (insertion order matches the iteration order
    the builder uses to consume ``versions``),
  * the on-disk hex-string encoding (``0x<lowercase-hex>``),
  * the per-group sidecar file ``<binary>_variants.csv``.

Section writers and the warn-log writer take an opaque
``variant_ref: str`` from this registry; they don't know about the
4-axis canonical tuple, the per-variant ``extra_metadata``, or the
``filename`` slot. That keeps the section-CSV schema flat: one
discriminator cell per row instead of repeating the full identity on
every row.

Design intent: the row index is the *only* cross-reference between
``<binary>_sections.csv`` / ``<binary>_unmatched_sections.csv`` /
``<binary>.warn.log`` rows and the sidecar metadata. Two variants
that share the canonical-4 axes (arch/compiler/version/opt) but
differ in flag_set / hardening / sanitizer / march therefore get
distinct rows and remain distinguishable in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class VariantRecord:
    """Sidecar-CSV row contents for one variant.

    ``flags`` is the canonical single-cell encoding of
    ``extra_metadata`` (sorted-key pipe-joined ``key=value`` pairs);
    empty string when the dict is empty.
    """

    filename: str
    arch: str
    compiler: str
    version: str
    opt: str
    pkg: str
    flags: str


def encode_flags(extra_metadata: Mapping[str, Any]) -> str:
    """Stable single-cell encoding of the opaque ``extra_metadata`` dict.

    Sorted by key so two equal dicts produce identical strings
    regardless of insertion order. Values are stringified via ``str()``
    — the upstream sidecar JSON already carries primitive scalars
    (strings, ints, bools); deeper structures are passed through
    verbatim and the encoding remains lossy-but-stable for those
    edge cases.
    """
    if not extra_metadata:
        return ""
    return "|".join(f"{k}={extra_metadata[k]}" for k in sorted(extra_metadata))


class VariantRegistry:
    """Assigns a stable row index to each variant and emits the sidecar.

    Construction takes the same ordered sequence of
    ``BinaryVersionInfo`` that ``build_memmap_files`` will iterate;
    every section/warn-log row that the builder writes for variant
    ``i`` carries the hex ref returned by ``ref(vkey_of_i)``.
    """

    def __init__(self, records: Iterable[VariantRecord], indices: Dict[Any, int]) -> None:
        self._records: List[VariantRecord] = list(records)
        self._indices = indices

    @classmethod
    def from_versions(cls, versions: Iterable[Any]) -> "VariantRegistry":
        """Build a registry from a ``List[BinaryVersionInfo]``.

        Each version's row index is its position in the input
        iteration; the ``vkey`` derived from the version's canonical-4
        + ``variant_id`` is the lookup key. Duplicate vkeys collapse to
        the first occurrence — the builder's pairing logic already
        de-duplicates by ``vkey`` before reaching here, so any later
        duplicate represents a bug upstream.
        """
        from .builder import VersionKey

        records: List[VariantRecord] = []
        indices: Dict[VersionKey, int] = {}
        for version in versions:
            vkey = VersionKey(
                arch=version.arch,
                compiler=version.compiler,
                compilerversion=version.compilerversion,
                opt=version.opt,
                variant_id=version.variant_id,
            )
            if vkey in indices:
                continue
            indices[vkey] = len(records)
            records.append(
                VariantRecord(
                    filename=getattr(version, "filename", "") or "",
                    arch=version.arch,
                    compiler=version.compiler,
                    version=version.compilerversion,
                    opt=version.opt,
                    pkg=version.pkg,
                    flags=encode_flags(version.extra_metadata),
                )
            )
        return cls(records, indices)

    @classmethod
    def from_vkeys(
        cls,
        vkeys: Iterable[Any],
        *,
        filename: str = "",
        pkg: str = "",
    ) -> "VariantRegistry":
        """Build a registry directly from an ordered list of vkeys.

        The legacy ``aligned_data.export`` entry-point lacks the per-
        variant filename and metadata that the runner-driven path
        carries; it only knows the canonical-4 axes. This factory lets
        that path still emit a well-formed ``_variants.csv`` (with
        empty ``flags`` and a caller-supplied ``filename`` / ``pkg``)
        without forcing the export module to synthesise full
        ``BinaryVersionInfo`` records.
        """
        records: List[VariantRecord] = []
        indices: Dict[Any, int] = {}
        for vkey in vkeys:
            if vkey in indices:
                continue
            indices[vkey] = len(records)
            records.append(
                VariantRecord(
                    filename=filename,
                    arch=vkey.arch,
                    compiler=vkey.compiler,
                    version=vkey.compilerversion,
                    opt=vkey.opt,
                    pkg=pkg,
                    flags="",
                )
            )
        return cls(records, indices)

    def ref(self, vkey: Any) -> str:
        """Return the row index of ``vkey`` as bare lowercase hex.

        Raises ``KeyError`` if the vkey was never registered — that's
        a programming error (every vkey the builder uses must come
        from the same ``versions`` list the registry was built from).
        """
        return f"{self._indices[vkey]:x}"

    def write_sidecar(self, output_dir: Path, binary_name: str) -> Path:
        """Emit ``<binary>_variants.csv`` and return the path written.

        Format: a header row (``filename,arch,compiler,version,opt,pkg,flags``)
        followed by one row per variant in registration order. The
        header makes the file self-describing for downstream consumers
        without forcing them to consult schema docs.
        """
        import csv

        path = output_dir / f"{binary_name}_variants.csv"
        with open(path, "w", newline="", encoding="ascii") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "arch", "compiler", "version", "opt", "pkg", "flags"])
            for record in self._records:
                writer.writerow(
                    [
                        record.filename,
                        record.arch,
                        record.compiler,
                        record.version,
                        record.opt,
                        record.pkg,
                        record.flags,
                    ]
                )
        return path


def write_warn_log_entry(warn_log, func_name: str, variant_ref: str, called_func: str) -> None:
    """Single chokepoint for the warn-log row format.

    Centralising the write here keeps the format change (4-axis
    cluster -> single ``0x<hex>`` ref) from leaking back into the
    helpers / writers / passes call sites: each of those just needs
    a ``variant_ref`` string and the called-function name.
    """
    warn_log.write(f"{func_name},{variant_ref},{called_func}\n")
