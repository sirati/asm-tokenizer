"""Standalone per-call ``J`` corruption validator for section catalogs.

Single concern: scan a columnar section catalog's per-call entries for
the two corruption shapes (:class:`CorruptionKind`) and report them as
a typed :class:`CorruptionReport` -- THE single corruption definition.
It runs ONCE, standalone (the CLI / one-time pass via
:func:`require_clean`); the dataloader and the indexer carry NO
corruption check -- they handle only the legitimate
``MISSING_VARIANT_INDEX`` (``0xFFFE``) sentinel, so corruption is meant
to be caught HERE, before consumption.

The two shapes (a well-formed builder never emits either; a catalog
carrying one is corrupt and must be rebuilt):

* ``OUT_OF_RANGE_J`` -- a concrete ``J >= n_callee_variants``;
* ``VKEY_MISMATCH`` -- an in-range ``J`` pointing at a callee variant
  whose vkey differs from the entry's owning caller-variant vkey.

``MISSING_VARIANT_INDEX`` (``0xFFFE``) is a legitimate sentinel, not
corruption -- it is reported separately as a benign count.

Run as a tool: ``python -m tokenizer.aligned_data.catalog_validate <dir> [name]
[--json]`` -- decodes the matched arm's ``sections.bin`` + ``index.bin``
for the binary, validates, prints a summary (or JSON), and exits
nonzero when corruption is present.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from dedup_hashmap import HashMapU32U32

from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.aligned_data.matched_sections_columnar import ColumnarSections


__all__ = [
    "CorruptionKind",
    "CorruptCatalogError",
    "CorruptionReport",
    "validate_per_call_js",
    "require_clean",
    "main",
]


class CorruptionKind(enum.Enum):
    """The shape of a corrupt per-call ``J`` the validator flags.

    A well-formed builder never emits either shape; a catalog carrying
    one is corrupt and must be REBUILT. Distinct from
    :data:`...matched_sections_bin.MISSING_VARIANT_INDEX` (``0xFFFE``),
    which is a legitimate cross-arm vkey-mismatch SENTINEL, not
    corruption.
    """

    #: A concrete ``J`` the callee's variant table cannot address
    #: (``J >= n_callee_variants``).
    OUT_OF_RANGE_J = "out_of_range_j"

    #: An in-range ``J`` pointing at a callee variant whose vkey differs
    #: from the entry's owning caller-variant vkey (same-FID sibling
    #: skew -- the J addresses the wrong variant).
    VKEY_MISMATCH = "vkey_mismatch"


class CorruptCatalogError(ValueError):
    """A section catalog carries per-call data the validator rejects.

    Raised ONLY by the validator's opt-in :func:`require_clean` (the
    one-time standalone pass / CLI) -- never by the dataloader or the
    indexer, which carry no corruption check and handle only the
    legitimate ``MISSING_VARIANT_INDEX`` sentinel. Carries a
    :attr:`kind` (the dominant / first-encountered
    shape), :attr:`counts` (per-:class:`CorruptionKind` occurrence
    counts), and :attr:`slot` (the first offending
    ``(owning_variant, called_idx, J)`` locator) so a catcher branches on
    the typed discriminator, never on the message. Subclasses
    :class:`ValueError` so callers with an existing ``except ValueError``
    keep catching it.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: CorruptionKind,
        counts: Optional[Mapping[CorruptionKind, int]] = None,
        slot: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.counts = dict(counts) if counts is not None else {kind: 1}
        self.slot = slot


_U32_MISS = np.uint32(0xFFFFFFFF)
_SAMPLE_CAP = 10


@dataclass(frozen=True)
class CorruptionReport:
    """Typed result of a per-call ``J`` validation scan.

    ``counts`` maps each present :class:`CorruptionKind` to its
    occurrence count; ``samples`` carries up to :data:`_SAMPLE_CAP`
    ``(owning_variant, called_idx, J)`` triples per kind for locating the
    offending slots. ``missing_count`` is the benign
    ``MISSING_VARIANT_INDEX`` sentinel population (NOT corruption).
    """

    counts: Dict[CorruptionKind, int] = field(default_factory=dict)
    samples: Dict[CorruptionKind, List[Tuple[int, int, int]]] = field(
        default_factory=dict
    )
    missing_count: int = 0
    first_kind: "CorruptionKind | None" = None
    """The kind of the FIRST corrupt per-call entry in scan order
    (``None`` when not corrupt) -- the kind :func:`require_clean` reports
    as the dominant shape on its raised error."""

    @property
    def is_corrupt(self) -> bool:
        """True iff any corruption shape occurred."""
        return bool(self.counts)

    @property
    def total_corrupt(self) -> int:
        return int(sum(self.counts.values()))

    def to_dict(self) -> dict:
        """JSON-serialisable view (enum keys -> their string values)."""
        return {
            "is_corrupt": self.is_corrupt,
            "total_corrupt": self.total_corrupt,
            "missing_count": self.missing_count,
            "counts": {k.value: v for k, v in self.counts.items()},
            "samples": {
                k.value: [list(t) for t in v] for k, v in self.samples.items()
            },
        }


def validate_per_call_js(
    cols: ColumnarSections, section_offsets: np.ndarray
) -> CorruptionReport:
    """Scan every per-call entry; return a typed :class:`CorruptionReport`.

    Pure -- no IO, no logging, no raising. Vectorized over the whole
    per-call-entry column (bounded by the catalog's on-disk table sizes).
    """
    n_sections = int(cols.n_variants.size)
    offs = np.asarray(section_offsets, dtype=np.int64).reshape(-1)
    total_pce = int(cols.pce_offsets[-1])
    if total_pce == 0:
        return CorruptionReport()

    sec_map = HashMapU32U32(capacity=max(8, n_sections * 2))
    sec_map.insert_ndarray(
        offs.astype(np.uint32), np.arange(n_sections, dtype=np.uint32)
    )

    sec_of_var = np.repeat(
        np.arange(n_sections, dtype=np.int64), cols.n_variants
    )
    sec_of_pce_var = np.repeat(sec_of_var, cols.var_n_calls)
    owning_var = np.repeat(
        np.arange(cols.var_n_calls.size, dtype=np.int64), cols.var_n_calls
    )
    slot = cols.ct_offsets[sec_of_pce_var] + cols.pce_called_idx.astype(
        np.int64
    )
    ptr = cols.ct_function_section_ptr[slot]
    callee_sec = sec_map.lookup_ndarray(ptr).astype(np.int64)
    known = callee_sec != int(_U32_MISS)

    J = cols.pce_section_variant_index.astype(np.int64)
    missing_count = int((cols.pce_section_variant_index == MISSING_VARIANT_INDEX).sum())
    concrete = (J != MISSING_VARIANT_INDEX) & known
    n_callee = np.zeros(J.size, dtype=np.int64)
    n_callee[known] = cols.n_variants[callee_sec[known]]

    oor = concrete & (J >= n_callee)
    in_range = concrete & (J < n_callee)
    callee_node = np.zeros(J.size, dtype=np.int64)
    callee_node[in_range] = (
        cols.var_offsets[:-1][callee_sec[in_range]] + J[in_range]
    )
    vkey_mismatch = in_range.copy()
    vkey_mismatch[in_range] = (
        cols.var_ref_offset[callee_node[in_range]]
        != cols.var_ref_offset[owning_var[in_range]]
    )

    counts: Dict[CorruptionKind, int] = {}
    samples: Dict[CorruptionKind, List[Tuple[int, int, int]]] = {}
    first_pos: Dict[CorruptionKind, int] = {}
    for kind, mask in (
        (CorruptionKind.OUT_OF_RANGE_J, oor),
        (CorruptionKind.VKEY_MISMATCH, vkey_mismatch),
    ):
        n = int(mask.sum())
        if not n:
            continue
        counts[kind] = n
        idxs = np.nonzero(mask)[0]
        first_pos[kind] = int(idxs[0])
        samples[kind] = [
            (
                int(owning_var[i]),
                int(cols.pce_called_idx[i]),
                int(J[i]),
            )
            for i in idxs[:_SAMPLE_CAP].tolist()
        ]
    first_kind = (
        min(first_pos, key=first_pos.get) if first_pos else None
    )
    return CorruptionReport(
        counts=counts,
        samples=samples,
        missing_count=missing_count,
        first_kind=first_kind,
    )


def require_clean(
    cols: ColumnarSections, section_offsets: np.ndarray
) -> None:
    """Raise :class:`CorruptCatalogError` iff the catalog is corrupt.

    The validator's own opt-in RAISING gate -- used by the one-time
    standalone pass / CLI to refuse a corrupt catalog. Consumers
    (dataloader, indexer) do NOT call this; they carry no corruption
    check and handle only the legitimate ``MISSING_VARIANT_INDEX``
    sentinel. Clean catalogs return without raising.
    """
    report = validate_per_call_js(cols, section_offsets)
    if not report.is_corrupt:
        return
    kind = report.first_kind
    slot = report.samples.get(kind, [None])[0] if kind is not None else None
    raise CorruptCatalogError(
        f"corrupt catalog: {report.total_corrupt} per-call Js "
        f"({', '.join(f'{k.value}={v}' for k, v in report.counts.items())})"
        f"; {report.missing_count} benign MISSING.",
        kind=kind,
        counts=report.counts,
        slot=slot,
    )


def main(argv: "List[str] | None" = None) -> int:
    """CLI: validate a binary's matched-arm catalog; exit nonzero if
    corrupt. ``--json`` prints the machine-readable report."""
    import argparse
    from pathlib import Path

    from tokenizer.aligned_data.csv_section_index import (
        read_csv_section_index_arrays,
    )
    from tokenizer.aligned_data.matched_sections_columnar import (
        parse_sections_columnar,
    )

    parser = argparse.ArgumentParser(
        prog="tokenizer.aligned_data.catalog_validate",
        description="Validate a section catalog's per-call Js for corruption.",
    )
    parser.add_argument("directory", help="per-binary memmap output dir")
    parser.add_argument("name", help="binary name (the <name>_*.bin prefix)")
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON"
    )
    args = parser.parse_args(argv)

    base = Path(args.directory)
    starts, lengths = read_csv_section_index_arrays(
        base / f"{args.name}_index.bin"
    )
    blob = np.fromfile(base / f"{args.name}_sections.bin", dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, lengths)
    report = validate_per_call_js(cols, starts)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif report.is_corrupt:
        print(
            f"CORRUPT: {report.total_corrupt} corrupt per-call Js "
            f"({', '.join(f'{k.value}={v}' for k, v in report.counts.items())})"
            f"; {report.missing_count} benign MISSING."
        )
    else:
        print(
            f"OK: no corruption ({report.missing_count} benign MISSING "
            "sentinels)."
        )
    return 1 if report.is_corrupt else 0


if __name__ == "__main__":  # pragma: no cover -- CLI entry
    raise SystemExit(main())
