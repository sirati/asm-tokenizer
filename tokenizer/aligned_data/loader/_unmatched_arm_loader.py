"""Unmatched-arm loader (BIN catalog).

Single concern: assemble the unmatched ``SectionArm`` from
``<binary>_unmatched_index.bin`` (per-record data-bin locator, one
entry per unmatched ``_unmatched_data.bin`` record) and
``<binary>_sections.bin`` (the BIN catalog; unmatched sections are
emitted by ``write_unmatched_sections_pass2`` immediately after the
matched arm's sections in encounter order).

Records are self-describing in ``_unmatched_data.bin`` (the record
header carries ``token_count``), so the per-record index entry is a
bare offset and there is no length / sentinel / overlong shadow.

How the unmatched arm finds its sections in the shared BIN: the
builder always emits matched sections first, then unmatched. The
matched-arm locator (``matched_index.bin``) lists every matched
section's offset + length. The byte just past the last matched
section is the start of the unmatched region; from there the walker
streams sections via :func:`parse_section_bin` until EOF. An empty
matched arm starts the walk at the file-level prelude end.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tokenizer.aligned_data.matched_sections_columnar import (
    parse_sections_columnar,
)

from ._sections_bin_walk import (
    read_sections_bin_blob,
    resolve_func_name_or_raise,
    unmatched_region_start,
    walk_section_starts,
)


def _columnar_unmatched_sections(
    sections_bin: Path,
    region_start: int,
    line_to_name: Dict[int, str],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Decode every section in ``[region_start, EOF)`` columnarly.

    Returns ``(func_names, section_starts, section_variant_counts)``
    where ``section_starts[i]`` is the BIN offset of the i-th unmatched
    section, ``func_names[i]`` is its resolved function name, and
    ``section_variant_counts[i]`` is the per-section variant count
    (used by the loader to build the per-record -> per-section lookup
    table). Encounter order is preserved so the i-th entry of every
    array describes the same section.

    The unmatched region carries no locator, so the section starts are
    discovered by a boundary-only walk (:func:`walk_section_starts`,
    header + jump-table reads only); those starts then feed the single
    vectorized :func:`parse_sections_columnar` decoder -- the source of
    truth for the ``sections.bin`` wire format -- from which this arm
    needs only the section-level ``function_name_ptr`` (-> names) and
    ``n_variants`` (-> the per-record lookup). Mirrors the matched
    arm's columnar build rather than materialising one full
    :class:`Section` (call_target + variant-block + per-call tables)
    per unmatched function only to discard it.
    """
    func_names: List[str] = []
    if not sections_bin.exists():
        return (
            func_names,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    raw, blob = read_sections_bin_blob(sections_bin)
    section_starts = walk_section_starts(blob, region_start)
    cols = parse_sections_columnar(np.asarray(raw), section_starts)
    fids = cols.function_name_ptr
    func_names = [
        resolve_func_name_or_raise(
            int(fids[i]), line_to_name, sections_bin, int(section_starts[i])
        )
        for i in range(len(section_starts))
    ]
    return func_names, section_starts, cols.n_variants.astype(np.int64)


def _build_record_to_section_idx(
    variant_counts: np.ndarray, total_records: int
) -> np.ndarray:
    """Build the per-record -> per-section index mapping.

    Sections are emitted in encounter order; per-record offsets in
    ``starts`` follow the same order, with ``variant_counts[i]``
    consecutive records belonging to section ``i``. The mapping is
    derived once at arm-load (O(M)) so the session's per-record
    dispatch is ``arm.record_to_section_idx[idx]`` (O(1)) instead of
    an O(K) section re-walk per call.

    A cardinality mismatch (sections.bin claims a different total than
    unmatched_index.bin advertises) raises :class:`ValueError` with a
    migration-pointing message rather than silently writing past the
    end of the mapping; this is the single chokepoint that catches
    builder-side index/catalog drift.
    """
    mapping = np.zeros(total_records, dtype=np.uint32)
    cursor = 0
    for section_idx, count in enumerate(variant_counts):
        n = int(count)
        if cursor + n > total_records:
            raise ValueError(
                f"unmatched section[{section_idx}] declares {n} variants "
                f"but only {total_records - cursor} records remain on the "
                f"unmatched index; sections.bin / unmatched_index.bin are "
                f"out of sync, re-run memmap_builder to regenerate"
            )
        mapping[cursor:cursor + n] = section_idx
        cursor += n
    if cursor != total_records:
        raise ValueError(
            f"unmatched arm consumed {cursor} records across "
            f"{len(variant_counts)} sections but unmatched_index.bin has "
            f"{total_records} entries; sections.bin / unmatched_index.bin "
            f"are out of sync, re-run memmap_builder to regenerate"
        )
    return mapping


def load_unmatched_arm(
    paths,
    line_to_name: Dict[int, str],
    *,
    matched_index: Path,
):
    """Build the unmatched ``SectionArm`` from BIN walk + v1 data index.

    Empty (no unmatched functions) -> the orchestrator's canonical
    ``_empty_arm()``. ``paths.index_bin`` is the v1
    ``<binary>_unmatched_index.bin`` (per-record data-bin offsets);
    ``paths.sections_bin`` is the shared section catalog;
    ``matched_index`` locates the matched region so the unmatched
    walker knows where to start streaming sections.
    """
    # Local imports break the import cycle between the orchestrator
    # (``metadata_loader``) and this module.
    from .metadata_loader import (
        SectionArm,
        _empty_arm,
        build_length_lookup_tables,
        load_index_once,
        load_unmatched_lengths,
    )

    if not paths.index_bin.exists():
        return _empty_arm()
    starts = load_index_once(paths.index_bin)
    if starts is None:
        return _empty_arm()

    token_counts = load_unmatched_lengths(paths, starts)
    edge_indices, count_per_length = build_length_lookup_tables(
        token_counts, scale_factor=1
    )

    region_start = unmatched_region_start(matched_index)
    func_names, section_starts, variant_counts = _columnar_unmatched_sections(
        paths.sections_bin, region_start, line_to_name
    )
    # Per-record -> per-section lookup table. Without this, the
    # session's per-record dispatch would re-parse sections from index
    # 0 on every load_unmatched(idx) call -- O(K) per record, so
    # O(N*K) to walk the whole corpus instead of O(N).
    record_to_section_idx = _build_record_to_section_idx(
        variant_counts, total_records=len(starts)
    )
    return SectionArm(
        _starts=starts,
        edge_indices=edge_indices,
        count_per_length=count_per_length,
        func_names=func_names,
        section_starts=section_starts,
        record_to_section_idx=record_to_section_idx,
    )
