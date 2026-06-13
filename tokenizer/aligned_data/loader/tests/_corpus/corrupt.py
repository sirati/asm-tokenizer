"""Deliberately-corrupt catalog builder: per-call ``J`` corruption.

Single concern: lay down a corpus whose on-disk ``_sections.bin`` carries
per-call ``section_variant_index`` (``J``) values the standalone
:mod:`tokenizer.aligned_data.catalog_validate` validator must flag --
both corruption shapes it recognises:

* OUT_OF_RANGE_J  -- a concrete ``J`` ``>=`` the callee section's
  variant-table size (cannot address a real callee variant);
* VKEY_MISMATCH   -- an in-range ``J`` whose callee variant carries a
  ``variant_ref_offset`` (vkey) different from the owning caller
  variant's vkey (the per-call entry points at the wrong sibling).

Mechanism: build a CLEAN corpus through the production pass-2 writers,
then byte-patch the two target per-call ``J`` fields directly in the
``_sections.bin`` blob. The on-disk per-call entry is
``PER_CALL_ENTRY_SIZE`` bytes -- ``called_idx`` (u16 LE) at byte 0,
``section_variant_index`` / ``J`` (u16 LE) at byte 2 -- exactly as
:func:`...matched_sections_columnar.parse_sections_columnar` reads it
(``pce_section_variant_index = _u16(b, pce_addr + 2)``). Patching those
two bytes is faithful on-disk corruption with NO writer internals and no
dependency on any normalize/guard pass.

The build is self-validating: it re-parses the patched bin and raises
unless the catalog still frames cleanly AND the ONLY anomalies are the
two injected shapes (so a layout change can never silently turn the
validator tests into no-ops, and the test exercises detection rather
than a parse error).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.csv_section_index import (
    read_csv_section_index_arrays,
)
from tokenizer.aligned_data.matched_sections_bin import (
    CALL_TARGET_ENTRY_SIZE,
    MISSING_VARIANT_INDEX,
    PER_CALL_ENTRY_SIZE,
    SECTION_HEADER_SIZE,
    VARIANT_HEADER_SIZE,
)
from tokenizer.aligned_data.matched_sections_columnar import (
    _csr,
    _flat_member_addresses,
    _u16,
    parse_sections_columnar,
)

from .builder import CorpusPaths, build_corpus
from .specs import MatchedFunctionSpec, make_simple_variant

#: Byte position of the ``J`` (``section_variant_index``) u16 within each
#: per-call entry -- mirrors the columnar parser's ``pce_addr + 2``.
_J_FIELD_BYTE_OFFSET = 2


@dataclass(frozen=True)
class CorruptCorpus:
    """Paths to the corrupt corpus the validator tests consume."""

    base_path: Path
    binary_name: str


def _mk(vkey: tuple, seed: int) -> "object":
    return make_simple_variant(vkey, token_seed=seed, n_tokens=6)


def _patch_u16(blob: np.ndarray, byte_addr: int, value: int) -> None:
    """Overwrite the little-endian u16 at ``byte_addr`` in ``blob``."""
    blob[byte_addr] = value & 0xFF
    blob[byte_addr + 1] = (value >> 8) & 0xFF


def build_corrupt_per_call_j_corpus(
    tmp_path: Path, binary_name: str = "corruptbin"
) -> CorruptCorpus:
    """Build a corpus then corrupt two per-call ``J`` fields on disk.

    Matched-arm spec order (== ``matched_index.bin`` order)::

        0 caller (2 variants, vkeys ka / kb; both call target)
        1 target (2 variants, vkeys ta / tb)

    Caller variant 0's call slot is patched OUT OF RANGE (``J`` set to
    the callee's variant count); caller variant 1's call slot is patched
    to the WRONG in-range sibling (``J`` pointing at a target variant
    whose vkey differs from the caller variant's own vkey) -- the two
    shapes :func:`...catalog_validate.validate_per_call_js` reports.
    """
    base = tmp_path / "corrupt"
    base.mkdir(parents=True, exist_ok=True)

    ka, kb, ta, tb = ("k", "a"), ("k", "b"), ("k", "ta"), ("k", "tb")
    specs = (
        MatchedFunctionSpec(
            func_name="caller",
            variants=(_mk(ka, 1), _mk(kb, 2)),
            called=("target",),
        ),
        MatchedFunctionSpec(
            func_name="target",
            variants=(_mk(ta, 3), _mk(tb, 4)),
            called=(),
        ),
    )
    paths = build_corpus(base, binary_name, matched=specs)

    _corrupt_two_slots(paths)
    _assert_corruption_landed(paths)
    return CorruptCorpus(base_path=base, binary_name=binary_name)


def _pce_byte_addrs(blob: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Every per-call entry's byte address, in columnar order.

    Recomputes the per-call addresses with the layout owner's own
    arithmetic (the public columnar parser keeps them internal); this is
    a re-parse on demand off the blob, not a memoised parallel index.
    """
    b = blob
    offs = np.asarray(starts, dtype=np.int64).reshape(-1)
    n_sections = offs.size
    n_call_targets = _u16(b, offs + 4)
    n_variants = _u16(b, offs + 6)
    jt_base = offs + SECTION_HEADER_SIZE
    var_section, jt_addr = _flat_member_addresses(jt_base, n_variants, 2)
    var_n_calls = _u16(b, jt_addr)
    var_offsets = _csr(n_variants)
    jt_padded_bytes = ((n_variants + 1) // 2) * 4
    ct_base = jt_base + jt_padded_bytes
    region_base = ct_base + CALL_TARGET_ENTRY_SIZE * n_call_targets
    calls_excl = _csr(var_n_calls)[:-1]
    section_first_excl = np.zeros(n_sections, dtype=np.int64)
    has_vars = n_variants > 0
    section_first_excl[has_vars] = calls_excl[var_offsets[:-1][has_vars]]
    total_vars = int(var_offsets[-1])
    within_var = (
        np.arange(total_vars, dtype=np.int64) - var_offsets[var_section]
    )
    vb_addr = (
        region_base[var_section]
        + VARIANT_HEADER_SIZE * within_var
        + PER_CALL_ENTRY_SIZE * (calls_excl - section_first_excl[var_section])
    )
    pce_base = vb_addr + VARIANT_HEADER_SIZE
    _pce_var, pce_addr = _flat_member_addresses(
        pce_base, var_n_calls, PER_CALL_ENTRY_SIZE
    )
    return pce_addr


def _corrupt_two_slots(paths: CorpusPaths) -> None:
    """Patch caller v0's slot OOR and caller v1's slot to the wrong vkey."""
    starts, lengths = read_csv_section_index_arrays(
        paths.base_path / f"{paths.binary_name}_index.bin"
    )
    blob = np.fromfile(paths.sections_bin, dtype=np.uint8)
    cols = parse_sections_columnar(blob, starts, lengths)
    pce_addr = _pce_byte_addrs(blob, starts)

    # caller is matched section 0 (global vars 0, 1); target is section 1.
    v0 = int(cols.var_offsets[0])      # caller variant 0 (vkey ka)
    v1 = v0 + 1                        # caller variant 1 (vkey kb)
    t0 = int(cols.var_offsets[1])      # target variant 0 (vkey ta)
    n_target = int(cols.n_variants[1])

    # Each caller variant has exactly one call slot in this fixture.
    p_v0 = int(cols.pce_offsets[v0])
    p_v1 = int(cols.pce_offsets[v1])
    assert int(cols.pce_offsets[v0 + 1]) - p_v0 == 1
    assert int(cols.pce_offsets[v1 + 1]) - p_v1 == 1

    # OUT_OF_RANGE_J: J == n_target is the first index past the table and
    # stays below MISSING_VARIANT_INDEX (a concrete, addressable-looking
    # value the validator flags as out of range).
    assert n_target < MISSING_VARIANT_INDEX
    _patch_u16(blob, int(pce_addr[p_v0]) + _J_FIELD_BYTE_OFFSET, n_target)

    # VKEY_MISMATCH: caller v1 (vkey kb) -> target variant 0 (vkey ta),
    # an in-range index whose callee vkey differs from the caller's own.
    assert int(cols.var_ref_offset[t0]) != int(cols.var_ref_offset[v1])
    _patch_u16(blob, int(pce_addr[p_v1]) + _J_FIELD_BYTE_OFFSET, 0)

    blob.tofile(paths.sections_bin)


def _assert_corruption_landed(paths: CorpusPaths) -> None:
    """Raise unless the patched bin frames cleanly and carries EXACTLY the
    two injected corruption shapes (nothing incidental)."""
    from tokenizer.aligned_data.catalog_validate import (
        CorruptionKind,
        validate_per_call_js,
    )

    starts, lengths = read_csv_section_index_arrays(
        paths.base_path / f"{paths.binary_name}_index.bin"
    )
    blob = np.fromfile(paths.sections_bin, dtype=np.uint8)
    # Framing must survive the patch: a per-call-field overwrite does not
    # move any offset, so the section-length cross-check in the parser
    # (passed ``lengths``) must still hold -- a raise here means the patch
    # corrupted framing, not just a datum.
    cols = parse_sections_columnar(blob, starts, lengths)

    report = validate_per_call_js(cols, starts)
    if set(report.counts) != {
        CorruptionKind.OUT_OF_RANGE_J,
        CorruptionKind.VKEY_MISMATCH,
    }:
        raise AssertionError(
            "corrupt fixture self-check: expected EXACTLY OUT_OF_RANGE_J + "
            f"VKEY_MISMATCH, got {set(report.counts)!r} "
            "(layout or writer behaviour changed?)"
        )
    if report.counts[CorruptionKind.OUT_OF_RANGE_J] != 1:
        raise AssertionError(
            "corrupt fixture self-check: expected exactly one OOR-J slot, "
            f"got {report.counts[CorruptionKind.OUT_OF_RANGE_J]}"
        )
    if report.counts[CorruptionKind.VKEY_MISMATCH] != 1:
        raise AssertionError(
            "corrupt fixture self-check: expected exactly one VKEY_MISMATCH "
            f"slot, got {report.counts[CorruptionKind.VKEY_MISMATCH]}"
        )
    if int(cols.pce_section_variant_index.max()) >= MISSING_VARIANT_INDEX:
        raise AssertionError(
            "corrupt fixture self-check: a patched J reached the "
            "MISSING_VARIANT_INDEX sentinel; corruption must stay concrete"
        )
