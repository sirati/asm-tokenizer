"""On-demand, section-bounded materialisation of the columnar catalog.

Single concern: present the SAME flat-column surface as
:class:`~.matched_sections_columnar.ColumnarSections`, but parse each
section's heavy payload (call_target table, variant headers, per-call
entries) only when a consumer first touches that section -- instead of
parsing the WHOLE ``sections.bin`` catalog at open. A vector_batch decode
samples <=5% of a binary's sections (BFS closure of the sampled roots),
so the eager full parse (~2.4 s on z3, ~1.9 M call-target / 40 M per-call
entries) is overwhelmingly dead work for the open path; this bounds it to
the touched set.

Boundary contract (the design-first sentence):

  *Given the ``sections.bin`` blob + the region's section offsets/lengths,
  expose a :class:`LazyColumnarSections` whose section-level columns +
  ALL CSR offset arrays are eager (cheap, one entry per section/variant)
  and whose heavy value columns are full-length numpy arrays filled
  per-section on first :meth:`~LazyColumnarSections.ensure_sections` --
  so every GLOBAL node / call-target / per-call-entry index a consumer
  uses still addresses the right slot, with no reparse of the bytes the
  catalog already owns.*

Why full-length (not compacted) backing: the flat NODE index space is
shared with the RLG3 geometry axes (body/id/value, one entry per catalog
node) and consumers index the heavy columns by GLOBAL node / slot / entry
positions (``cols.pce_offsets[parents]``, ``cols.ct_type[slot]``). Keeping
the arrays full-length and globally addressed makes the lazy catalog
DROP-IN for the eager one -- no consumer indexing changes -- while only
the touched sections' bytes are ever parsed.

Reuse (no hand-rolled parse): :meth:`~LazyColumnarSections.ensure_sections`
delegates the per-section value decode to the SAME
:func:`~.matched_sections_columnar.parse_sections_columnar` the eager path
uses, run over the touched sections' offsets/lengths, then scatters its
value columns into the global backing at the eager CSR slots. The decode
is byte-identical because every section's tables are addressed from its
own offset (self-contained on disk), so parsing a section in isolation
yields exactly the bytes the full-batch parse would have written there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np

from .matched_sections_bin import (
    CALL_TARGET_ENTRY_SIZE,
    MISSING_VARIANT_INDEX,
    PER_CALL_ENTRY_SIZE,
    SECTION_HEADER_SIZE,
    VARIANT_HEADER_SIZE,
    _SECTION_DUPLICATED_BIT,
    _SECTION_FID_MASK,
)
from .matched_sections_columnar import (
    ColumnarSections,
    _csr,
    _u16,
    _u32,
    parse_sections_columnar,
)


__all__ = ["LazyColumnarSections", "parse_sections_columnar_lazy"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Skeleton:
    """The eager, section-level catalog skeleton (one entry per section).

    Carries every column the columnar parse can compute WITHOUT reading
    the per-variant jump table or any heavy table -- the section headers
    plus the three CSR offset arrays. ``section_pce_base`` is each
    section's GLOBAL first-per-call-entry index, derived from the section
    byte LENGTH (matched arm) so the per-call-entry CSR is globally
    consistent the moment a section is filled, with no global jump-table
    read. See :func:`_build_skeleton` for the derivation + its limits.
    """

    function_name_ptr: np.ndarray  # u32[n_sections]
    is_duplicated: np.ndarray  # bool[n_sections]
    n_call_targets: np.ndarray  # i64[n_sections]
    n_variants: np.ndarray  # i64[n_sections]
    ct_offsets: np.ndarray  # i64[n_sections + 1]
    var_offsets: np.ndarray  # i64[n_sections + 1]
    section_total_calls: np.ndarray  # i64[n_sections]
    section_pce_base: np.ndarray  # i64[n_sections + 1] (CSR of total_calls)


def _build_skeleton(
    blob_u8: np.ndarray,
    section_offsets: np.ndarray,
    section_lengths: Optional[np.ndarray],
) -> _Skeleton:
    """Section-level columns + CSR offsets, WITHOUT the heavy parse.

    Reads only the per-section header (``<IHH`` = raw_fid, n_call_targets,
    n_variants), mirroring :func:`parse_sections_columnar`'s section-level
    decode (``:308-318``). ``section_total_calls`` (and thus the per-call
    -entry CSR base) is recovered from the section byte LENGTH by inverting
    the on-disk layout:

        end = offs + HEADER + jt_padded + CT_SIZE*n_ct
              + VH_SIZE*n_var + PCE_SIZE*total_calls   (already aligned)

    Every term is a multiple of :data:`SECTION_ALIGNMENT` (4) -- header 8,
    ``jt_padded`` is ``((n_var+1)//2)*4``, CT 12, VH 8, PCE 4 -- so the
    section end is ALWAYS already 4-aligned and the alignment pad is
    structurally zero. Hence ``total_calls = (length - fixed) / PCE_SIZE``
    is EXACT (no pad ambiguity), giving the global per-section base with
    pure section-level arithmetic instead of the ~14.8 M-wide jump-table
    gather. Requires ``section_lengths`` (the matched locator); the
    region-agnostic caller supplies it (the unmatched region's structural
    walk derives equivalent lengths upstream).
    """
    offs = np.asarray(section_offsets, dtype=np.int64).reshape(-1)
    n_sections = offs.size
    raw_fid = _u32(blob_u8, offs).astype(np.uint32)
    is_duplicated = (
        raw_fid & np.uint32(_SECTION_DUPLICATED_BIT)
    ).astype(bool)
    function_name_ptr = (raw_fid & np.uint32(_SECTION_FID_MASK)).astype(
        np.uint32
    )
    n_call_targets = _u16(blob_u8, offs + 4).astype(np.int64)
    n_variants = _u16(blob_u8, offs + 6).astype(np.int64)
    ct_offsets = _csr(n_call_targets)
    var_offsets = _csr(n_variants)

    if section_lengths is None:
        raise ValueError(
            "LazyColumnarSections requires per-section byte lengths to "
            "derive the per-call-entry CSR base without a global "
            "jump-table read; pass the region's section_lengths"
        )
    lens = np.asarray(section_lengths, dtype=np.int64).reshape(-1)
    if lens.size != n_sections:
        raise ValueError(
            f"section_lengths size {lens.size} does not match "
            f"section_offsets size {n_sections}"
        )
    jt_padded = ((n_variants + 1) // 2) * 4
    fixed = (
        SECTION_HEADER_SIZE
        + jt_padded
        + CALL_TARGET_ENTRY_SIZE * n_call_targets
        + VARIANT_HEADER_SIZE * n_variants
    )
    rem = lens - fixed
    bad = np.nonzero((rem < 0) | (rem % PER_CALL_ENTRY_SIZE != 0))[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            "sections.bin lazy skeleton: section at offset "
            f"{int(offs[i])} has length {int(lens[i])} inconsistent with "
            f"its header (fixed prefix {int(fixed[i])}); the catalog is "
            "corrupt or the locator lengths are stale"
        )
    section_total_calls = rem // PER_CALL_ENTRY_SIZE
    section_total_calls[n_variants == 0] = 0
    section_pce_base = _csr(section_total_calls)
    return _Skeleton(
        function_name_ptr=function_name_ptr,
        is_duplicated=is_duplicated,
        n_call_targets=n_call_targets,
        n_variants=n_variants,
        ct_offsets=ct_offsets,
        var_offsets=var_offsets,
        section_total_calls=section_total_calls,
        section_pce_base=section_pce_base,
    )


class LazyColumnarSections:
    """Drop-in :class:`ColumnarSections` whose heavy columns fill on touch.

    Exposes the SAME attribute names as :class:`ColumnarSections`. The
    section-level columns + every CSR offset array are eager; the heavy
    value columns (``ct_*``, ``var_ref_offset``, ``var_data_offset_
    shifted``, ``var_n_calls``, ``pce_*``) are full-length numpy arrays
    that :meth:`ensure_sections` fills per section on first touch. A
    consumer that only ever indexes touched sections (the vector_batch
    decode path) therefore pays the heavy parse for those sections alone.

    Construct via :func:`parse_sections_columnar_lazy`. Substitutable
    wherever a ``ColumnarSections`` is expected (the type hint stays
    ``ColumnarSections`` at the consumer boundary); the only added surface
    is :meth:`ensure_sections`, called by the small set of sites that
    introduce a new section set (the splice adjacency's frontier resolve
    and the batch's root seed).
    """

    def __init__(
        self,
        blob_u8: np.ndarray,
        section_offsets: np.ndarray,
        section_lengths: np.ndarray,
    ) -> None:
        self._blob = blob_u8
        self._offsets = np.asarray(section_offsets, dtype=np.int64).reshape(-1)
        self._lengths = np.asarray(section_lengths, dtype=np.int64).reshape(-1)
        skel = _build_skeleton(blob_u8, self._offsets, self._lengths)
        self._skel = skel
        n_sections = self._offsets.size
        total_cts = int(skel.ct_offsets[-1])
        total_vars = int(skel.var_offsets[-1])
        total_entries = int(skel.section_pce_base[-1])

        # --- eager section-level columns (exposed verbatim) ---------------
        self.function_name_ptr = skel.function_name_ptr
        self.is_duplicated = skel.is_duplicated
        self.n_call_targets = skel.n_call_targets
        self.n_variants = skel.n_variants
        self.ct_offsets = skel.ct_offsets
        self.var_offsets = skel.var_offsets

        # --- heavy value columns: full-length backing, lazily filled ------
        self.ct_function_name_ptr = np.zeros(total_cts, dtype=np.uint32)
        self.ct_function_section_ptr = np.zeros(total_cts, dtype=np.uint32)
        self.ct_type = np.zeros(total_cts, dtype=np.uint8)
        self.ct_is_matched = np.zeros(total_cts, dtype=bool)
        self.var_ref_offset = np.zeros(total_vars, dtype=np.uint32)
        self.var_data_offset_shifted = np.zeros(total_vars, dtype=np.uint32)
        self.var_n_calls = np.zeros(total_vars, dtype=np.int64)
        # pce_offsets is the GLOBAL per-call-entry CSR (one per variant + 1).
        # Each section's slice is filled from its eager section_pce_base +
        # the intra-section cumsum of var_n_calls, so the absolute entry
        # indices are correct the instant a section is filled. The boundary
        # entries (var_offsets[s]) are seeded eagerly to section_pce_base[s]
        # so cross-section reads (pce_offsets[v0]/[v1] in the fallback table,
        # pce_offsets[parents+1] in expand) are correct even before fill.
        self.pce_offsets = np.zeros(total_vars + 1, dtype=np.int64)
        self.pce_offsets[skel.var_offsets] = skel.section_pce_base
        self.pce_called_idx = np.zeros(total_entries, dtype=np.uint16)
        self.pce_section_variant_index = np.zeros(
            total_entries, dtype=np.uint16
        )

        self._filled = np.zeros(n_sections, dtype=bool)
        # Touched-bounded running tally of MISSING_VARIANT_INDEX per-call
        # entries (the dropped-splice-edge data-quality diagnostic). The
        # eager catalog scans the whole catalog once; the lazy catalog never
        # has it all resident, so it accumulates per fill and logs the
        # incremental contribution -- bounded to the touched sections.
        self._missing_count = 0

    # -- the global node->section inverse (same contract as the eager one) -
    @cached_property
    def sec_of_var(self) -> np.ndarray:
        """``i64[total_variants]`` -- owning SECTION index per flat variant.

        Identical contract + value to :attr:`ColumnarSections.sec_of_var`
        (``np.repeat(arange(n_sections), n_variants)``); depends only on the
        eager section-level ``n_variants``, so it is available without any
        heavy fill.
        """
        return np.repeat(
            np.arange(self.n_variants.size, dtype=np.int64), self.n_variants
        )

    def pce_variant(self) -> np.ndarray:
        """``i64[total_entries]`` -- owning flat variant index per entry.

        The CSR inverse of :attr:`pce_offsets`. Mirrors
        :meth:`ColumnarSections.pce_variant`; it reads ``var_n_calls``
        (heavy) so it is correct only once every section is filled. No
        vector_batch decode consumer calls it (it is part of the eager
        catalog's surface kept for substitutability); a caller that needs
        it over the whole catalog must :meth:`ensure_sections` for all
        sections first.
        """
        return np.repeat(
            np.arange(self.var_n_calls.size, dtype=np.int64),
            self.var_n_calls,
        )

    # -- the lazy-fill entry point -----------------------------------------
    def ensure_sections(self, section_indices: np.ndarray) -> None:
        """Materialise the heavy columns for ``section_indices`` (idempotent).

        Parses ONLY the not-yet-filled sections among ``section_indices``
        via the SAME :func:`parse_sections_columnar` over their offsets +
        lengths, then scatters its value columns into this catalog's global
        backing at the eager CSR slots. Vectorised + idempotent: an
        already-filled section is skipped, so repeated calls (every batch,
        every frontier level) cost only the ``np.unique`` + mask.
        """
        secs = np.asarray(section_indices, dtype=np.int64).reshape(-1)
        if secs.size == 0:
            return
        secs = np.unique(secs)
        secs = secs[~self._filled[secs]]
        if secs.size == 0:
            return
        sub = parse_sections_columnar(
            self._blob, self._offsets[secs], self._lengths[secs]
        )
        self._scatter_call_targets(secs, sub)
        self._scatter_variants(secs, sub)
        self._scatter_per_call_entries(secs, sub)
        self._filled[secs] = True
        self._tally_missing(sub)

    def _tally_missing(self, sub: ColumnarSections) -> None:
        """Accumulate + log the MISSING-edge count for the just-filled set.

        The per-binary inventory the eager catalog emits in one ERROR line
        at adjacency construction; here it is bounded to (and emitted per)
        the touched sections, so the open path never scans the untouched
        ~95% of the catalog. Same ERROR semantics (each MISSING entry is a
        silently-dropped splice edge), now reported as it is discovered.
        """
        new_missing = int(
            (sub.pce_section_variant_index == MISSING_VARIANT_INDEX).sum()
        )
        if new_missing:
            self._missing_count += new_missing
            logger.error(
                "sorted_index: %d per-call entries carry "
                "MISSING_VARIANT_INDEX in newly-touched sections "
                "(running total %d). Each one silently drops a splice edge "
                "-- the callee's variant set does not cover the caller's "
                "vkey.",
                new_missing,
                self._missing_count,
            )

    def missing_variant_index_count(self) -> int:
        """Touched-bounded running MISSING-edge tally (not the full catalog).

        The lazy twin of :meth:`ColumnarSections.missing_variant_index_count`
        -- it returns only what the FILLED sections have contributed (0 at
        construction). The full-catalog count is deliberately never computed
        on the lazy open path; the incremental :meth:`_tally_missing` log is
        the diagnostic surface instead.
        """
        return self._missing_count

    def _scatter_call_targets(
        self, secs: np.ndarray, sub: ColumnarSections
    ) -> None:
        """Write the touched sections' ``ct_*`` slices into the backing.

        ``sub.ct_offsets`` are subset-local; the global destination of
        subset section ``j`` (= global ``secs[j]``) is
        ``ct_offsets[secs[j]] : ct_offsets[secs[j]+1]``. A per-member
        global index (subset member order maps 1:1 to global member order
        because both lay sections out ascending) scatters every ``ct_*``
        column in one shot.
        """
        dst = self._global_member_index(
            secs, self.ct_offsets, sub.ct_offsets
        )
        self.ct_function_name_ptr[dst] = sub.ct_function_name_ptr
        self.ct_function_section_ptr[dst] = sub.ct_function_section_ptr
        self.ct_type[dst] = sub.ct_type
        self.ct_is_matched[dst] = sub.ct_is_matched

    def _scatter_variants(
        self, secs: np.ndarray, sub: ColumnarSections
    ) -> None:
        """Write the touched sections' variant-level slices into the backing.

        Covers ``var_ref_offset`` / ``var_data_offset_shifted`` /
        ``var_n_calls`` (per-variant) AND completes ``pce_offsets`` for the
        touched sections: each section's intra-cumsum of ``var_n_calls``
        rebased onto its eager ``section_pce_base`` (the global per-section
        first-entry index), so the absolute per-call-entry offsets match
        the eager catalog exactly.
        """
        dst = self._global_member_index(
            secs, self.var_offsets, sub.var_offsets
        )
        self.var_ref_offset[dst] = sub.var_ref_offset
        self.var_data_offset_shifted[dst] = sub.var_data_offset_shifted
        self.var_n_calls[dst] = sub.var_n_calls
        # Complete pce_offsets[var_offsets[s]+1 .. var_offsets[s+1]] for each
        # touched section s: the GLOBAL per-call-entry CSR is each section's
        # eager section_pce_base[s] plus the intra-section cumsum of
        # var_n_calls. Computing it directly from the section base + the just
        # -written var_n_calls (rather than rebasing the subset's own pce CSR)
        # keeps the global cumulative property without depending on any
        # subset-internal CSR -- the same values the eager parse's _csr would
        # have produced at these global slots.
        sub_var_section = np.repeat(
            np.arange(secs.size, dtype=np.int64), sub.n_variants
        )
        # Intra-section INCLUSIVE running entry count per variant, then add
        # the section's global base. The subset's own per-call-entry CSR
        # (sub.pce_offsets) is already the subset-GLOBAL cumsum; subtracting
        # each variant's section base (sub.pce_offsets at the section's first
        # variant) leaves the intra-section inclusive end, which rebased onto
        # the eager global section_pce_base reproduces the eager pce_offsets.
        section_base = self._skel.section_pce_base[secs]
        sub_section_first_excl = sub.pce_offsets[sub.var_offsets[:-1]]
        intra_inclusive = (
            sub.pce_offsets[1:] - sub_section_first_excl[sub_var_section]
        )
        self.pce_offsets[dst + 1] = (
            section_base[sub_var_section] + intra_inclusive
        )

    def _scatter_per_call_entries(
        self, secs: np.ndarray, sub: ColumnarSections
    ) -> None:
        """Write the touched sections' ``pce_*`` slices into the backing.

        The per-call-entry members are laid out per VARIANT; the global
        destination index uses the eager global ``pce_offsets`` (now
        completed for these sections by :meth:`_scatter_variants`) against
        the subset's per-call-entry CSR.
        """
        global_pce = self.pce_offsets
        # Global per-variant entry CSR for the touched variants, in subset
        # order: gather the global pce range of each touched section's
        # variants. The subset member order == global member order (ascending
        # sections, ascending variants), so a flat global-member index maps
        # the subset pce_* arrays straight into the backing.
        dst = self._global_entry_index(secs, global_pce, sub)
        self.pce_called_idx[dst] = sub.pce_called_idx
        self.pce_section_variant_index[dst] = sub.pce_section_variant_index

    def _global_member_index(
        self,
        secs: np.ndarray,
        global_offsets: np.ndarray,
        sub_offsets: np.ndarray,
    ) -> np.ndarray:
        """Flat GLOBAL member indices for the touched sections' members.

        ``global_offsets`` is the eager catalog's per-section CSR (e.g.
        ``ct_offsets`` / ``var_offsets``); ``sub_offsets`` the subset
        parse's parallel CSR. Returns, for member ``m`` of subset section
        ``j``, its global index ``global_offsets[secs[j]] + within_j(m)``
        -- so the subset's value column scatters 1:1 into the global slot.
        """
        counts = np.diff(sub_offsets)
        total = int(sub_offsets[-1])
        parent = np.repeat(np.arange(secs.size, dtype=np.int64), counts)
        within = np.arange(total, dtype=np.int64) - sub_offsets[parent]
        return global_offsets[secs[parent]] + within

    def _global_entry_index(
        self,
        secs: np.ndarray,
        global_pce: np.ndarray,
        sub: ColumnarSections,
    ) -> np.ndarray:
        """Flat GLOBAL per-call-entry indices for the touched sections.

        Per-call entries are variant-major; entry ``e`` of subset flat
        variant ``v`` maps to global variant ``g(v)`` and lands at
        ``global_pce[g(v)] + within_v(e)``. ``g(v)`` is the global flat
        variant index of the subset's v-th variant, recovered from the
        eager ``var_offsets`` of the touched sections; ``global_pce`` is
        this catalog's per-call-entry CSR (completed for ``secs`` by
        :meth:`_scatter_variants`).
        """
        global_var = self._global_member_index(
            secs, self.var_offsets, sub.var_offsets
        )
        sub_pce = sub.pce_offsets
        total_e = int(sub_pce[-1])
        counts = np.diff(sub_pce)
        entry_var = np.repeat(
            np.arange(global_var.size, dtype=np.int64), counts
        )
        within_var = np.arange(total_e, dtype=np.int64) - sub_pce[entry_var]
        return global_pce[global_var[entry_var]] + within_var


def parse_sections_columnar_lazy(
    blob_u8: np.ndarray,
    section_offsets: np.ndarray,
    section_lengths: np.ndarray,
) -> LazyColumnarSections:
    """Open a :class:`LazyColumnarSections` over ``section_offsets``.

    The lazy twin of :func:`parse_sections_columnar`: the section-level
    skeleton + CSR offsets are computed eagerly (cheap), and the heavy
    table/variant/per-call columns fill on first
    :meth:`LazyColumnarSections.ensure_sections`. ``section_lengths`` is
    REQUIRED (the per-call-entry CSR base is derived from it without a
    global jump-table read; see :func:`_build_skeleton`).
    """
    return LazyColumnarSections(blob_u8, section_offsets, section_lengths)
