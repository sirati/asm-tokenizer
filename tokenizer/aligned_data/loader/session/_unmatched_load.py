"""Unmatched-arm load path for :class:`BinarySession`.

Single concern: parse an unmatched BIN section (one record per
variant) + materialise its variant bodies from
``<binary>_unmatched_data.bin``, plus the per-record -> per-section
mapping helpers that locate the owning section. Exposed as a mixin
:class:`_UnmatchedLoadMixin` so the methods stay on
:class:`BinarySession` (callers need not know about the split). Every
attribute it reads -- ``_meta_get``, ``_binary_name``,
``_parse_section_at``, ``_open_data``, ``_slice_data_record``,
``get_variant_by_ref`` -- is owned by :class:`BinarySession` itself;
this class holds no state.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ...index_format import ALIGNMENT_SHIFT
from ...matched_sections_bin import Section
from .._session_parsers import arm_arrays, build_unmatched_function_data
from ..function_data import FunctionData


class _UnmatchedLoadMixin:
    """Mixin providing the unmatched-arm load + section-lookup helpers.

    Every method reads attributes / methods that :class:`BinarySession`
    owns; this class deliberately holds no state of its own. ``self``
    is typed ``Any`` inside the bodies because the concrete attributes
    live on the subclass.
    """

    def _load_unmatched_record_and_section(  # type: ignore[no-untyped-def]
        self, idx: int
    ) -> Tuple[Section, int, FunctionData]:
        """Parse the unmatched record at ``idx`` + its owning section.

        Returns ``(section, section_offset, FunctionData)`` where
        ``section_offset`` is the BIN byte offset of the owning section
        (NOT the per-record ``_unmatched_data.bin`` offset). Shared by
        :py:meth:`load_unmatched` and the batch-decode pipeline.

        The per-record ``idx`` maps to a ``(section base record, variant
        slot)`` pair via the arm's ``record_to_section_idx`` mapping; the
        body load delegates to :py:meth:`_load_unmatched_variant_body`,
        which slices the variant block's OWN
        ``data_offset_shifted << ALIGNMENT_SHIFT``. The writer emits the
        per-record index entries in encounter order but sorts the section's
        variant blocks by ``variant_ref_offset``, so the positional
        ``starts[idx]`` and the slot-J variant block are NOT guaranteed to
        coincide; slicing by the variant's own offset is the single robust
        source of truth (symmetric with the matched arm and the callee
        walk), with no residual dependence on emit-order==vref-order.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        section, section_offset = self._unmatched_section_for_record(arm, idx)
        # Per-record -> per-variant slot inside the owning section.
        # Unmatched sections store one record per variant; the slot is
        # the offset from the section's first-record idx in the arm's
        # ``record_to_section_idx`` mapping.
        section_idx = self._unmatched_section_idx(arm, idx)
        base = self._unmatched_record_slot_base(arm, section_idx)
        variant_slot = idx - base
        fd = self._load_unmatched_variant_body(base, variant_slot, section)
        return section, section_offset, fd

    def _load_unmatched_variant_body(  # type: ignore[no-untyped-def]
        self, idx: int, variant_index: int, section: Section
    ) -> FunctionData:
        """Load ONE unmatched section variant body, reusing the section.

        ``section`` is the already-parsed owning section the caller
        obtained from :py:meth:`_unmatched_section_meta` (threaded through
        the callee walk's :class:`ResolvedCalleeMeta`), so this load does
        NOT re-derive it via :py:meth:`_unmatched_section_for_record` (no
        ``_sections.bin`` re-parse).

        The data record is sliced at ``section.variants[variant_index]``'s
        OWN ``data_offset_shifted << ALIGNMENT_SHIFT`` -- the SAME way the
        matched arm slices (:func:`parse_matched_variant`). Unmatched
        sections store one DISTINCT body record per variant, so loading by
        the variant block's own offset (rather than the positional
        ``starts[base + variant_index]``) splices variant-``variant_index``'s
        body regardless of whether the index-entry order and the
        variant-block order coincide -- removing the silent dependence on
        the writer's emit-order==vref-order lock-step. ``idx`` is retained
        for the section-keyed ``func_names`` lookup (the name is a section
        property, shared by every variant) and its bounds check. Raises
        :class:`IndexError` if ``variant_index`` is out of range.

        For ``variant_index == 0`` this is byte-identical to the legacy
        first-record load: the section's first variant block carries the
        same ``data_offset_shifted`` as ``starts[base]``.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        if variant_index < 0 or variant_index >= len(section.variants):
            raise IndexError(
                f"unmatched section idx={idx} has {len(section.variants)} "
                f"variants; variant_index {variant_index} out of range"
            )
        start = (
            section.variants[variant_index].data_offset_shifted << ALIGNMENT_SHIFT
        )
        data_mmap = self._open_data("unmatched")
        insn_rl, block_rl, tokens = self._slice_data_record(data_mmap, start)
        line_to_name = self._meta_get("line_to_name") or {}
        return build_unmatched_function_data(
            section,
            self._unmatched_func_name(arm, idx),
            start,
            tokens, insn_rl, block_rl,
            variant_slot=variant_index,
            resolve_ref=self.get_variant_by_ref,
            line_to_name=line_to_name,
        )

    def _unmatched_section_meta(  # type: ignore[no-untyped-def]
        self, idx: int
    ) -> Tuple[Section, int]:
        """Parse an unmatched record's owning section only (no body).

        Returns ``(section, section_offset)`` for the record at per-record
        ``idx`` -- the same parsed :class:`Section` and BIN section offset
        :py:meth:`_load_unmatched_record_and_section` produces, but
        WITHOUT slicing the ``_unmatched_data.bin`` record body. The callee
        walk defers the body load (sliced at the J-resolved variant block's
        own offset, via :py:meth:`_load_unmatched_variant_body`) to the
        surviving pairs.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        return self._unmatched_section_for_record(arm, idx)

    def _load_unmatched_section_and_all_variants(  # type: ignore[no-untyped-def]
        self, idx: int
    ) -> Tuple[Section, int, list]:
        """Parse the unmatched section at ``idx`` + build every variant's FunctionData.

        Mirrors :py:meth:`_load_matched_section_and_variants` for the
        unmatched arm: returns ``(section, section_offset, list[FunctionData])``
        where the per-variant list is parallel to ``section.variants``.
        Unmatched sections store one record per variant; each body is
        sliced at its variant block's own
        ``data_offset_shifted << ALIGNMENT_SHIFT`` via the shared
        :py:meth:`_load_unmatched_variant_body` (the single owner of the
        slice), so the result never depends on the writer's encounter-order
        index entries lining up with the sorted variant blocks. ``idx``
        MUST be the section's first-record idx (the value
        :py:meth:`_idx_for_section_offset` returns for the unmatched arm);
        a non-base record idx raises :class:`ValueError`.
        """
        arm = self._meta_get("unmatched_arm")
        starts = arm_arrays(arm, "unmatched", self._binary_name)
        if idx >= len(starts):
            raise IndexError(f"Index {idx} out of bounds for unmatched functions")
        # Pin ``idx`` to the section's first-record slot. Loading a
        # non-base record into the per-section variants list would lose
        # the preceding slots, breaking the parallel
        # ``section.variants`` <-> returned list contract.
        section_idx = self._unmatched_section_idx(arm, idx)
        base = self._unmatched_record_slot_base(arm, section_idx)
        if idx != base:
            raise ValueError(
                f"unmatched section variants require first-record idx "
                f"(section[{section_idx}] base={base}); got idx={idx}"
            )
        section, section_offset = self._unmatched_section_for_record(arm, base)
        variants = [
            self._load_unmatched_variant_body(base, slot, section)
            for slot in range(len(section.variants))
        ]
        return section, section_offset, variants

    def _unmatched_section_for_record(  # type: ignore[no-untyped-def]
        self, arm: Any, idx: int
    ) -> Tuple[Section, int]:
        """Resolve the BIN section that owns the per-record ``idx``.

        The unmatched index is per-RECORD (one entry per
        ``_unmatched_data.bin`` record). The arm pre-computes
        ``record_to_section_idx[idx]`` at load time (O(M) once over
        the BIN walk); this dispatch is O(1) for the section-idx
        lookup — negligible compared to the BIN parse it then triggers.

        Returns ``(section, section_offset)`` -- the parsed section and
        its BIN byte offset, so callers (notably the batch-decode pipeline)
        can use the offset as a cycle key without re-deriving it.

        No positional ``starts[idx] == variant.data_offset_shifted << 4``
        drift check is performed: the body load slices each variant block's
        OWN ``data_offset_shifted`` (see :py:meth:`_load_unmatched_variant_body`),
        so the per-record index-entry offset is never used to locate a body.
        The writer emits index entries in encounter order but sorts variant
        blocks by ``variant_ref_offset``, so the two orders legitimately
        differ; an equality assertion against ``starts[idx]`` would FALSELY
        reject correctly-written corpora rather than guard corruption.
        """
        section_idx = self._unmatched_section_idx(arm, idx)
        section_starts = getattr(arm, "section_starts", None)
        section_offset = int(section_starts[section_idx])
        section = self._parse_section_at(section_offset)
        return section, section_offset

    def _unmatched_func_name(  # type: ignore[no-untyped-def]
        self, arm: Any, idx: int
    ) -> str:
        """Per-record function name via the pre-cached mapping.

        Falls through to the ``unmatched_<idx>`` sentinel only when
        the section index points beyond ``func_names`` -- this
        indicates the function-names sidecar drifted from the BIN
        catalog at build time and is normally caught earlier by the
        arm-load FID-resolution check.
        """
        names = getattr(arm, "func_names", None) or []
        section_idx = self._unmatched_section_idx(arm, idx)
        if 0 <= section_idx < len(names):
            return names[section_idx]
        return f"unmatched_{idx}"

    def _unmatched_section_idx(  # type: ignore[no-untyped-def]
        self, arm: Any, idx: int
    ) -> int:
        """Look up the per-record -> per-section index via the arm's
        pre-cached mapping. Raises :class:`IndexError` on out-of-range
        ``idx`` with the same wording the legacy section walk used.
        """
        mapping = getattr(arm, "record_to_section_idx", None)
        if mapping is None or len(mapping) == 0:
            raise IndexError(
                f"unmatched arm has no record_to_section_idx for record "
                f"{idx} on binary {self._binary_name}"
            )
        if idx < 0 or idx >= len(mapping):
            raise IndexError(
                f"unmatched record idx={idx} out of bounds (have "
                f"{len(mapping)} records on binary {self._binary_name})"
            )
        return int(mapping[idx])

    def _unmatched_record_slot_base(  # type: ignore[no-untyped-def]
        self, arm: Any, section_idx: int
    ) -> int:
        """First record index belonging to section ``section_idx``.

        Derived once per call from the pre-cached mapping; the slot
        within the section is ``idx - base``. ``np.searchsorted`` on
        the contiguous-section mapping is O(log K) — the same
        derivation the legacy section-walk used to accumulate via the
        ``consumed`` counter, just sourced from the mapping instead
        of re-parsing the BIN.
        """
        mapping = arm.record_to_section_idx
        # np.searchsorted on the contiguous-section mapping returns
        # the first record index whose section_idx >= target; for an
        # exact match this is exactly the section's base record.
        return int(np.searchsorted(mapping, section_idx, side="left"))
