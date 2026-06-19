"""Matched-arm load path for :class:`BinarySession`.

Single concern: parse a matched BIN section + materialise its variant
bodies from ``<binary>_data.bin``. Exposed as a mixin
:class:`_MatchedLoadMixin` so the methods stay on
:class:`BinarySession` (callers need not know about the split). Every
attribute it reads -- ``_meta_get``, ``_binary_name``,
``_parse_section_at``, ``_open_data``, ``_slice_data_record``,
``get_variant_by_ref`` -- is owned by :class:`BinarySession` itself;
this class holds no state.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...matched_sections_bin import Section
from ...matched_sections_columnar import read_n_variants_columnar
from .._session_parsers import (
    arm_arrays,
    parse_matched_section,
    parse_matched_variant,
)
from ..function_data import FunctionData
from ..matched_function import MatchedFunction


class _MatchedLoadMixin:
    """Mixin providing the matched-arm load + per-variant body helpers.

    The matched + unmatched ``load_*`` paths share a need with the
    batch-decode pipeline: BOTH want the parsed :class:`Section` (for
    call_target walking) and the BIN section offset (for cycle keys)
    alongside the per-function data. Factoring those reads into
    dedicated private helpers keeps ``load_matched`` byte-for-byte
    semantically identical (single source of truth) while exposing the
    section + offset to ``_load_matched_for_splice`` without re-parsing.

    Every method reads attributes / methods that :class:`BinarySession`
    owns; this class deliberately holds no state of its own. ``self``
    is typed ``Any`` inside the bodies because the concrete attributes
    live on the subclass.
    """

    def _load_matched_section_and_variants(  # type: ignore[no-untyped-def]
        self, idx: int
    ) -> Tuple[Section, int, MatchedFunction]:
        """Parse the matched section at ``idx`` + build all its variants.

        Returns ``(section, section_offset, MatchedFunction)``. The
        ``Section`` is the parsed BIN catalog entry (call_targets +
        variant blocks); ``section_offset`` is the BIN byte offset
        from ``bin_starts[idx]``. Shared by :py:meth:`load_matched`
        and the batch-decode pipeline.
        """
        arm = self._meta_get("matched_arm")
        bin_starts, _bin_lengths = arm_arrays(arm, "matched", self._binary_name)
        if idx >= len(bin_starts):
            raise IndexError(f"Index {idx} out of bounds for matched functions")
        section_offset = int(bin_starts[idx])
        section = self._parse_section_at(section_offset)
        data_mmap = self._open_data("matched")
        func_names = getattr(arm, "func_names", None) or []
        if idx >= len(func_names):
            raise IndexError(
                f"matched arm func_names short of index {idx} "
                f"(have {len(func_names)})"
            )
        func_name = func_names[idx]
        matched = parse_matched_section(
            section,
            func_name=func_name,
            data_slice=lambda o: self._slice_data_record(data_mmap, o),
            resolve_ref=self.get_variant_by_ref,
        )
        return section, section_offset, matched

    def _matched_section_offset(  # type: ignore[no-untyped-def]
        self, idx: int
    ) -> int:
        """BIN byte offset of the matched section at ``idx`` -- NO parse.

        The matched locator stores the section offset directly
        (``bin_starts[idx]``), so this is a pure O(1) index lookup that
        never touches ``_sections.bin``. The single source of truth for
        the matched ``idx -> section_offset`` map: both the parse-paying
        :py:meth:`_matched_section_meta` and the parse-free
        vector_batch geometry resolve key off this.
        """
        arm = self._meta_get("matched_arm")
        bin_starts, _bin_lengths = arm_arrays(arm, "matched", self._binary_name)
        if idx >= len(bin_starts):
            raise IndexError(f"Index {idx} out of bounds for matched functions")
        return int(bin_starts[idx])

    def _matched_section_meta(  # type: ignore[no-untyped-def]
        self, idx: int
    ) -> Tuple[Section, int]:
        """Parse a matched section's BIN catalog entry only (no bodies).

        Returns ``(section, section_offset)`` -- the same parsed
        :class:`Section` and BIN byte offset
        :py:meth:`_load_matched_section_and_variants` produces, but
        WITHOUT touching ``_data.bin`` (no per-variant body materialised).
        The callee walk's once-only inclusion decision keys solely on the
        callee ``section_offset`` and the parent's per-call J-resolution,
        so the body load is deferred to the survivors via
        :py:meth:`_load_matched_variant_body`.
        """
        section_offset = self._matched_section_offset(idx)
        section = self._parse_section_at(section_offset)
        return section, section_offset

    def _matched_section_variant_counts(  # type: ignore[no-untyped-def]
        self, section_indices: np.ndarray
    ) -> np.ndarray:
        """Per-section variant count for matched ``section_indices`` -- header-only.

        Maps each matched section idx to its ``bin_starts`` byte offset
        (the same ``idx -> section_offset`` source of truth
        :py:meth:`_matched_section_offset` owns) and reads the count via
        the header-only :func:`...matched_sections_columnar.read_n_variants_columnar`
        gather over the cached ``_sections.bin`` uint8 mapping. Body-free:
        it pages in only the touched section headers (the ``n_variants``
        u16 at ``offset + 6``), never a jump table / call_target table /
        variant block -- the cheapest way to get a whole batch's variant
        counts without the per-section :func:`parse_section_bin` object
        build.

        ``section_indices`` is an integer ndarray of matched section idx;
        returns ``int64[len(section_indices)]`` parallel to the input.
        Raises :class:`IndexError` if any idx is out of range for the
        matched arm.
        """
        idx = np.asarray(section_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return np.zeros(0, dtype=np.int64)
        arm = self._meta_get("matched_arm")
        bin_starts, _bin_lengths = arm_arrays(arm, "matched", self._binary_name)
        if int(idx.max()) >= len(bin_starts):
            raise IndexError(
                f"matched section idx {int(idx.max())} out of bounds "
                f"(have {len(bin_starts)} matched sections)"
            )
        section_offsets = np.asarray(bin_starts, dtype=np.int64)[idx]
        return read_n_variants_columnar(
            self._sections_bin_u8(), section_offsets
        )

    def _load_matched_variant_body(  # type: ignore[no-untyped-def]
        self, idx: int, variant_index: int, section: Section
    ) -> FunctionData:
        """Load ONE matched section variant body from ``_data.bin``.

        ``section`` is the already-parsed BIN catalog entry the caller
        obtained from :py:meth:`_matched_section_meta` (threaded through
        the callee walk's :class:`ResolvedCalleeMeta`), so this load does
        NOT re-parse ``_sections.bin`` -- it materialises only
        ``section.variants[variant_index]`` via the shared
        :func:`parse_matched_variant`. That is the same single-variant
        parse :py:meth:`_load_matched_section_and_variants` runs per
        variant, so the returned body is byte-identical to that path's
        ``MatchedFunction.variants[variant_index]``. ``idx`` is retained
        for the O(1) ``func_names[idx]`` lookup (the name carried on the
        body) and its bounds check. Raises :class:`IndexError` if
        ``variant_index`` is out of range.
        """
        arm = self._meta_get("matched_arm")
        if variant_index < 0 or variant_index >= len(section.variants):
            raise IndexError(
                f"matched function idx={idx} has {len(section.variants)} "
                f"variants; variant_index {variant_index} out of range"
            )
        func_names = getattr(arm, "func_names", None) or []
        if idx >= len(func_names):
            raise IndexError(
                f"matched arm func_names short of index {idx} "
                f"(have {len(func_names)})"
            )
        data_mmap = self._open_data("matched")
        return parse_matched_variant(
            section,
            section.variants[variant_index],
            func_name=func_names[idx],
            data_slice=lambda o: self._slice_data_record(data_mmap, o),
            resolve_ref=self.get_variant_by_ref,
        )

    def _load_matched_variant_bodies(  # type: ignore[no-untyped-def]
        self, idx: int, section: Section, variant_indices
    ) -> "list[FunctionData]":
        """Load several matched variant bodies of one section.

        Parallel to ``variant_indices`` (the plural counterpart the
        resolver's batch body load dispatches to). The matched arm carries
        no per-slot whole-section rebuild, so this is a plain loop over
        :py:meth:`_load_matched_variant_body` -- the symmetry with the
        unmatched plural loader keeps the resolver arm-agnostic, with no
        shared bundle to thread.
        """
        return [
            self._load_matched_variant_body(idx, v, section)
            for v in variant_indices
        ]
