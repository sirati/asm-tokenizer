"""Stateless parsers used by ``BinarySession``.

Split out so ``session.py`` stays focused on lifecycle: section parsing
into ``FunctionData`` / ``MatchedFunction`` is pure on its inputs (the
BIN-parsed :class:`Section` / per-variant block + a resolver callable)
and has no business touching the lazy-open machinery. Tests can
exercise these helpers without standing up a full session.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..matched_sections_bin import Section
from ..metadata import extract_metadata_from_variant_block
from .category_counts import compute_category_counts
from .function_data import FunctionData
from .matched_function import MatchedFunction


# Empty uint16 buffer reused when a variant ref cannot be resolved. Sharing one
# instance avoids per-call allocation; consumers MUST treat it as read-only
# (matches the rest of the loader's lazy-view discipline).
_EMPTY_VARIANT_TOKENS: np.ndarray = np.zeros(0, dtype=np.uint16)


def _variant_tokens_from_row(variant_row: Optional[Dict[str, Any]]) -> np.ndarray:
    """Pull the resolver's ``variant_tokens`` ndarray off a variant dict.

    Empty / missing variant row -> the shared empty uint16 buffer so
    ``FunctionData.variant_tokens`` is always a valid uint16 ndarray
    (zero-length signals "no variant resolved", matching the plan's
    "zero-length only on a corrupt dataset" contract).
    """
    if not variant_row:
        return _EMPTY_VARIANT_TOKENS
    tokens = variant_row.get("variant_tokens")
    if tokens is None:
        return _EMPTY_VARIANT_TOKENS
    return tokens


def arm_arrays(arm: Any, kind: str, binary_name: str):
    """Per-function arrays the session uses to slice for ``kind``.

    Matched ``load(idx)`` slices the BIN catalog (per-function), so it
    returns the 2-tuple ``(bin_starts, bin_lengths)`` from
    ``matched_index.bin``. Unmatched ``load(idx)`` slices
    ``_unmatched_data.bin`` directly (the unmatched index is per-
    record 1:1) and the record is self-describing, so it returns a
    SINGLE ndarray of per-record offsets -- no companion ``lengths``
    array.
    """
    if arm is None:
        raise IndexError(f"{kind} arm not loaded for binary {binary_name}")
    if kind == "matched":
        bin_starts = getattr(arm, "bin_starts", None)
        bin_lengths = getattr(arm, "bin_lengths", None)
        if bin_starts is None or bin_lengths is None:
            raise IndexError(
                f"matched arm has no bin_starts/bin_lengths for binary {binary_name}"
            )
        return bin_starts, bin_lengths
    starts = getattr(arm, "starts", None)
    if starts is None:
        raise IndexError(
            f"unmatched arm has no starts for binary {binary_name}"
        )
    return starts


def parse_matched_section(
    section: Section,
    *,
    func_name: str,
    data_slice: Callable,
    resolve_ref: Callable,
) -> MatchedFunction:
    """Parse one matched section's variant blocks into a ``MatchedFunction``.

    ``section`` is the BIN-parsed catalog entry whose variant blocks
    drive per-version ``FunctionData`` construction. ``data_slice(offset)``
    returns ``(insn_rl, block_rl, tokens)``; ``resolve_ref(ref_str)``
    returns the variant dict (or ``None``). Both injected so this
    helper does not import the session's lazy openers.

    Per variant we extract the metadata dict (variant_ref, inlining
    info, data_offset) via
    :func:`extract_metadata_from_variant_block`, merge in the resolver
    output for the canonical-4 axes / filename, and slice the data-bin
    record at the recovered offset. The record is self-describing --
    its header carries every geometry field a reader needs -- so no
    companion length / overlong flag rides alongside the offset.
    """
    variants: List[FunctionData] = []
    for variant in section.variants:
        metadata = extract_metadata_from_variant_block(section, variant)
        variant_row = resolve_ref(metadata["variant_ref"])
        if variant_row is not None:
            for k, v in variant_row.items():
                metadata.setdefault(k, v)
        insn_rl, block_rl, tokens = data_slice(metadata["data_offset"])
        # Per-function COUNTER-Category unique-id counts feed Stage 4a's
        # ALG-4 offset bump. The loader is the single source of truth
        # for this metadata; downstream stages read it from
        # ``FunctionData.metadata["category_counts"]`` without re-decoding.
        metadata["category_counts"] = compute_category_counts(tokens)
        variants.append(
            FunctionData(
                func_name, metadata, tokens, insn_rl, block_rl,
                variant_tokens=_variant_tokens_from_row(variant_row),
            )
        )
    return MatchedFunction(func_name, variants)


def build_unmatched_function_data(
    section: Section,
    func_name: str,
    start: int,
    tokens,
    insn_rl,
    block_rl,
    *,
    variant_slot: int,
    resolve_ref: Callable,
    line_to_name: Dict[int, str],
) -> FunctionData:
    """Assemble an unmatched ``FunctionData`` from its BIN section + bytes.

    ``variant_slot`` indexes into ``section.variants`` and identifies
    THIS record's variant (unmatched sections store one record per
    variant). Its ``variant_ref_offset`` is resolved to pull the
    canonical-4 axes (``arch / compiler / compilerversion / opt``) and
    the variant's prefix-token stream out of ``_variants.bin``; the
    metadata dict that lands on :class:`FunctionData` carries those
    per-record axes directly, NOT the unmatched-arm ``"unknown"``
    placeholder. Downstream consumers (inspector, variant-identity
    factory) read the canonical axes off ``FunctionData.metadata``
    without knowing this is the unmatched arm.

    Output dict carries ``arch``/``compiler``/``compilerversion``/``opt``
    (recovered per-slot), ``variant_refs``, ``variants``, ``called``,
    ``call_targets``, and ``data_offset``. Fields derive from
    ``section``'s parsed call_target table + per-variant blocks:

    * ``variant_refs`` -- hex strings from each variant block's
      ``variant_ref_offset``.
    * ``variants`` -- resolver dicts for every ref that resolved
      (legacy datasets without ``_variants.bin`` see an empty list).
    * ``called`` -- function names recovered from each call_target's
      ``function_name_ptr`` via ``line_to_name``.
    * ``call_targets`` -- ``[[called_idx, function_section_ptr,
      section_variant_index, is_matched_int]]`` flattened across every
      variant's ``per_call_entries``. Records are self-describing in
      ``_data.bin`` so no length / overlong flag crosses this boundary.
    * ``data_offset`` -- the per-record offset the session passed in
      (from ``unmatched_index.bin``).

    The ``variant_refs`` / ``variants`` / ``call_targets`` lists are
    still section-wide (every variant block), preserving the legacy
    metadata contract; the per-record axes ride alongside them.
    """
    variant_refs = [f"{v.variant_ref_offset:x}" for v in section.variants]
    variants = [
        v for v in (resolve_ref(r) for r in variant_refs) if v is not None
    ]
    if variant_slot < 0 or variant_slot >= len(section.variants):
        raise IndexError(
            f"unmatched variant_slot={variant_slot} out of bounds for "
            f"section with {len(section.variants)} variants"
        )
    # Resolve THIS record's variant directly — its ``variant_ref_offset``
    # may differ from neighbouring section variants (cross-arch / cross-
    # compiler variants in the same unmatched section), so picking the
    # first resolved entry of ``variants`` would mis-label every slot
    # past 0. Falling back to ``None`` keeps legacy datasets without a
    # ``_variants.bin`` sidecar (resolver returns ``None``) working.
    this_variant_row = resolve_ref(variant_refs[variant_slot])
    called: List[str] = []
    for ct in section.call_targets:
        name = line_to_name.get(ct.function_name_ptr)
        if name is not None:
            called.append(name)
    call_targets: List[List[int]] = []
    for variant in section.variants:
        for called_idx, section_variant_index in variant.per_call_entries:
            ct = section.call_targets[called_idx]
            call_targets.append(
                [
                    called_idx,
                    ct.function_section_ptr,
                    section_variant_index,
                    1 if ct.is_matched else 0,
                ]
            )

    metadata: Dict[str, Any] = {
        "variant_refs": variant_refs,
        "variants": variants,
        "called": called,
        "call_targets": call_targets,
        "data_offset": start,
        # Per-function COUNTER-Category unique-id counts feed Stage 4a's
        # ALG-4 offset bump. Loader-side single source of truth; the
        # unmatched arm shares the same metadata contract as the matched
        # arm so downstream stages remain arm-agnostic.
        "category_counts": compute_category_counts(tokens),
    }
    # Per-slot canonical-4 axes via the variant resolver. The resolver
    # returns ``{arch, compiler, compilerversion, opt, ...}`` keys; we
    # merge them in so consumers (``VariantInfo.from_function_data_metadata``,
    # the inspector's label_axes, the arch-prefix elider) see real
    # axes instead of the ``"unknown"`` placeholder. Missing resolver
    # (legacy datasets without ``_variants.bin``) keeps the historical
    # unknown-axis fallback.
    if this_variant_row is not None:
        for axis_key in ("arch", "compiler", "compilerversion", "opt"):
            value = this_variant_row.get(axis_key)
            if value is not None:
                metadata[axis_key] = value
        for k, v in this_variant_row.items():
            metadata.setdefault(k, v)
    metadata.setdefault("arch", "unknown")
    metadata.setdefault("compiler", "unknown")
    metadata.setdefault("compilerversion", "unknown")
    metadata.setdefault("opt", "unknown")
    return FunctionData(
        func_name, metadata, tokens, insn_rl, block_rl,
        variant_tokens=_variant_tokens_from_row(this_variant_row),
    )
