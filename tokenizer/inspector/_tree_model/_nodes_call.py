"""``InlineCallNode`` -- inline call site under a block.

Expandable whenever the :class:`InlineCallEntry` carried a resolvable
:class:`SectionPointerSpec`; LOCAL and PLT both qualify (the writer
treats them identically -- a PLT thunk is sectioned just like a local
function, its body is the small jump-thunk to the resolved extern).
EXTERN has no callee section at all so it never expands.
Expansion constructs a synthetic callee :class:`FunctionNode` against
the same :class:`BackendFactory` the parent tree-open used; the
matching variant's blocks are surfaced DIRECTLY (skipping the
intermediate variant-list level, per plan decision D2) and the other
variants are bundled under a :class:`ShowAllVariantsNode` sibling.
The fallback path (no caller-identity match in the callee's surviving
set, e.g. cross-arm where the callee carries different arches) is to
surface every variant as a sibling, matching the pre-D2 contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.variant_info import VariantIdentity


if TYPE_CHECKING:
    from tokenizer.inspector._render._protocol import (
        BackendFactory,
        FunctionHandle,
    )

    from ._nodes_variant import VariantNode


__all__ = ["InlineCallNode"]


@dataclass
class InlineCallNode:
    """Inline call from a block to another function.

    ``kind`` is the canonical
    :class:`~tokenizer.aligned_data.call_target_type.CallTargetType`
    enum (LOCAL / PLT / EXTERN). ``provider`` is the library /
    sidecar name appended after ``@`` on EXTERN rows; ``None`` for
    LOCAL / PLT and for EXTERN rows whose provider is unknown.
    ``callee_handle`` is ``None`` when the call has no addressable
    callee section (EXTERN, Phase-1 BatchDecode, FtlBackend per plan
    decision 24, or LOCAL/PLT row whose pointer failed to resolve).
    """

    factory: "BackendFactory"
    kind: CallTargetType
    counter_id: int
    callee_name: str
    callee_handle: "FunctionHandle | None"
    variant_idx: int
    provider: str | None
    # Caller-row :class:`VariantIdentity`; used as a fallback pin when
    # the callee-side ``variant_idx`` is :data:`MISSING_VARIANT_INDEX`
    # (e.g. Function-ID self-references where no vkey pin is recorded).
    # Matching is on the canonical-4 axes only (``arch / compiler /
    # compiler_version / opt``) — see :func:`_find_variant_by_caller_identity`
    # — because the raw per-section ``variant_idx`` is opaque (the same
    # numeric index refers to different variants across MATCHED vs
    # UNMATCHED sections, so cross-arm matching by integer index would
    # land on arch-incompatible content). ``None`` keeps the pre-
    # existing all-variants fallback for constructors that do not
    # thread an identity.
    caller_variant_identity: Optional[VariantIdentity] = None
    is_failed: bool = False
    # Per-row horizontal scroll memory; see :mod:`tokenizer.inspector._app._tree_widget`.
    remembered_scroll_x: int = field(default=0, init=False)

    @property
    def can_expand(self) -> bool:
        # Single dispatch point; the UI gates the expand call on this.
        # LOCAL and PLT call_targets both point at a section the writer
        # emitted (the writer's ``_resolve_function_section_ptr`` treats
        # them identically -- PLT thunks are sectioned just like local
        # functions, their body being the small jump-thunk that calls
        # the resolved extern). EXTERN has no callee section so its
        # ``callee_handle`` is always ``None`` here; LOCAL/PLT without
        # a resolved pointer (cross-arm miss) likewise cannot construct
        # a FunctionNode.
        return self.callee_handle is not None

    def expand(self) -> list:
        """Open the callee + inline the matching variant's blocks.

        Constructs a synthetic :class:`FunctionNode` against the same
        factory; ``expand`` on that node returns one
        :class:`VariantNode` per callee variant. The pinned variant
        is chosen by ``self.variant_idx`` when it is not
        :data:`MISSING_VARIANT_INDEX` — that integer is a valid INTRA-
        section reference recorded by the encoder in
        ``per_call_entries``, so it directly indexes into the callee's
        :meth:`RenderBackend.variants` list. When the explicit pin is
        missing, the fallback path matches the callee's variants by
        :attr:`caller_variant_identity` on the canonical-4 build axes
        (:func:`_find_variant_by_caller_identity`) — so Function-ID
        self-references and other no-vkey-pin cases default to a
        variant that shares the caller's ``(arch, compiler,
        compiler_version, opt)`` instead of surfacing the full variant
        list (and crucially do NOT land on an arch-incompatible
        variant the way matching by raw ``variant_idx`` would across
        MATCHED vs UNMATCHED arms). That matched variant's blocks are
        spliced in DIRECTLY as the InlineCallNode's children, skipping
        the intermediate variant-list level (plan decision D2). The
        other variants are bundled under a
        :class:`ShowAllVariantsNode` sibling appended after the
        spliced blocks so they remain reachable.

        When neither pin resolves to a surviving callee variant
        (no explicit pin AND no identity match — e.g. cross-arm call
        whose callee carries only the other arch's variants), the
        fallback is to surface every variant directly as VariantNode
        siblings, matching the pre-D2 contract.
        """
        from ._nodes_function import FunctionNode
        from ._nodes_leaf import ShowAllVariantsNode

        if self.callee_handle is None:
            raise RuntimeError(
                "InlineCallNode.expand called on a non-expandable node "
                f"(kind={self.kind.name}); UI should gate on can_expand."
            )
        callee = FunctionNode(factory=self.factory, handle=self.callee_handle)
        all_variants = callee.expand()

        matched_idx_in_list: int | None
        if self.variant_idx != MISSING_VARIANT_INDEX:
            # Explicit intra-section pin from ``per_call_entries`` —
            # the integer is valid in the callee's own variant index
            # space, so position-by-``variant_idx`` is correct here.
            matched_idx_in_list = _find_matching_variant_index(
                all_variants, self.variant_idx
            )
        elif self.caller_variant_identity is not None:
            matched_idx_in_list = _find_variant_by_caller_identity(
                all_variants, self.caller_variant_identity
            )
        else:
            matched_idx_in_list = None

        if matched_idx_in_list is None:
            # No pinned/matched variant survives -- fall back to
            # surfacing every variant directly.
            return list(all_variants)

        matched = all_variants[matched_idx_in_list]
        others = tuple(
            v for i, v in enumerate(all_variants) if i != matched_idx_in_list
        )
        # Splice the matched variant's blocks in directly (D2: skip the
        # intermediate variant-list level for the pinned variant). The
        # callee's :class:`FunctionNode` already owns the
        # :class:`RenderBackend` instance the :class:`VariantNode` was
        # threaded with, so ``matched.expand()`` resolves blocks via
        # the same backend cache — no extra factory.make call.
        children: list = list(matched.expand())
        if others:
            children.append(
                ShowAllVariantsNode(
                    label="show all variants", other_variants=others
                )
            )
        return children


def _find_matching_variant_index(
    variants: list["VariantNode"], target_variant_idx: int
) -> int | None:
    """Position in ``variants`` whose ``variant_idx`` matches the target.

    The pre-Wave-5 implementation matched on object identity of the
    underlying ``variant_block``; the post-Wave-5 typed contract uses
    the int ``variant_idx`` carried on every :class:`VariantNode`,
    which both backends pin to the same canonical value (FtlBackend's
    lockstep position, BatchDecodeBackend's per-section variant
    index) -- the Protocol is the single source of truth.

    Used ONLY for explicit intra-section ``per_call_entries`` pins
    (where the integer is valid in the callee's own variant index
    space); the cross-arm caller-fallback path goes through
    :func:`_find_variant_by_caller_identity` instead.
    """
    for i, v in enumerate(variants):
        if v.variant_idx == target_variant_idx:
            return i
    return None


def _find_variant_by_caller_identity(
    variants: list["VariantNode"], caller_identity: VariantIdentity,
) -> int | None:
    """Position in ``variants`` whose canonical-4 build axes match.

    Single source of truth for the caller-fallback match key: the
    tuple ``(arch, compiler, compiler_version, opt)`` projected off
    each callee variant's :attr:`VariantNode.label_axes` Mapping
    (both backends emit the same canonical-order Mapping per the
    :class:`RenderedVariant` Protocol contract). The remaining
    :class:`VariantIdentity` fields (``pkg``, ``variant_id``) are
    caller-side bookkeeping — ``pkg`` differs between caller and
    callee whenever they live in different binaries, and
    ``variant_id`` is the writer's dedup-disambiguator — so they are
    intentionally NOT part of the match key.

    Returns the first match (call-target tables emit one variant per
    canonical-4 by the writer's :class:`VariantRegistry` contract, so
    "first" = "unique"); returns ``None`` when no callee variant
    shares the caller's build axes (cross-arm call into a callee that
    only has variants for the other arch — the common case the typed
    identity exists to handle).
    """
    from tokenizer.variant_tokens.prefixes import (
        ARCH_PREFIX, COMP_PREFIX, CVER_PREFIX, OPT_PREFIX,
    )

    target = (
        caller_identity.arch,
        caller_identity.compiler,
        caller_identity.compiler_version,
        caller_identity.opt,
    )
    for i, v in enumerate(variants):
        candidate = (
            v.label_axes.get(ARCH_PREFIX),
            v.label_axes.get(COMP_PREFIX),
            v.label_axes.get(CVER_PREFIX),
            v.label_axes.get(OPT_PREFIX),
        )
        if candidate == target:
            return i
    return None
