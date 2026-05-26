"""Leaf node dataclasses for the inspector tree model.

Terminal nodes never expand into children; the UI gates expand on
``can_expand``. The leaves are :class:`AsmLeaf` (one rendered asm-like
line, optionally carrying :data:`Openable` sidecar entries that produce
child rows on expand), :class:`NumberPrecisionLeaf` (terminal
full-precision display for a NUMBER row), and
:class:`ShowAllVariantsNode` (the sibling shown under an inline-call
when only some of the callee's variants matched the caller's pin).

``AsmLeaf`` implements the 3-arm expand contract (per plan W3-2
W4-amended):

* ``len(openables) == 0`` -> terminal; ``can_expand`` is False.
* ``len(openables) == 1`` -> expanding produces THAT openable's target
  node directly (no intermediate wrapper row).
* ``len(openables) >= 2`` -> expanding produces one wrapper row per
  openable; each row is itself expandable per the 1-arm rule.

Dispatch on openable identity is a ``match`` statement on the
dataclass type -- there is intentionally NO ``OpenableKind`` enum.
The leaf carries the parent BlockNode's ``factory`` / ``backend`` /
``variant_idx`` so the InlineCall / InlineJump branches can construct
their target node with the same model-graph context the legacy
sibling-LineItem path used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

from tokenizer.aligned_data.loader.decoded._number_render import (
    InlineNumberPrecisionEntry,
)
from tokenizer.inspector._label import inline_call_label, inline_jump_label
from tokenizer.inspector._render._protocol import (
    InlineCallEntry,
    InlineJumpEntry,
    Openable,
)


if TYPE_CHECKING:
    from tokenizer.inspector._render._protocol import (
        BackendFactory,
        RenderBackend,
    )

    from ._nodes_variant import VariantNode


__all__ = [
    "AsmLeaf",
    "NumberPrecisionLeaf",
    "ShowAllVariantsNode",
]


@dataclass
class AsmLeaf:
    """One asm-like line inside a block.

    Carries the per-instruction :data:`Openable` sidecar (empty tuple
    by default; populated by the row-walker / FTL emitter when the
    instruction has expandable inline call sites, jump targets, or
    full-precision number renderings). The 3-arm expand contract is
    implemented in :meth:`expand`; ``can_expand`` is True iff
    ``openables`` is non-empty.

    ``factory`` / ``backend`` / ``variant_idx`` are the parent
    BlockNode's refs threaded down so an :class:`InlineCallEntry`
    can spawn an :class:`InlineCallNode` and an :class:`InlineJumpEntry`
    can spawn an :class:`InlineJumpNode` -- mirroring what the legacy
    sibling-LineItem translation did inline. Default ``None`` keeps
    test fixtures + minimal call sites (``AsmLeaf(text=...)``) working
    when the leaf has no openables to expand.
    """

    text: str
    openables: Tuple[Openable, ...] = field(default_factory=tuple)
    factory: Optional["BackendFactory"] = None
    backend: Optional["RenderBackend"] = None
    variant_idx: int = -1
    is_failed: bool = False
    # Per-row horizontal scroll memory; the UI saves the row's current
    # ``scroll_offset.x`` here on manual pan and restores it when the
    # cursor returns to this row. See :mod:`tokenizer.inspector._app._tree_widget`.
    remembered_scroll_x: int = field(default=0, init=False)

    @property
    def can_expand(self) -> bool:
        # Gate expansion on at least one openable being attached; an
        # AsmLeaf with no sidecar entries is a terminal row.
        return len(self.openables) > 0

    def expand(self) -> list:
        """3-arm expand: 0 -> raise, 1 -> dispatch, 2+ -> wrap each.

        Callers MUST gate on :attr:`can_expand` -- the 0-arm path
        raises :class:`NotImplementedError` to match the legacy
        terminal-leaf contract.
        """
        n = len(self.openables)
        if n == 0:
            raise NotImplementedError(
                "AsmLeaf with empty openables is terminal; "
                "gate expansion on can_expand"
            )
        if n == 1:
            return [
                _expand_single_openable(
                    self.openables[0],
                    factory=self.factory,
                    backend=self.backend,
                    variant_idx=self.variant_idx,
                )
            ]
        # 2+ openables -> one wrapper row per openable; each wrapper's
        # own expand surfaces the underlying openable's target node.
        return [
            _wrap_openable_as_node(
                openable,
                factory=self.factory,
                backend=self.backend,
                variant_idx=self.variant_idx,
            )
            for openable in self.openables
        ]


def _expand_single_openable(
    openable: Openable,
    *,
    factory: Optional["BackendFactory"],
    backend: Optional["RenderBackend"],
    variant_idx: int,
):
    """Return the tree node a single openable expands to.

    Dispatch is a ``match`` on the dataclass identity (per plan W3-2
    W4-amended: no ``OpenableKind`` enum, no string discriminator).
    The InlineCall / InlineJump branches construct the concrete model
    node the legacy :func:`_translate_line_items` produced for the
    sibling LineItem; the number-precision branch produces a terminal
    :class:`NumberPrecisionLeaf` (no further expand).
    """
    # Lazy imports break the cycle between ``_nodes_block`` /
    # ``_nodes_call`` and this module.
    from tokenizer.inspector._render._protocol import FunctionHandle

    from ._nodes_block import InlineJumpNode
    from ._nodes_call import InlineCallNode

    match openable:
        case InlineCallEntry():
            spec = openable.callee_section_pointer
            callee_handle = (
                None
                if spec is None
                else FunctionHandle(
                    arm=spec.arm, idx=spec.idx, name=openable.callee_name
                )
            )
            return InlineCallNode(
                factory=factory,
                kind=openable.kind,
                counter_id=openable.counter_id,
                callee_name=openable.callee_name,
                callee_handle=callee_handle,
                variant_idx=openable.variant_idx,
                provider=openable.provider,
                caller_variant_identity=openable.caller_variant_identity,
            )
        case InlineJumpEntry():
            return InlineJumpNode(
                factory=factory,
                backend=backend,
                variant_idx=variant_idx,
                target_block_idx=openable.target_block_idx,
            )
        case InlineNumberPrecisionEntry():
            return NumberPrecisionLeaf(text=openable.full_text)
        case _:
            # Closed Openable union; any miss is a render/model drift.
            raise TypeError(
                f"unknown openable type: {type(openable).__name__}"
            )


def _wrap_openable_as_node(
    openable: Openable,
    *,
    factory: Optional["BackendFactory"],
    backend: Optional["RenderBackend"],
    variant_idx: int,
) -> "AsmLeaf":
    """Wrap one openable in its own AsmLeaf (for the 2+ case).

    Each wrapper row carries exactly that single openable so its own
    ``expand`` falls through the 1-arm path above. The wrapper's
    label text is the openable's natural row label -- a short
    descriptor that matches the legacy sibling label so the UI's
    label dispatcher keeps producing the right text without a special
    case for the wrapper kind. The parent's factory / backend /
    variant_idx are propagated so the wrapper's own expand can hit
    the same construction logic as the 1-arm path.
    """
    return AsmLeaf(
        text=_label_for_openable(openable),
        openables=(openable,),
        factory=factory,
        backend=backend,
        variant_idx=variant_idx,
    )


def _label_for_openable(openable: Openable) -> str:
    """Short row label for a wrapper-AsmLeaf (2+ case).

    Routes per-openable label assembly through the canonical
    :mod:`tokenizer.inspector._label` helpers so the 2+-arm wrapper
    rows and the 1-arm dispatched-node rows render labels off the
    SAME source of truth (single concern: ``_label.py`` owns inline-
    row label formatting). The number-precision case has no canonical
    helper -- its label IS the pre-rendered ``full_text`` carried on
    the entry, so passthrough is the right contract.
    """
    match openable:
        case InlineCallEntry():
            return inline_call_label(
                openable.kind,
                openable.counter_id,
                openable.callee_name,
                openable.provider,
            )
        case InlineJumpEntry():
            return inline_jump_label(openable.target_block_idx)
        case InlineNumberPrecisionEntry():
            return openable.full_text
        case _:
            raise TypeError(
                f"unknown openable type: {type(openable).__name__}"
            )


@dataclass
class NumberPrecisionLeaf:
    """Terminal leaf showing one number's full-precision text.

    Spawned by :meth:`AsmLeaf.expand` when the row's lone openable is
    an :class:`InlineNumberPrecisionEntry` (1-arm dispatch). The full
    text is what the encoder's full-precision renderer produced; no
    further expansion is possible.
    """

    text: str
    can_expand: bool = field(default=False, init=False)
    is_failed: bool = False
    # Per-row horizontal scroll memory; see :class:`AsmLeaf`.
    remembered_scroll_x: int = field(default=0, init=False)

    def expand(self) -> list:
        """Terminal node -- callers must gate on ``can_expand``."""
        raise NotImplementedError(
            "NumberPrecisionLeaf is terminal; gate expansion on can_expand"
        )


@dataclass
class ShowAllVariantsNode:
    """``[+] show all variants`` sibling under an InlineCallNode.

    Holds the variants of the callee that are NOT the caller's
    matching variant; expand yields them.
    """

    label: str
    other_variants: tuple["VariantNode", ...]
    can_expand: bool = field(default=True, init=False)
    is_failed: bool = False
    # Per-row horizontal scroll memory; see :class:`AsmLeaf`.
    remembered_scroll_x: int = field(default=0, init=False)

    def expand(self) -> list:
        return list(self.other_variants)
