"""Per-kind typed :class:`NodeKey` dataclasses + :data:`NodePath` alias.

Single concern: define the typed identity discriminators used by the
:class:`NodePath` capture / restore machinery. Each tree-model node
kind contributes its own frozen :class:`dataclass` carrying the
minimum field-set needed to uniquely identify the node within its
parent's child list (some are globally-unique, others
parent-relative -- documented per dataclass).

No string-typed discriminators cross the boundary (per the codebase's
no-stringly-typed-API rule); :class:`VariantGroupKey` holds the
typed :class:`AxisKind` enum + raw axis ``key`` + bucket value
verbatim, and :class:`InlineCallKey` / :class:`AsmLeafWrapperCallKey`
hold the canonical :class:`CallTargetType` enum.

The path itself (:data:`NodePath`) is just a hashable tuple of
:class:`NodeKey` instances, root-first; the :mod:`._walk` +
:mod:`._match` modules consume the tuple shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.inspector._render._protocol import (
    BlockKind,
    FunctionHandle,
)
from tokenizer.variant_info import VariantIdentity

from .._order import AxisKind


__all__ = [
    "AsmLeafTerminalKey",
    "AsmLeafWrapperCallKey",
    "AsmLeafWrapperJumpKey",
    "AsmLeafWrapperNumberKey",
    "BlockKey",
    "FunctionKey",
    "InlineCallKey",
    "InlineJumpKey",
    "NodeKey",
    "NodePath",
    "NumberPrecisionKey",
    "ShowAllVariantsKey",
    "VariantGroupKey",
    "VariantKey",
]


# ---------------------------------------------------------------------------
# Per-kind typed NodeKey dataclasses (one per relevant tree-model class).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionKey:
    """Identifies a :class:`FunctionNode` by its canonical handle."""

    handle: FunctionHandle


@dataclass(frozen=True)
class VariantKey:
    """Identifies a :class:`VariantNode` by its canonical
    :class:`VariantIdentity`.

    The model's ``variant_idx`` is a per-backend integer that may
    re-number across rebuilds; the canonical identity (shared with the
    rest of the codebase via :class:`VariantInfo.__eq__`) survives.
    """

    identity: VariantIdentity


@dataclass(frozen=True)
class VariantGroupKey:
    """Identifies a :class:`VariantGroupNode` by its axis + bucket value."""

    axis_kind: AxisKind
    axis_key: str
    axis_value: str


@dataclass(frozen=True)
class BlockKey:
    """Identifies a :class:`BlockNode` by its typed kind + block index."""

    kind: BlockKind
    block_idx: int


@dataclass(frozen=True)
class InlineJumpKey:
    """Identifies an :class:`InlineJumpNode` by its target block index."""

    target_block_idx: int


@dataclass(frozen=True)
class InlineCallKey:
    """Identifies an :class:`InlineCallNode` by ``(kind, counter_id)``.

    :class:`InlineCallNode.counter_id` is the encoder's per-Category
    counter (unique per call-target kind within one block); the
    :class:`CallTargetType` enum disambiguates across categories.
    """

    kind: CallTargetType
    counter_id: int


@dataclass(frozen=True)
class AsmLeafWrapperCallKey:
    """:class:`AsmLeaf` wrapper for an :class:`InlineCallEntry`.

    Mirrors :class:`InlineCallKey` so the wrapper's identity matches
    the node it expands into (one-arm dispatch).
    """

    kind: CallTargetType
    counter_id: int


@dataclass(frozen=True)
class AsmLeafWrapperJumpKey:
    """:class:`AsmLeaf` wrapper for an :class:`InlineJumpEntry`."""

    target_block_idx: int


@dataclass(frozen=True)
class AsmLeafWrapperNumberKey:
    """:class:`AsmLeaf` wrapper for an :class:`InlineNumberPrecisionEntry`."""

    full_text: str


@dataclass(frozen=True)
class AsmLeafTerminalKey:
    """Terminal (no-openable) :class:`AsmLeaf` -- sibling index identity.

    Used only for cursor restoration on a leaf row; terminal asm
    lines do not appear in the expand set (they have no children).
    The line-item stream is deterministic per
    :meth:`RenderBackend.render_block`, so the sibling index is
    stable across rebuilds.
    """

    sibling_index: int


@dataclass(frozen=True)
class ShowAllVariantsKey:
    """Singleton key for the lone :class:`ShowAllVariantsNode` under a parent."""

    # Frozen empty dataclass — equality is structural, hashable via dataclass.


@dataclass(frozen=True)
class NumberPrecisionKey:
    """Identifies a :class:`NumberPrecisionLeaf` by its rendered text."""

    text: str


# Union of every concrete per-kind key (closed set; the matcher
# dispatcher in :mod:`._walk` stays exhaustive).
NodeKey = Union[
    FunctionKey,
    VariantKey,
    VariantGroupKey,
    BlockKey,
    InlineJumpKey,
    InlineCallKey,
    AsmLeafWrapperCallKey,
    AsmLeafWrapperJumpKey,
    AsmLeafWrapperNumberKey,
    AsmLeafTerminalKey,
    ShowAllVariantsKey,
    NumberPrecisionKey,
]


# A NodePath is a hashable tuple of NodeKeys, root-first.
NodePath = Tuple[NodeKey, ...]
