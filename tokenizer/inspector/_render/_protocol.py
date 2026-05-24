"""Typed line-source Protocol for the inspector tree-model.

Single concern: define the one boundary the tree-model crosses to
fetch its per-:class:`FunctionNode` line-source. The tree-model never
sees a :class:`BinarySession`, a dataset, an FTL CSV, or a
:class:`BatchDecodeResult` -- only this Protocol and the typed
dataclasses it returns.

Both concrete backends (FtlBackend, BatchDecodeBackend; landed in
sibling subpackages by later Wave-5 phases) implement
:class:`RenderBackend`. The factory Protocol (:class:`BackendFactory`)
owns construction + lifetime; one :class:`RenderBackend` instance
exists per ``FunctionNode.expand`` call.

Failure model (per plan section 4): any method raises on data-
integrity violation. Empty result = empty sequence, never ``None``.
Only sentinels are:

* :attr:`InlineCallEntry.callee_section_pointer` is ``None``
  ("non-expandable callee" -- the UI hides expansion).
* :attr:`InlineCallEntry.variant_idx` equals
  :data:`MISSING_VARIANT_INDEX` ("no per-variant pin" -- caller
  arbitrates which variant of the callee to open).

After :meth:`RenderBackend.close`, every subsequent method call MUST
raise :class:`RuntimeError`; the contract is enforced by each concrete
backend. :attr:`RenderBackend.closed` is the observable flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.metadata_loader import SectionKind


__all__ = [
    "AsmLine",
    "BackendFactory",
    "FunctionHandle",
    "InlineCallEntry",
    "InlineJumpEntry",
    "LineItem",
    "RenderBackend",
    "RenderedBlock",
    "RenderedVariant",
]


# ---------------------------------------------------------------------------
# Function-handle: typed coordinate replacing raw (arm, idx) attribute pairs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionHandle:
    """Typed coordinate for one function across backends.

    ``arm`` is the canonical :class:`SectionKind` -- ``MATCHED`` for
    FTL discovery (no MATCHED/UNMATCHED distinction at the CSV layer
    per plan decision 24/25); either ``MATCHED`` or ``UNMATCHED`` for
    BatchDecodeBackend (driven by ``dataset.matched_func_names`` /
    ``dataset.unmatched_func_names``).

    ``idx`` is the per-arm index used by the underlying data source
    (lockstep position for FTL, dataset position for BatchDecode).

    ``name`` is the display name -- carried alongside the indices so
    consumers (notably :meth:`InlineCallNode.expand` spawning a
    callee :class:`FunctionNode`) never round-trip through a name
    lookup table.
    """

    arm: SectionKind
    idx: int
    name: str


# ---------------------------------------------------------------------------
# Per-variant + per-block metadata (returned by ``variants()`` / ``blocks()``)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedVariant:
    """Per-variant metadata for tree-row labels.

    ``label_axes`` is the flattened positional-axis mapping in the
    canonical ``POSITIONAL_PREFIXES`` order ``(arch, comp, cver,
    opt)`` -- both backends emit the same shape so
    :func:`_label.variant_label` reads only this Mapping (plan
    decision 1). Backends MUST wrap their underlying dict as
    :class:`types.MappingProxyType` so callers cannot mutate the
    frozen-dataclass field (plan decision 21).

    ``variant_idx`` is the backend-internal variant index threaded
    back into :meth:`RenderBackend.blocks` /
    :meth:`RenderBackend.render_block`.
    """

    variant_idx: int
    label_axes: Mapping[str, Optional[str]]


@dataclass(frozen=True)
class RenderedBlock:
    """Per-block metadata for tree-row labels.

    ``preview`` carries the raw asm-text head WITHOUT any UI
    truncation -- the UI layer (:func:`_label.block_preview`) owns
    the length policy (plan section 3, ``_label.py`` row).

    ``block_idx`` is the backend-internal block index threaded back
    into :meth:`RenderBackend.render_block`.
    """

    block_idx: int
    preview: str


# ---------------------------------------------------------------------------
# Per-line items (the unit consumed by ``_tree_model``'s BlockNode)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AsmLine:
    """A plain assembly-like line emitted for one instruction.

    Numeric tokens render here too (hex form ``"<basename>:<bits>"``
    per plan decision 18). The discriminator between asm / call / jump
    is the dataclass type, not a string prefix.
    """

    text: str


@dataclass(frozen=True)
class InlineCallEntry:
    """One inline call site under a block.

    Fields mirror the legacy :mod:`tokenizer.inspector._render`
    definition unchanged so backends + tree-model agree on shape.

    ``counter_id`` is the encoder's per-Category counter (= position
    of the K-th call_target of ``kind``). ``callee_section_pointer``
    is the :class:`SectionPointerSpec` the tree model can hand to
    :func:`batch_decode` for expansion; ``None`` for ext calls, for
    LOCAL/PLT call_targets whose ``function_section_ptr`` did not
    resolve, and for every Phase-1 BatchDecodeBackend entry (plan
    decision 10). ``variant_idx`` equals :data:`MISSING_VARIANT_INDEX`
    when no per-variant pin exists (EXTERN, FtlBackend per decision
    24, Phase-1 BatchDecodeBackend, or callee lacks a vkey match).
    ``kind`` is the wire-format :class:`CallTargetType`; the
    rendering layer routes per-kind label words off this enum so no
    string-typed discriminator crosses the boundary. ``provider`` is
    the library name appended after ``@`` for EXTERN rows; ``None``
    for LOCAL/PLT and for EXTERN rows whose provider is unknown.
    """

    kind: CallTargetType
    counter_id: int
    callee_name: str
    callee_section_pointer: Optional[SectionPointerSpec]
    variant_idx: int
    provider: Optional[str]


@dataclass(frozen=True)
class InlineJumpEntry:
    """One within-function jump target referenced by an instruction.

    ``target_block_idx`` is the block index the jump targets within
    the SAME variant -- always a valid index into the variant's
    :meth:`RenderBackend.blocks` result.
    """

    target_block_idx: int


LineItem = Union[AsmLine, InlineCallEntry, InlineJumpEntry]


# ---------------------------------------------------------------------------
# RenderBackend Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RenderBackend(Protocol):
    """Per-:class:`FunctionNode`-open instance.

    Owns its own decode/parse state. Constructed via
    :meth:`BackendFactory.make`; one instance per
    ``FunctionNode.expand`` call (including synthetic callee nodes
    spawned by ``InlineCallNode.expand``). On collapse, the parent
    :class:`BackendFactory` holds NO ref to the instance; descendant
    nodes (VariantNode / BlockNode / InlineJumpNode / InlineCallNode)
    hold a ref via a single ``backend: RenderBackend`` field. GC'd on
    collapse + re-expand. NO process-global memoisation.

    Calls after :meth:`close` MUST raise :class:`RuntimeError`. The
    factory's :meth:`BackendFactory.close` cascades to live backend
    instances IF the descendant tree still holds refs; ordinarily
    the tree-model collapses descendants first.
    """

    @property
    def handle(self) -> FunctionHandle:
        """Typed ``(arm, idx, name)`` coordinate of this backend's function."""
        ...

    @property
    def closed(self) -> bool:
        """``True`` after :meth:`close`; subsequent method calls raise."""
        ...

    def variants(self) -> Sequence[RenderedVariant]:
        """Lazy enumeration of variants for this function.

        Metadata only -- MUST NOT parse FTL records or walk
        ``batch_decode`` rows. Backends MAY cache the result; result
        is invalidated on :meth:`close`.
        """
        ...

    def blocks(self, variant_idx: int) -> Sequence[RenderedBlock]:
        """Lazy enumeration of blocks for the given variant.

        First call per ``variant_idx`` triggers that variant's parse;
        idempotent (instance-cached). The cache is invalidated on
        :meth:`close`. Per the cache contract: a subsequent
        :meth:`render_block` MUST NOT re-walk this variant.
        """
        ...

    def render_block(
        self, variant_idx: int, block_idx: int
    ) -> Iterable[LineItem]:
        """Materialise the per-block line-item stream.

        Returns an :class:`Iterable` (not a list) so consumers cannot
        mutate cached state. Re-callable on the same coordinates.
        Raises on data-integrity violation; the UI's central
        dispatcher (``_app.py``) catches and renders the error row.
        """
        ...

    def close(self) -> None:
        """Release backend resources. Idempotent.

        Sets :attr:`closed` to ``True``; subsequent
        :meth:`variants` / :meth:`blocks` / :meth:`render_block`
        calls raise ``RuntimeError(f"{type(self).__name__} closed")``.
        """
        ...


# ---------------------------------------------------------------------------
# BackendFactory Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BackendFactory(Protocol):
    """Per-binary factory: owns discovery + shared backend resources.

    Mirrors the shape of :class:`MetadataLookup` in
    :mod:`tokenizer.disasm` -- a Protocol, not a closure-bag dataclass
    (plan decision 22). Concrete implementations are private to their
    opener modules under :mod:`tokenizer.inspector._backend_factory`.

    ``handles`` is the sorted, deterministic list of functions the
    UI seeds into the tree (``_app.compose`` iterates it). For
    FtlBackend it is the lockstep-records order; for BatchDecodeBackend
    it is ``dataset.matched_func_names`` order.
    """

    handles: Sequence[FunctionHandle]

    def make(self, handle: FunctionHandle) -> RenderBackend:
        """Open a fresh :class:`RenderBackend` for one function.

        Called per ``FunctionNode.expand``; backends are NOT cached
        across collapse + re-expand.
        """
        ...

    def close(self) -> None:
        """Release factory-owned shared state (vocab cache, session, ...).

        Idempotent. Live backend instances cascade-close via their
        own refs to factory-owned state IF the descendant tree still
        holds them.
        """
        ...
