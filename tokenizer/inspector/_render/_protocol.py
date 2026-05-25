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

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from tokenizer.aligned_data.call_target_type import CallTargetType
from tokenizer.aligned_data.loader.batch_decode._types import SectionPointerSpec
from tokenizer.aligned_data.loader.decoded._number_render import (
    InlineNumberPrecisionEntry,
)
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.matched_sections_bin import MISSING_VARIANT_INDEX
from tokenizer.tokens import TokenType
from tokenizer.variant_info import VariantIdentity


__all__ = [
    "AsmLine",
    "BackendFactory",
    "BlockKind",
    "FunctionHandle",
    "InlineCallEntry",
    "InlineJumpEntry",
    "InlineNumberPrecisionEntry",
    "LineItem",
    "Openable",
    "RenderBackend",
    "RenderedBlock",
    "RenderedVariant",
    "VariantIdentity",
]


# ---------------------------------------------------------------------------
# Section discriminator -- variant-header / function-id / body
# ---------------------------------------------------------------------------


class BlockKind(Enum):
    """Discriminator for the three :class:`RenderedBlock` section kinds.

    Variant-level layout (BatchDecodeBackend, the only producer that
    splits all three):

    * :attr:`VARIANT_HEADER` -- the variant_tokens prefix
      (``arch:/comp:/cver:/opt:`` rows). Spans cols ``[0, n_axis)``.
      ``block_idx == -1`` (sentinel: not a body block).
    * :attr:`FUNCTION_ID` -- the row-assembler-owned LOCAL_FUNC
      self-prepend slot (the function's own identity reference). One
      column at ``col == n_axis``. ``block_idx == -1``.
    * :attr:`BODY` -- one section per basic block. Spans the post-
      header-pair content of the block; the ``Block_Def`` + ``block_v2``
      header pair is consumed silently by the walker so the section
      content starts at the first real instruction. ``block_idx`` is
      the block index encoded by the consumed header.

    FtlBackend emits only :attr:`BODY` sections (no variant_tokens /
    self-prepend at the FTL stream layer).
    """

    VARIANT_HEADER = "variant_header"
    FUNCTION_ID = "function_id"
    BODY = "body"


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

    ``extra_metadata`` carries the per-variant non-axis metadata
    residue (sidecar fields, build-flag groups, hardening / sanitizer
    settings, ...) as a frozen string mapping. Both backends derive it
    via :meth:`tokenizer.variant_info.VariantInfo.from_function_data_metadata`
    or by projecting the variant's :class:`VariantInfo.extra_metadata`
    onto a string mapping — the inspector keys its EXTRA_META axis
    grouping on this field. Wrapped as :class:`types.MappingProxyType`
    per plan decision 21.

    ``variant_identity`` is the typed canonical-identity
    (:class:`tokenizer.variant_info.VariantIdentity`) shared with the
    rest of the codebase via :class:`VariantInfo.__eq__` /
    :class:`VariantInfo.__hash__`. The inspector's expand-state
    preservation keys on this value across grouping rebuilds — using a
    hand-rolled tuple would collide on the canonical-4 across variants
    that differ only in :attr:`VariantIdentity.variant_id` (the
    dedup-disambiguator documented at
    ``tokenizer.aligned_data.io.write_matched_section_csv``).
    """

    variant_idx: int
    label_axes: Mapping[str, Optional[str]]
    extra_metadata: Mapping[str, str]
    variant_identity: VariantIdentity


@dataclass(frozen=True)
class RenderedBlock:
    """Per-section metadata for tree-row labels.

    Three section kinds are discriminated by :attr:`kind`
    (:class:`BlockKind`): a BatchDecodeBackend variant exposes
    ``[VARIANT_HEADER, FUNCTION_ID, BODY*]`` so the variant_tokens
    prefix, LOCAL_FUNC self-prepend, and per-block body each get a
    semantically-correct tree section. FtlBackend emits only
    :attr:`BlockKind.BODY` entries.

    ``preview`` carries the raw asm-text head WITHOUT any UI
    truncation -- the UI layer (:func:`_label.block_preview`) owns
    the length policy (plan section 3, ``_label.py`` row).

    ``block_idx`` is the backend-internal block index threaded back
    into :meth:`RenderBackend.render_block`; ``-1`` for the non-body
    kinds (``VARIANT_HEADER`` / ``FUNCTION_ID``) where there is no
    block to address.
    """

    kind: BlockKind
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

    ``openables`` carries the per-instruction sidecar entries the
    tree-model lazily expands into child rows: inline call sites
    (:class:`InlineCallEntry`), intra-function jump targets
    (:class:`InlineJumpEntry`), and full-precision number expansions
    (:class:`InlineNumberPrecisionEntry`). Empty tuple = leaf row
    (no expansion). Discriminator is the dataclass type itself
    (``isinstance`` / ``match``); see :data:`Openable`. The field is
    a tuple (not list) so the frozen-dataclass immutability extends
    to the sidecar payload.
    """

    text: str
    openables: Tuple["Openable", ...] = field(default_factory=tuple)


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
    ``caller_variant_idx`` is the variant_idx of the row that emitted
    this entry — :class:`InlineCallNode.expand` falls back to it when
    :attr:`variant_idx` equals :data:`MISSING_VARIANT_INDEX` (e.g.
    Function-ID self-references, or callees whose vkey did not match)
    so the inline-call defaults to the caller's variant instead of
    surfacing the full variant list. :data:`MISSING_VARIANT_INDEX`
    here means "no caller pin known" (FtlBackend's Phase-1 default,
    test fixtures that don't thread it).
    """

    kind: CallTargetType
    counter_id: int
    callee_name: str
    callee_section_pointer: Optional[SectionPointerSpec]
    variant_idx: int
    provider: Optional[str]
    caller_variant_idx: int = MISSING_VARIANT_INDEX


@dataclass(frozen=True)
class InlineJumpEntry:
    """One within-function jump target referenced by an instruction.

    ``target_block_idx`` is the block index the jump targets within
    the SAME variant -- always a valid index into the variant's
    :meth:`RenderBackend.blocks` result.
    """

    target_block_idx: int


Openable = Union[InlineCallEntry, InlineJumpEntry, InlineNumberPrecisionEntry]
"""Sidecar entry attached to an :class:`AsmLine` for lazy expansion.

The tree-model dispatches on the dataclass identity (``isinstance`` /
``match``) -- there is intentionally NO ``OpenableKind`` enum and NO
``openable_kind`` property. One openable type -> one wrapper-node
type at the tree-model boundary (per integrated plan W3-2 W4-amended).
"""


LineItem = AsmLine
"""Post-R2 wire shape of one item in the
:meth:`RenderBackend.render_block` stream.

Narrowed to :class:`AsmLine` only (cluster #3 of the integrated plan
W3-2 W4-amended): inline call sites, jump targets, and number-
precision expansions now ride :attr:`AsmLine.openables` rather than
sibling top-level LineItems. The type alias is retained -- not
collapsed into ``AsmLine`` everywhere -- so the Protocol's
``render_block`` signature still reads as a stream of typed items
(open to future re-broadening if the stream ever carries a non-asm
row again).
"""


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
        self, variant_idx: int, kind: BlockKind, block_idx: int
    ) -> Iterable[LineItem]:
        """Materialise the per-section line-item stream.

        The ``(kind, block_idx)`` pair addresses one section produced
        by :meth:`blocks`. :attr:`BlockKind.BODY` sections use the
        real block index; :attr:`BlockKind.VARIANT_HEADER` and
        :attr:`BlockKind.FUNCTION_ID` use ``block_idx == -1`` (the
        kind discriminates between them so the sentinel never
        collides). Returns an :class:`Iterable` (not a list) so
        consumers cannot mutate cached state. Re-callable on the
        same coordinates. Raises on data-integrity violation; the
        UI's central dispatcher (``_app.py``) catches and renders
        the error row.
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
