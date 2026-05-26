"""``FtlBackend`` -- the per-binary-CSV implementation of ``RenderBackend``.

Single concern: route :meth:`variants` / :meth:`blocks` /
:meth:`render_block` calls for one :class:`FunctionHandle` into the
shared :func:`tokenizer.inspector._render._render_block.render_block`
body, backed by the per-binary :class:`CsvIndex`.

The per-binary discovery + vocab cache + parsed-record cache live in
:class:`CsvIndex` (one instance per binary, owned by the
``_backend_factory``). The per-function instance below is
constructor-lightweight (no parse on init) and lazy on every method.
"""

from __future__ import annotations

import types
from typing import Dict, Iterable, List, Mapping, Optional

from tokenizer.inspector._label import block_preview_from_asm_texts
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    FunctionHandle,
    LineItem,
    RenderedBlock,
    RenderedVariant,
)
from tokenizer.inspector._render._render_block import render_block
from tokenizer.variant_info import VariantInfo
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
)

from ._block_header import block_header, body_block_view
from ._csv_index import CsvIndex
from ._variant_state import VariantState, build_variant_state


__all__ = ["FtlBackend"]


def _build_label_axes(info: VariantInfo) -> Mapping[str, Optional[str]]:
    """Pre-flatten the POSITIONAL_PREFIXES order ``(arch, comp, cver, opt)``.

    Wrapped as :class:`types.MappingProxyType` per plan decision 21 so
    callers cannot mutate the frozen :class:`RenderedVariant` field.
    """
    return types.MappingProxyType(
        {
            ARCH_PREFIX: info.arch,
            COMP_PREFIX: info.compiler,
            CVER_PREFIX: info.compiler_version,
            OPT_PREFIX: info.opt,
        }
    )


def _build_extra_metadata(info: VariantInfo) -> Mapping[str, str]:
    """Project :attr:`VariantInfo.extra_metadata` onto the string-valued
    frozen mapping :class:`RenderedVariant.extra_metadata` expects.

    Keys are emitted in sorted order (one bucket per unique value-set
    so the inspector's EXTRA_META axis grouping stays stable); list
    values comma-join, matching the BatchDecode backend's residue
    derivation in :meth:`VariantInfo.from_function_data_metadata`.
    """
    coerced: dict[str, str] = {}
    for key in sorted(info.extra_metadata):
        value = info.extra_metadata[key]
        if value is None:
            coerced[key] = ""
        elif isinstance(value, list):
            coerced[key] = ",".join(str(v) for v in value)
        else:
            coerced[key] = str(value)
    return types.MappingProxyType(coerced)


def _no_callee_arm(_function_section_ptr: int) -> None:
    """FtlBackend cannot resolve callees to a section pointer.

    Per plan section 5 / decision 24: FtlBackend has no cross-section
    nav (no batch_decode session backing it); every
    ``InlineCallEntry.callee_section_pointer`` is ``None``.
    """
    return None


class FtlBackend:
    """Per-:class:`FunctionHandle` :class:`RenderBackend` over FTL CSVs.

    Constructed by the factory with the shared per-binary
    :class:`CsvIndex` + the typed function handle. Lazy on every
    method: ``variants()`` materialises labels only,
    ``blocks(v)`` parses on first touch (cached per ``variant_idx``),
    ``render_block(v, b)`` reads the cached parse + delegates to the
    shared renderer.

    ``closed`` flips on :meth:`close`; subsequent calls raise per the
    Protocol contract.
    """

    def __init__(self, csv_index: CsvIndex, handle: FunctionHandle) -> None:
        self._csv_index = csv_index
        self._handle = handle
        self._variants_cache: Optional[List[RenderedVariant]] = None
        self._variant_states: Dict[int, VariantState] = {}
        self._blocks_cache: Dict[int, List[RenderedBlock]] = {}
        self._closed = False

    @property
    def csv_index(self) -> CsvIndex:
        return self._csv_index

    @property
    def handle(self) -> FunctionHandle:
        return self._handle

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------
    # Protocol API
    # ------------------------------------------------------------------

    def variants(self) -> List[RenderedVariant]:
        """Per-variant :class:`RenderedVariant` for slots that contain us.

        Walks ``csv_index.csv_paths`` and yields one
        :class:`RenderedVariant` per slot where
        :meth:`CsvIndex.has_variant` reports membership. Variant
        indices preserve their lockstep position (= position in
        ``csv_paths``) so :meth:`blocks` / :meth:`render_block` can
        re-key into the parsed-record cache.
        """
        self._raise_if_closed()
        if self._variants_cache is None:
            self._variants_cache = self._build_variants()
        return self._variants_cache

    def blocks(self, variant_idx: int) -> List[RenderedBlock]:
        """Per-block :class:`RenderedBlock`; triggers the variant parse.

        FTL records carry only function-body sections (no variant_tokens
        prefix + no per-CT self-prepend; the FTL stream STARTS at the
        function-body opening), so emitted entries are either
        :attr:`BlockKind.BODY` (basic blocks) or
        :attr:`BlockKind.JUMP_TABLE` (writer-emitted jump-table footer
        blocks). The variant_header + function_id sections are a
        BatchDecodeBackend concern.

        ``kind`` + ``block_idx`` are read from each block's
        ``[Block_Def, <BLOCK_V2|JUMP_TABLE>:N]`` opening pair via
        :func:`block_header` -- the sibling positional index only
        matches N for the simplest straight-line functions, and the
        UI label needs the authoritative N.

        The preview is sourced from the SAME render walk
        :meth:`render_block` returns on expand (:func:`render_block`
        from :mod:`._render_block`) so the asm text shown next to
        ``Block: <i>`` matches the items the row yields on expand --
        in particular the MEM-bracket / register-list display
        substitution (``mem[``->``[``, ``]mem``->``]``, ...) is applied
        through :func:`substitute_display_chars` exactly as the
        expanded body sees. The block is wrapped in a
        :class:`BodyBlockView` so the leading
        ``[Block_Def, <BLOCK_V2|JUMP_TABLE>:N]`` header pair stays
        absorbed (mirrors BatchDecode's ``pending_header`` latch).
        """
        self._raise_if_closed()
        if variant_idx in self._blocks_cache:
            return self._blocks_cache[variant_idx]
        state = self._ensure_variant_state(variant_idx)

        rendered: List[RenderedBlock] = []
        for blk in state.blocks:
            section_kind, n = block_header(blk)
            rendered.append(
                RenderedBlock(
                    kind=section_kind,
                    block_idx=n,
                    preview=self._block_preview(state, blk),
                )
            )
        self._blocks_cache[variant_idx] = rendered
        return rendered

    @staticmethod
    def _block_preview(state: VariantState, blk) -> str:
        """Render ``blk`` through the shared row walker and join the
        :class:`AsmLine` texts for the preview.

        Same call shape as :meth:`render_block` so both paths emit
        identical AsmLine text streams -- the preview is GUARANTEED
        to match what the user sees on expand, including the
        :func:`substitute_display_chars` substitutions that turn the
        MEM-bracket / register-list vocab strings into the polished
        display chars (``mem[``->``[``, ``]mem``->``]``, ...).
        """
        lines = render_block(
            block=body_block_view(blk),
            section=state.view,
            kind_to_called_idx=state.kind_to_called_idx,
            variant_pins={},
            line_to_name=state.line_to_name,
            line_to_provider=state.line_to_provider,
            callee_arm_resolver=_no_callee_arm,
        )
        return block_preview_from_asm_texts(
            line.text for line in lines if isinstance(line, AsmLine)
        )

    def render_block(
        self, variant_idx: int, kind: BlockKind, block_idx: int
    ) -> Iterable[LineItem]:
        """Materialise the line-item stream for ``(variant, kind, block)``.

        FTL produces :attr:`BlockKind.BODY` + :attr:`BlockKind.JUMP_TABLE`
        sections; any other ``kind`` lands on a :class:`KeyError` so
        callers don't accidentally render an FTL "variant header" /
        "function id" section that doesn't exist in the FTL stream
        layer.

        ``(kind, block_idx)`` together address one block: the kind
        discriminates which identity namespace ``N`` lives in (body
        blocks vs jump-table footers), the index is the encoded N. We
        look up the matching block by ``(kind, N)`` rather than by
        sibling position so callers and jump targets address blocks
        in the same coordinate space. The block is wrapped in a
        :class:`BodyBlockView` so the renderer's per-instruction walk
        skips the opening header pair.
        """
        self._raise_if_closed()
        if kind not in (BlockKind.BODY, BlockKind.JUMP_TABLE):
            raise KeyError(
                f"FtlBackend.render_block: kind={kind!r} not "
                f"supported (FTL emits BODY + JUMP_TABLE sections only)"
            )
        state = self._ensure_variant_state(variant_idx)
        block = self._block_for_header(state, kind, block_idx)
        # FtlBackend has no encoder-side per-call pin data (the CSV
        # stream carries no per_call_entries); every InlineCallEntry
        # emits with ``variant_idx == MISSING_VARIANT_INDEX`` so
        # :meth:`InlineCallNode.expand` falls through to the
        # all-variants surface.
        return render_block(
            block=body_block_view(block),
            section=state.view,
            kind_to_called_idx=state.kind_to_called_idx,
            variant_pins={},
            line_to_name=state.line_to_name,
            line_to_provider=state.line_to_provider,
            callee_arm_resolver=_no_callee_arm,
        )

    @staticmethod
    def _block_for_header(
        state: VariantState, kind: BlockKind, block_idx: int
    ):
        """Locate the block whose header matches ``(kind, block_idx)``.

        Linear scan keeps the lookup self-describing (the cached
        ``state.blocks`` is the single source of truth); typical block
        counts per function are small enough that an index dict would
        add complexity without measurable benefit. Raises
        :class:`KeyError` on miss so a stale jump-target N surfaces
        with the same diagnostic shape as the BatchDecodeBackend.
        """
        for blk in state.blocks:
            blk_kind, n = block_header(blk)
            if blk_kind is kind and n == block_idx:
                return blk
        have = [block_header(b) for b in state.blocks]
        raise KeyError(
            f"FtlBackend.render_block: no block with header "
            f"({kind.name}, {block_idx}) (have {have!r})"
        )

    def close(self) -> None:
        """Drop per-variant caches. Idempotent.

        The per-binary :class:`CsvIndex` is factory-owned and outlives
        this backend; we never close it here.
        """
        if self._closed:
            return
        self._variant_states.clear()
        self._blocks_cache.clear()
        self._variants_cache = None
        self._closed = True

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} closed")

    def _build_variants(self) -> List[RenderedVariant]:
        rendered: List[RenderedVariant] = []
        for variant_idx, csv_path in enumerate(self.csv_index.csv_paths):
            if not self.csv_index.has_variant(self._handle.idx, variant_idx):
                continue
            info = VariantInfo.from_csv(csv_path)
            rendered.append(
                RenderedVariant(
                    variant_idx=variant_idx,
                    label_axes=_build_label_axes(info),
                    extra_metadata=_build_extra_metadata(info),
                    variant_identity=info.identity,
                )
            )
        return rendered

    def _ensure_variant_state(self, variant_idx: int) -> VariantState:
        cached = self._variant_states.get(variant_idx)
        if cached is not None:
            return cached
        record = self.csv_index.parsed_record_for(self._handle.idx, variant_idx)
        if record is None:
            raise ValueError(
                f"variant_idx={variant_idx} has no parsed record for "
                f"function {self._handle.name!r} (handle.idx={self._handle.idx})"
            )
        csv_path = self.csv_index.csv_paths[variant_idx]
        vocab = self.csv_index.vocab_for(csv_path)
        if vocab is None:
            raise RuntimeError(
                f"vocab failed to load for {csv_path}; cannot render "
                f"function {self._handle.name!r}"
            )
        state = build_variant_state(record, vocab)
        self._variant_states[variant_idx] = state
        return state
