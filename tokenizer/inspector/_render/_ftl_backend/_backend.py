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

from tokenizer.inspector._render._protocol import (
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

        FTL records carry only basic-block bodies (no variant_tokens
        prefix + no per-CT self-prepend; the FTL stream STARTS at the
        function-body opening), so every emitted entry is
        :attr:`BlockKind.BODY`. The variant_header + function_id
        sections are a BatchDecodeBackend concern.
        """
        self._raise_if_closed()
        if variant_idx in self._blocks_cache:
            return self._blocks_cache[variant_idx]
        state = self._ensure_variant_state(variant_idx)
        from tokenizer.inspector._label import block_preview

        rendered = [
            RenderedBlock(
                kind=BlockKind.BODY,
                block_idx=i,
                preview=block_preview(blk),
            )
            for i, blk in enumerate(state.blocks)
        ]
        self._blocks_cache[variant_idx] = rendered
        return rendered

    def render_block(
        self, variant_idx: int, kind: BlockKind, block_idx: int
    ) -> Iterable[LineItem]:
        """Materialise the line-item stream for ``(variant, kind, block)``.

        FTL only produces :attr:`BlockKind.BODY` sections; any other
        ``kind`` lands on a :class:`KeyError` so callers don't
        accidentally render an FTL "variant header" / "function id"
        section that doesn't exist in the FTL stream layer.
        """
        self._raise_if_closed()
        if kind is not BlockKind.BODY:
            raise KeyError(
                f"FtlBackend.render_block: kind={kind!r} not "
                f"supported (FTL emits BODY sections only)"
            )
        state = self._ensure_variant_state(variant_idx)
        block = state.blocks[block_idx]
        return render_block(
            block=block,
            section=state.view,
            kind_to_called_idx=state.kind_to_called_idx,
            variant_pins={},
            line_to_name=state.line_to_name,
            line_to_provider=state.line_to_provider,
            callee_arm_resolver=_no_callee_arm,
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
