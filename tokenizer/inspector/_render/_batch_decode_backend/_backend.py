"""BatchDecodeBackend -- :class:`RenderBackend` over flat ``BatchDecodeResult`` tensors.

Single concern: implement :class:`RenderBackend` for the memmap path.
The backend opens lazily -- the constructor stores its inputs and
defers the :func:`batch_decode` call to first
:meth:`variants` / :meth:`blocks` / :meth:`render_block` access.

Plan reference: ``inspector-render-backends.md`` §6 + decisions #8,
#19, #20, #22, #23, #29 + audits A-MED-4 / A-MED-5 / A-HIGH-3.

Lifetime: cheap constructor (stores refs only); first
``variants()``/``blocks()``/``render_block()`` call triggers
:meth:`_ensure_result` which runs :func:`batch_decode` once.
:meth:`close` drops the cached result + flips :attr:`closed`;
subsequent calls raise :class:`RuntimeError` (audit A-HIGH-3).
"""

from __future__ import annotations

import types
from typing import Callable, Iterable, List, Mapping, Optional, Sequence

from tokenizer.aligned_data.loader.batch_decode._auto_size import (
    compute_auto_sizes,
)
from tokenizer.aligned_data.loader.batch_decode._entry import batch_decode
from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
    Stage1Variant,
    VariantPadding,
)
from tokenizer.inspector._render._protocol import (
    AsmLine,
    BlockKind,
    FunctionHandle,
    LineItem,
    RenderedBlock,
    RenderedVariant,
)
from tokenizer.token_manager import VocabularyManager
from tokenizer.variant_info import VariantIdentity, VariantInfo
from tokenizer.variant_tokens.prefixes import (
    ARCH_PREFIX,
    COMP_PREFIX,
    CVER_PREFIX,
    OPT_PREFIX,
    POSITIONAL_PREFIXES,
)

from ._arch_prefix import arch_prefix_tuple
from ._fid_table import FidBaseTable
from ._row_walk import RowSection, render_row_blocks


__all__ = ["BatchDecodeBackend"]


# Positional-axis prefix -> :class:`VariantIdentity` field name. Used to
# project the typed identity onto the canonical ``label_axes`` Mapping
# both backends emit; the runtime metadata-key shape lives inside
# :meth:`VariantInfo.from_function_data_metadata` so this module never
# sees the wire-layer ``compilerversion`` key directly.
_AXIS_PREFIX_TO_IDENTITY_FIELD: dict[str, str] = {
    ARCH_PREFIX: "arch",
    COMP_PREFIX: "compiler",
    CVER_PREFIX: "compiler_version",
    OPT_PREFIX: "opt",
}
assert set(_AXIS_PREFIX_TO_IDENTITY_FIELD) == set(POSITIONAL_PREFIXES), (
    "_AXIS_PREFIX_TO_IDENTITY_FIELD must match POSITIONAL_PREFIXES"
)


def _project_variant(
    s1v: Stage1Variant,
) -> tuple[Mapping[str, Optional[str]], Mapping[str, str], VariantIdentity]:
    """Project a :class:`Stage1Variant` into the
    ``(label_axes, extra_metadata, variant_identity)`` triple
    :class:`RenderedVariant` needs.

    Single canonical step — both the inspector's POSITIONAL grouping
    axes and the EXTRA_META grouping axes derive from the same
    factory call. ``VariantInfo.from_function_data_metadata`` owns the
    structural-key strip list + canonical-axis extraction, so this
    backend stays free of any "everything-else" residue knowledge.
    """
    metadata = s1v.call_targets[0].function_data.metadata
    identity, extra_metadata = VariantInfo.from_function_data_metadata(metadata)
    label_axes: dict[str, Optional[str]] = {
        prefix: getattr(identity, _AXIS_PREFIX_TO_IDENTITY_FIELD[prefix])
        for prefix in POSITIONAL_PREFIXES
    }
    return types.MappingProxyType(label_axes), extra_metadata, identity


class BatchDecodeBackend:
    """RenderBackend over :func:`batch_decode`'s flat tensors.

    Constructor is lazy (audit A-MED-4): stores inputs, does no I/O.
    First :meth:`variants`/:meth:`blocks`/:meth:`render_block` access
    triggers :meth:`_ensure_result`. Per audit A-HIGH-3, :attr:`closed`
    is the observable flag and every public method raises
    :class:`RuntimeError` after :meth:`close`.
    """

    def __init__(
        self,
        session,
        vocab_manager: VocabularyManager,
        handle: FunctionHandle,
        line_to_name: Mapping[int, str],
        line_to_provider: Mapping[int, str],
        callee_arm_resolver: Callable[
            [int], Optional[SectionPointerSpec]
        ],
    ) -> None:
        self._session = session
        self._vocab_manager = vocab_manager
        self._handle = handle
        self._line_to_name = line_to_name
        self._line_to_provider = line_to_provider
        self._callee_arm_resolver = callee_arm_resolver
        self._closed: bool = False
        self._result: Optional[BatchDecodeResult] = None
        self._fid_table: Optional[FidBaseTable] = None
        self._variants_cache: Optional[List[RenderedVariant]] = None
        self._row_sections_by_variant: dict[int, List[RowSection]] = {}
        self._variant_row_index: dict[int, int] = {}

    # Public RenderBackend surface -----------------------------------------

    @property
    def handle(self) -> FunctionHandle:
        return self._handle

    @property
    def closed(self) -> bool:
        return self._closed

    def variants(self) -> Sequence[RenderedVariant]:
        self._assert_open()
        self._ensure_result()
        if self._variants_cache is None:
            self._variants_cache = self._build_variants()
        return self._variants_cache

    def blocks(self, variant_idx: int) -> Sequence[RenderedBlock]:
        """Per-section :class:`RenderedBlock` enumeration.

        Returns ``[VARIANT_HEADER, FUNCTION_ID, BODY[0], BODY[1], ...]``
        for every variant -- the per-row walker stamps each section
        with its :class:`BlockKind` discriminator so the UI layer can
        compose the right label for each tree row (Variant Header /
        Function ID / Block: N). The preview text is the first
        :class:`AsmLine`'s body within the section -- empty string
        when the section carries no AsmLines (e.g. the FUNCTION_ID
        section, which holds an :class:`InlineCallEntry` for the
        self-prepend, not an AsmLine).
        """
        self._assert_open()
        self._ensure_result()
        sections = self._row_sections_for_variant(variant_idx)
        return [
            RenderedBlock(
                kind=section.kind,
                block_idx=section.block_idx,
                preview=_preview_for_section(section),
            )
            for section in sections
        ]

    def render_block(
        self, variant_idx: int, kind: BlockKind, block_idx: int
    ) -> Iterable[LineItem]:
        """Materialise the items for one section.

        Dispatch is by ``(kind, block_idx)`` pair: BODY sections
        carry their real block index; the two non-body kinds
        (VARIANT_HEADER + FUNCTION_ID) share ``block_idx == -1`` and
        are disambiguated by ``kind``. Returns a tuple snapshot so
        consumers can't mutate the cached list (Iterable per
        Protocol).
        """
        self._assert_open()
        self._ensure_result()
        sections = self._row_sections_for_variant(variant_idx)
        for section in sections:
            if section.kind is kind and section.block_idx == block_idx:
                return tuple(section.items)
        raise KeyError(
            f"BatchDecodeBackend.render_block: no section "
            f"kind={kind!r} block_idx={block_idx} "
            f"for variant {variant_idx}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._result = None
        self._fid_table = None
        self._variants_cache = None
        self._row_sections_by_variant.clear()
        self._variant_row_index.clear()

    # Lazy load + per-variant walk ----------------------------------------

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} closed")

    def _ensure_result(self) -> None:
        if self._result is not None:
            return
        spec = SectionPointerSpec(arm=self._handle.arm, idx=self._handle.idx)
        sizing = compute_auto_sizes(self._session, [spec])
        self._result = batch_decode(
            self._session,
            [spec],
            num_variants_per_section=sizing.num_variants_per_section,
            context_len=sizing.context_len,
            max_depth=0,
            variant_padding=VariantPadding.PAD_NULL,
            include_fid_sidecar=True,
            keep_intermediate=True,
            emit_block_n_insns_runlength=True,
        )
        self._fid_table = FidBaseTable.from_result(self._result)

    def _build_variants(self) -> List[RenderedVariant]:
        """Project Stage 1's per-section variants into :class:`RenderedVariant` s.

        With one ``SectionPointerSpec`` (the constructor's handle) the
        Stage 1 batch has exactly one section; read its variants list and
        record the per-variant batch_idx -> row mapping for later
        :meth:`blocks` / :meth:`render_block` lookups.
        """
        assert self._result is not None
        stage1 = self._result.intermediate.stage2.stage1
        assert len(stage1.sections) == 1, (
            f"BatchDecodeBackend expects a single section pointer; "
            f"got {len(stage1.sections)}"
        )
        rendered: List[RenderedVariant] = []
        # PAD_NULL is hard-pinned at construction (see :meth:`_build`):
        # every Stage 1 variant has a non-``None`` batch_idx, so the
        # row mapping is total.
        for s1v in stage1.sections[0].variants:
            self._variant_row_index[s1v.variant_idx] = int(s1v.batch_idx)
            label_axes, extra_metadata, variant_identity = _project_variant(s1v)
            rendered.append(
                RenderedVariant(
                    variant_idx=s1v.variant_idx,
                    label_axes=label_axes,
                    extra_metadata=extra_metadata,
                    variant_identity=variant_identity,
                )
            )
        return rendered

    def _row_sections_for_variant(self, variant_idx: int) -> List[RowSection]:
        """Lazy per-variant walk; cached idempotently."""
        cached = self._row_sections_by_variant.get(variant_idx)
        if cached is not None:
            return cached
        # variants() populates _variant_row_index; idempotent.
        self.variants()
        row = self._variant_row_index.get(variant_idx)
        if row is None:
            raise KeyError(
                f"BatchDecodeBackend: unknown variant_idx {variant_idx}; "
                f"valid={sorted(self._variant_row_index)}"
            )
        stage1 = self._result.intermediate.stage2.stage1
        stage2_section = self._result.intermediate.stage2.sections[0]
        stage1_variant = stage1.sections[0].variants[variant_idx]
        stage2_variant = stage2_section.variants[variant_idx]
        n_axis = int(stage1_variant.variant_tokens.shape[0])
        # Per-CT slot budget the row writer actually emitted (post-cut).
        # surviving_token_count includes the row-assembler-owned self-
        # prepend slot at the CT's body start (col = ct_start) and the
        # function-body that follows (cols ct_start+1 .. ct_start+pcl).
        pcl = [
            int(ct.surviving_token_count) for ct in stage2_variant.call_targets
        ]
        # Per-CT call_targets_section: each Stage1CallTarget owns its
        # own table (root + every inlined callee). The FUNCTION-band
        # token resolver indexes into the CURRENT walking CT's table,
        # NOT the root section's table, because inlined-callee call
        # sites reference THEIR own table.
        call_targets_per_ct = [
            ct.call_targets_section for ct in stage1_variant.call_targets
        ]
        # Arch-prefix tuple for INSTR_REP display elision. Read the raw
        # ``arch`` value out of the runtime metadata dict directly so
        # the missing-key collapse to ``""`` (empty-tuple no-op for
        # backends that haven't plumbed the arch) stays separate from
        # the canonical-identity coercion (``None -> "unknown"``) that
        # :meth:`VariantInfo.from_function_data_metadata` applies.
        root_metadata = stage1_variant.call_targets[0].function_data.metadata
        arch_axis = root_metadata.get("arch")
        arch_prefixes = arch_prefix_tuple(
            "" if arch_axis is None else str(arch_axis)
        )
        walked = render_row_blocks(
            result=self._result,
            row=row,
            caller_variant_idx=variant_idx,
            n_axis=n_axis,
            partial_cut_lengths=pcl,
            call_targets_per_ct=call_targets_per_ct,
            vocab_manager=self._vocab_manager,
            fid_table=self._fid_table,
            line_to_name=self._line_to_name,
            line_to_provider=self._line_to_provider,
            callee_arm_resolver=self._callee_arm_resolver,
            arch_prefixes=arch_prefixes,
        )
        self._row_sections_by_variant[variant_idx] = walked
        return walked


def _preview_for_section(section: RowSection) -> str:
    """First :class:`AsmLine`'s text in the section, or empty string.

    Mirrors the FtlBackend's preview contract (the
    :func:`tokenizer.inspector._label.block_preview` helper truncates
    as the UI policy layer); this backend feeds the raw asm-text head
    so the truncation policy applies uniformly across backends. The
    FUNCTION_ID section commonly carries no AsmLines (its single
    entry is an :class:`InlineCallEntry` for the self-prepend), so
    the preview falls through to empty -- the UI labels that section
    with a fixed string, not a preview.
    """
    for item in section.items:
        if isinstance(item, AsmLine):
            return item.text
    return ""
