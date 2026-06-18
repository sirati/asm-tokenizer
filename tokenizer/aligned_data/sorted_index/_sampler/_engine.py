"""Selectable per-binary-group decode engine for the cross-binary batch.

Single concern: *which decode engine turns the grouped per-binary section
pointers into per-binary* :class:`BatchDecodeResult` *rows that
* :func:`._concat._concat_results` *stitches.* The grouping (by
``binary_name``) and the cross-binary concatenation are owned by
:func:`._batch.decode_pointer_batch` and are SHARED across engines; this
module owns only the per-engine "grouped pointers -> per-binary results"
step, so the two engines feed the IDENTICAL concat assembly.

Two engines, ONE seam:

* :attr:`DecodeEngine.BATCH_DECODE` (default) -- the staged
  deferred-dispatch path: one shared
  :class:`BucketedRunLengthCollector` spans every per-binary Stage 1
  walk, ONE flush amortises the pow2-bucketed ``run_lengths`` dispatch
  across the whole batch, then every :class:`PendingBatchDecode` is
  finalised. This is the byte-for-byte historical
  :func:`decode_pointer_batch` decode.
* :attr:`DecodeEngine.VECTOR_BATCH` -- the geometry-first path: each
  per-binary group runs :func:`vector_batch_tokens` against that
  binary's :class:`VectorBatchArmSet`, and the returned
  :class:`VectorBatchResult` is adapted (:func:`_as_batch_decode_result`)
  into the SAME :class:`BatchDecodeResult` shape the concat consumes.
  vector_batch is geometry-self-contained (no shared collector), so this
  engine does its own per-binary prepass + scatter and never touches the
  collector / flush machinery.

Byte-identity contract: both engines are driven from the SAME shared
``rng`` in the SAME alphabetical ``binary_name`` order, so the
per-binary samples (the arm-agnostic ``resolve_section_pointers`` +
``compute_batch_idx_mapping`` draws, reused verbatim by vector_batch)
are identical draw-for-draw. With backfill OFF (the only mode here) the
two engines assemble byte-identical ``tokens`` / sidecars per binary,
and the shared concat therefore produces a byte-identical
:class:`MultiBinaryBatchDecodeResult`.

Module boundary -- what the caller sees: a caller selects an engine via
the :class:`DecodeEngine` enum and (only for VECTOR_BATCH) supplies a
``handle_provider`` callable ``binary_name -> VectorBatchArmSet`` that
owns the per-binary handle lifetime exactly the way ``sessions`` owns
session lifetime. The caller never imports vector_batch internals; the
provider hands back an opaque handle bundle this module routes to the
vector_batch entry. For BATCH_DECODE the provider is unused.
"""

from __future__ import annotations

from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from tokenizer.aligned_data.loader.batch_decode._entry import (
    PendingBatchDecode,
    batch_decode,
)
from tokenizer.aligned_data.loader.batch_decode._types import (
    BatchDecodeResult,
    SectionPointerSpec,
    VariantPadding,
)
from tokenizer.aligned_data.loader.decoded._bucketed_run_lengths import (
    BucketedRunLengthCollector,
)
from tokenizer.aligned_data.loader.session import BinarySession

from .._types import PerBinaryDecodeResult

# vector_batch imports sorted_index (``_graph_lengths._adjacency``), so a
# module-level import of the vector_batch entry here forms a cycle
# (sorted_index -> _sampler._engine -> vector_batch -> sorted_index). The
# runtime call ``vector_batch_tokens`` is imported lazily inside
# :func:`_decode_groups_vector_batch` (by call time both packages are
# fully initialised); ``VectorBatchResult`` is annotation-only (the file
# uses ``from __future__ import annotations``) so it is needed only by
# type-checkers.
if TYPE_CHECKING:
    from tokenizer.aligned_data.loader.vector_batch._entry import (
        VectorBatchResult,
    )


__all__ = [
    "DecodeEngine",
    "VectorBatchHandleProvider",
    "decode_groups",
]


class DecodeEngine(Enum):
    """Which decode engine the cross-binary batch routes each group through.

    * :attr:`BATCH_DECODE` -- the staged collector/flush/finalise path
      (default; the byte-for-byte historical decode).
    * :attr:`VECTOR_BATCH` -- the geometry-first per-binary path, adapted
      to the same :class:`BatchDecodeResult` shape so the cross-binary
      concat is unchanged. Requires a ``handle_provider``.
    """

    BATCH_DECODE = "batch_decode"
    VECTOR_BATCH = "vector_batch"


#: A caller-owned lazy provider of one binary's both-arms vector_batch
#: handle bundle (a ``VectorBatchArmSet``), keyed by ``binary_name``. The
#: provider owns the handle lifetime (open lazily, hold, close on
#: teardown) the same way the ``sessions`` mapping owns session lifetime;
#: this module only reads from it. Typed as ``Callable[[str], object]``
#: because the handle bundle is opaque to this seam -- it is threaded
#: straight to ``vector_batch_tokens`` and never inspected here.
VectorBatchHandleProvider = Callable[[str], object]


#: Per-binary ``max_depth``: each binary maps to a scalar ``int`` (the
#: historical path) or a per-section-pointer ``int`` array (the
#: cross-depth path). The grouping into this shape is owned by
#: :func:`._batch.decode_pointer_batch`; this module only routes each
#: group's value to its engine.
PerBinaryMaxDepth = Mapping[str, Union[int, np.ndarray]]


def decode_groups(
    sessions: Mapping[str, BinarySession],
    per_binary_pointers: Mapping[str, List[SectionPointerSpec]],
    *,
    engine: DecodeEngine,
    context_len: int,
    num_variants_per_section: int,
    max_depth: PerBinaryMaxDepth,
    rng: np.random.Generator,
    variant_padding: VariantPadding,
    inlined_equivalent_call_targets_only: bool,
    include_fid_sidecar: bool,
    handle_provider: Optional[VectorBatchHandleProvider] = None,
) -> List[PerBinaryDecodeResult]:
    """Decode the per-binary groups into per-binary results via ``engine``.

    Iterates ``per_binary_pointers`` in ``sorted(name)`` order (the
    canonical alphabetical order the concat + ``binary_id_per_row``
    numbering rely on) and returns a :class:`PerBinaryDecodeResult` per
    binary in that order (each carrying the binary's
    :class:`BatchDecodeResult` AND its per-row source depth), ready for
    :func:`._concat._concat_results`.

    The grouping itself is the caller's concern; this function only
    decodes the already-grouped pointers. Both engines are driven from
    the SAME shared ``rng`` in the SAME order so their per-binary samples
    are byte-identical draw-for-draw.

    ``max_depth`` is the per-binary mapping (:data:`PerBinaryMaxDepth`):
    each group's value is threaded to its own decode. A per-section-
    pointer array value is VECTOR_BATCH-only; the BATCH_DECODE engine
    raises :class:`NotImplementedError` for it.

    Raises
    ------
    ValueError
        When a group names a binary absent from ``sessions``.
    ValueError
        When ``engine`` is VECTOR_BATCH but no ``handle_provider`` is
        supplied.
    NotImplementedError
        When ``engine`` is BATCH_DECODE but a group's ``max_depth`` is a
        per-pointer array (cross-depth is a vector_batch-only feature).
    """
    if engine is DecodeEngine.BATCH_DECODE:
        return _decode_groups_batch_decode(
            sessions,
            per_binary_pointers,
            context_len=context_len,
            num_variants_per_section=num_variants_per_section,
            max_depth=max_depth,
            rng=rng,
            variant_padding=variant_padding,
            inlined_equivalent_call_targets_only=(
                inlined_equivalent_call_targets_only
            ),
            include_fid_sidecar=include_fid_sidecar,
        )
    if handle_provider is None:
        raise ValueError(
            "decode_groups: engine=VECTOR_BATCH requires a handle_provider",
        )
    return _decode_groups_vector_batch(
        sessions,
        per_binary_pointers,
        context_len=context_len,
        num_variants_per_section=num_variants_per_section,
        max_depth=max_depth,
        rng=rng,
        variant_padding=variant_padding,
        include_fid_sidecar=include_fid_sidecar,
        handle_provider=handle_provider,
    )


def _require_session(
    sessions: Mapping[str, BinarySession], binary_name: str
) -> BinarySession:
    """Look up ``binary_name``'s open session or raise (a missing session
    is a hard caller error, not a skip)."""
    if binary_name not in sessions:
        raise ValueError(
            "decode_groups: no open session for binary "
            f"{binary_name!r}",
        )
    return sessions[binary_name]


def _scalar_depth(max_depth: PerBinaryMaxDepth, binary_name: str) -> int:
    """The scalar depth for ``binary_name``, or reject a per-pointer array.

    The staged BATCH_DECODE engine has no per-row depth seam, so a
    per-pointer array value is a hard NotImplementedError (cross-depth is
    vector_batch-only).
    """
    value = max_depth[binary_name]
    if np.asarray(value).ndim != 0:
        raise NotImplementedError(
            "DecodeEngine.BATCH_DECODE does not support a per-section-pointer "
            "max_depth array; cross-depth decoding is VECTOR_BATCH-only",
        )
    return int(value)


def _decode_groups_batch_decode(
    sessions: Mapping[str, BinarySession],
    per_binary_pointers: Mapping[str, List[SectionPointerSpec]],
    *,
    context_len: int,
    num_variants_per_section: int,
    max_depth: PerBinaryMaxDepth,
    rng: np.random.Generator,
    variant_padding: VariantPadding,
    inlined_equivalent_call_targets_only: bool,
    include_fid_sidecar: bool,
) -> List[PerBinaryDecodeResult]:
    """Staged collector/flush/finalise decode (the historical path).

    One shared :class:`BucketedRunLengthCollector` spans every per-binary
    Stage 1 walk; ONE flush amortises the pow2-bucketed ``run_lengths``
    dispatch across every call_target row in the whole batch; every
    pending decode is then finalised against the flushed run-length
    results.

    This engine is single-depth per binary (the staged path has no
    per-row depth seam), so each binary's ``depth_per_row`` is its scalar
    depth broadcast over its rows.
    """
    collector = BucketedRunLengthCollector()
    pending_decodes: List[Tuple[str, int, PendingBatchDecode]] = []
    for binary_name in sorted(per_binary_pointers):
        session = _require_session(sessions, binary_name)
        scalar_depth = _scalar_depth(max_depth, binary_name)
        pending = batch_decode(
            session,
            per_binary_pointers[binary_name],
            num_variants_per_section=num_variants_per_section,
            context_len=context_len,
            max_depth=scalar_depth,
            variant_padding=variant_padding,
            inlined_equivalent_call_targets_only=(
                inlined_equivalent_call_targets_only
            ),
            include_fid_sidecar=include_fid_sidecar,
            keep_intermediate=False,
            rng=rng,
            collector=collector,
        )
        pending_decodes.append((binary_name, scalar_depth, pending))

    runlen_results = collector.flush()
    out: List[PerBinaryDecodeResult] = []
    for binary_name, scalar_depth, pending in pending_decodes:
        result = pending.finalise(runlen_results)
        out.append(
            PerBinaryDecodeResult(
                binary_name=binary_name,
                result=result,
                depth_per_row=np.full(
                    result.tokens.shape[0], scalar_depth, dtype=np.int64
                ),
            )
        )
    return out


def _decode_groups_vector_batch(
    sessions: Mapping[str, BinarySession],
    per_binary_pointers: Mapping[str, List[SectionPointerSpec]],
    *,
    context_len: int,
    num_variants_per_section: int,
    max_depth: PerBinaryMaxDepth,
    rng: np.random.Generator,
    variant_padding: VariantPadding,
    include_fid_sidecar: bool,
    handle_provider: VectorBatchHandleProvider,
) -> List[PerBinaryDecodeResult]:
    """Geometry-first per-binary decode, adapted to the concat contract.

    Each group runs :func:`vector_batch_tokens` against the binary's
    handle bundle (from ``handle_provider``) on the SAME shared ``rng``,
    in the SAME alphabetical order the batch_decode engine uses, so the
    per-binary samples are byte-identical. The returned
    :class:`VectorBatchResult` is adapted to a :class:`BatchDecodeResult`
    so :func:`._concat._concat_results` consumes it unchanged; its
    per-row source depth is carried out verbatim on the
    :class:`PerBinaryDecodeResult` (this is the cross-depth engine, so the
    depth genuinely varies row to row).

    vector_batch is geometry-self-contained: there is no shared collector
    and no deferred flush; each per-binary decode is complete on return.
    """
    # Lazy import to break the sorted_index <-> vector_batch import cycle
    # (see the module-level note); safe at call time.
    from tokenizer.aligned_data.loader.vector_batch._entry import (
        vector_batch_tokens,
    )

    results: List[PerBinaryDecodeResult] = []
    for binary_name in sorted(per_binary_pointers):
        session = _require_session(sessions, binary_name)
        vb_result = vector_batch_tokens(
            session,
            per_binary_pointers[binary_name],
            handles=handle_provider(binary_name),
            num_variants_per_section=num_variants_per_section,
            context_len=context_len,
            max_depth=max_depth[binary_name],
            variant_padding=variant_padding,
            rng=rng,
            include_fid_sidecar=include_fid_sidecar,
        )
        results.append(
            PerBinaryDecodeResult(
                binary_name=binary_name,
                result=_as_batch_decode_result(vb_result),
                depth_per_row=np.asarray(vb_result.depth_per_row, dtype=np.int64),
            )
        )
    return results


def _as_batch_decode_result(vb: VectorBatchResult) -> BatchDecodeResult:
    """Adapt a :class:`VectorBatchResult` to the :class:`BatchDecodeResult`
    contract the cross-binary concat consumes.

    ``VectorBatchResult`` carries every field the concat reads (tokens,
    the ``(section_idx, variant_idx)`` mapping, the dense identity +
    number arrays + offsets, and the optional FID sidecars) byte-
    identically to ``batch_decode`` with backfill off. The fields the
    concat treats as all-or-none and that neither engine populates here
    -- the metatoken run-length sidecars
    (``block_runlength`` / ``insn_runlength`` + offsets) and the
    ``intermediate`` Stage3 batch -- are ``None``, matching the
    ``batch_decode`` engine (which is driven without
    ``emit_block_n_insns_runlength`` / ``keep_intermediate``). The concat
    then sees a consistent all-``None`` set across both engines.
    """
    return BatchDecodeResult(
        tokens=vb.tokens,
        identities=vb.identities,
        identity_row_offsets=vb.identity_row_offsets,
        numbers_significant=vb.numbers_significant,
        numbers_sign_exponent=vb.numbers_sign_exponent,
        number_row_offsets=vb.number_row_offsets,
        batch_idx_to_section_variant=vb.batch_idx_to_section_variant,
        fid_sidecar=vb.fid_sidecar,
        fid_row_offsets=vb.fid_row_offsets,
        fid_per_category_counts=vb.fid_per_category_counts,
        block_runlength=None,
        block_runlength_row_offsets=None,
        insn_runlength=None,
        insn_runlength_row_offsets=None,
        intermediate=None,
    )
