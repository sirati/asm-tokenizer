"""Concrete :class:`BackendFactory` implementations + CLI openers.

Single concern: own per-binary backend construction. The two factories
(:class:`_FtlBackendFactory` for FTL CSV input, :class:`_BatchDecodeBackendFactory`
for memmap input) implement the locked :class:`BackendFactory` Protocol
in :mod:`tokenizer.inspector._render._protocol`: a ``handles`` attribute
listing the seed :class:`FunctionHandle` s, ``make(handle) -> RenderBackend``
constructing one per :class:`FunctionNode` expand, and ``close()`` releasing
shared per-binary state.

The two ``make_*_factory`` openers wrap the per-binary discovery so the
CLI ``__main__`` constructs a factory without knowing the backend's
internal types -- the only string-typed dispatch lives in
``__main__._open_backend_factory``'s argparse-driven mutex branch.

Plan reference: ``inspector-render-backends.md`` section 8.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from tokenizer.aligned_data.loader.batch_decode._types import (
    SectionPointerSpec,
)
from tokenizer.aligned_data.loader.binary_dataset import BinaryDataset
from tokenizer.aligned_data.loader.metadata_loader import SectionKind
from tokenizer.aligned_data.loader.session import BinarySession
from tokenizer.aligned_data.loader.unified_vocab_gate import (
    load_and_validate_unified_vocab,
)

from ._batch_decode_backend import BatchDecodeBackend
from ._ftl_backend import FtlBackend
from ._ftl_backend._csv_index import CsvIndex
from ._protocol import BackendFactory, FunctionHandle, RenderBackend


__all__ = [
    "make_batch_decode_factory",
    "make_ftl_factory",
]


# ---------------------------------------------------------------------------
# FTL factory (per-binary CSV input)
# ---------------------------------------------------------------------------


@dataclass
class _FtlBackendFactory:
    """:class:`BackendFactory` over a per-binary :class:`CsvIndex`.

    Owns the shared per-binary discovery + per-CSV vocab cache; one
    :class:`FtlBackend` per :meth:`make` call. :meth:`close` cascades
    to :meth:`CsvIndex.close`.
    """

    csv_index: CsvIndex
    handles: Sequence[FunctionHandle]
    _closed: bool = False

    def make(self, handle: FunctionHandle) -> RenderBackend:
        if self._closed:
            raise RuntimeError("_FtlBackendFactory closed")
        return FtlBackend(self.csv_index, handle)

    def close(self) -> None:
        if self._closed:
            return
        self.csv_index.close()
        self._closed = True


def make_ftl_factory(csv_dir: Path, binary_name: str) -> BackendFactory:
    """Build the FTL factory for one binary in ``csv_dir``.

    :class:`CsvIndex` performs the per-binary CSV discovery + lockstep
    parse; ``handles`` is the dense, ordered list of functions surfaced
    by :meth:`CsvIndex.function_keys`. ``handle.arm`` is
    :data:`SectionKind.MATCHED` for every FTL row -- the CSV layer does
    not distinguish matched / unmatched (plan decision 24/25).
    """
    csv_index = CsvIndex(csv_dir, binary_name)
    keys = csv_index.function_keys()
    handles: List[FunctionHandle] = [
        FunctionHandle(arm=SectionKind.MATCHED, idx=idx, name=name)
        for idx, (name, _hash) in enumerate(keys)
    ]
    return _FtlBackendFactory(csv_index=csv_index, handles=handles)


# ---------------------------------------------------------------------------
# BatchDecode factory (per-binary memmap input)
# ---------------------------------------------------------------------------


@dataclass
class _BatchDecodeBackendFactory:
    """:class:`BackendFactory` over a per-binary memmap (open session).

    Wraps a :class:`BinaryDataset` + an entered :class:`BinarySession`;
    :meth:`make` constructs one :class:`BatchDecodeBackend` per
    :meth:`FunctionNode.expand` call. :meth:`close` exits the session
    (mirrors :class:`_FtlBackendFactory.close` which closes its
    :class:`CsvIndex`) so callers register only one shutdown hook.

    The factory also OWNS the per-session callee_arm_resolver closure;
    LOCAL / PLT call_targets resolve their ``function_section_ptr``
    through ``session._idx_for_section_offset`` to a
    :class:`SectionPointerSpec` (or ``None`` for cross-arm /
    missing-section / EXTERN). The closure is built once at factory
    construction time and shared across every backend instance it
    spawns -- the inspector never reaches into session internals.
    """

    dataset: BinaryDataset
    session: BinarySession
    handles: Sequence[FunctionHandle]
    _closed: bool = False

    def make(self, handle: FunctionHandle) -> RenderBackend:
        if self._closed:
            raise RuntimeError("_BatchDecodeBackendFactory closed")
        return BatchDecodeBackend(
            session=self.session,
            vocab_manager=self.dataset.vocab_manager,
            handle=handle,
            line_to_name=self.dataset.line_to_name,
            line_to_provider=self.dataset.line_to_provider,
            callee_arm_resolver=_make_callee_arm_resolver(self.session),
        )

    def close(self) -> None:
        if self._closed:
            return
        self.session.close()
        self._closed = True


def _make_callee_arm_resolver(
    session: BinarySession,
) -> Callable[[int], Optional[SectionPointerSpec]]:
    """Build the session-backed callee section-pointer resolver.

    Each invocation maps a ``function_section_ptr`` byte offset (from
    :class:`CallTarget.function_section_ptr`) to a
    :class:`SectionPointerSpec` ``(arm, idx)`` -- the same pair every
    :func:`batch_decode` request consumes for expansion. The two arms
    are probed in MATCHED-then-UNMATCHED order; the first hit wins.
    ``None`` is returned when neither arm resolves (cross-arm /
    missing-section pointer). EXTERN call_targets are handled at the
    walker (no body to inline); this closure is only consumed for
    LOCAL / PLT.
    """

    def resolve(function_section_ptr: int) -> Optional[SectionPointerSpec]:
        idx_matched = session._idx_for_section_offset(
            function_section_ptr, "matched"
        )
        if idx_matched is not None:
            return SectionPointerSpec(
                arm=SectionKind.MATCHED, idx=int(idx_matched)
            )
        idx_unmatched = session._idx_for_section_offset(
            function_section_ptr, "unmatched"
        )
        if idx_unmatched is not None:
            return SectionPointerSpec(
                arm=SectionKind.UNMATCHED, idx=int(idx_unmatched)
            )
        return None

    return resolve


def make_batch_decode_factory(
    memmap_dir: Path, binary_name: str
) -> BackendFactory:
    """Build the BatchDecode factory for one binary in ``memmap_dir``.

    The returned factory owns the entered :class:`BinarySession`;
    :meth:`BackendFactory.close` exits it. ``handles`` is the seed
    list of matched-arm functions in ``dataset.matched_func_names``
    order -- unmatched-arm functions are not seeded into the tree
    (plan decision D3). The matched-func-names list is dataset-level
    invariant (length ``== dataset.matched_count``), so the handle
    index is read unconditionally.
    """
    vocab = load_and_validate_unified_vocab(memmap_dir / "unified_vocab.csv")
    dataset = BinaryDataset(memmap_dir, binary_name, vocab_manager=vocab)
    session = dataset.open_session()
    session.__enter__()
    func_names = dataset.matched_func_names
    handles: List[FunctionHandle] = [
        FunctionHandle(arm=SectionKind.MATCHED, idx=idx, name=func_names[idx])
        for idx in range(dataset.matched_count)
    ]
    return _BatchDecodeBackendFactory(
        dataset=dataset,
        session=session,
        handles=handles,
    )
