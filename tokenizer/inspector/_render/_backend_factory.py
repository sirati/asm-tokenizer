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
from typing import List, Sequence

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

    Wraps a :class:`BinaryDataset` + an already-opened
    :class:`BinarySession`; :meth:`make` constructs one
    :class:`BatchDecodeBackend` per :meth:`FunctionNode.expand` call.

    Session lifetime is owned by the caller (``__main__`` enters
    + exits the ``with session:`` block). :meth:`close` is therefore
    a no-op on the session -- it only flips the closed flag so
    subsequent :meth:`make` calls raise.
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
        )

    def close(self) -> None:
        # Session is caller-owned (see class docstring); no-op beyond
        # the closed-flag flip.
        self._closed = True


def make_batch_decode_factory(
    memmap_dir: Path, binary_name: str
) -> tuple[BackendFactory, BinaryDataset, BinarySession]:
    """Build the BatchDecode factory for one binary in ``memmap_dir``.

    Returns the triple ``(factory, dataset, session)``; the caller
    drives ``with session:`` so handles release on exit. ``handles`` is
    the seed list of matched-arm functions in
    ``dataset.matched_func_names`` order -- unmatched-arm functions are
    not seeded into the tree (plan decision D3).
    """
    vocab = load_and_validate_unified_vocab(memmap_dir / "unified_vocab.csv")
    dataset = BinaryDataset(memmap_dir, binary_name, vocab_manager=vocab)
    session = dataset.open_session()
    func_names = dataset.matched_func_names
    handles: List[FunctionHandle] = [
        FunctionHandle(
            arm=SectionKind.MATCHED,
            idx=idx,
            name=func_names[idx] if idx < len(func_names) else "?",
        )
        for idx in range(dataset.matched_count)
    ]
    factory = _BatchDecodeBackendFactory(
        dataset=dataset,
        session=session,
        handles=handles,
    )
    return factory, dataset, session
