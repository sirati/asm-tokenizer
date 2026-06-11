"""Corpus-level collection layer over many indexed memmap directories.

Re-exports the public surface: :class:`IndexedMemmapCollection` (the
unbiased length-bucketed batch source), :class:`CollectionMember` (the
typed discovered-binary triple), and :class:`MissingIndexPolicy` (the
missing-``.idx`` discriminator).
"""

from __future__ import annotations

from ._collection import IndexedMemmapCollection
from ._member import CollectionMember, MissingIndexPolicy


__all__ = [
    "CollectionMember",
    "IndexedMemmapCollection",
    "MissingIndexPolicy",
]
