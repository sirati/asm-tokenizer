"""Typed surface for the collection layer: member + missing-index policy.

Single concern: the pure typed pieces (no I/O, no sampling) the
collection layer hands across its boundary -- :class:`CollectionMember`
(the structured ``(qualified_name, memmap_dir, binary_name)`` triple)
and :class:`MissingIndexPolicy` (the discriminator for how a binary
lacking its requested ``.idx`` is handled).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


__all__ = ["CollectionMember", "MissingIndexPolicy"]


@dataclass(frozen=True)
class CollectionMember:
    """One discovered binary in the collection.

    ``qualified_name`` is the sampler/session key (bare ``binary_name``
    when unique across the whole corpus, else ``<dir.name>/<binary>``);
    ``memmap_dir`` is the directory holding the sidecars; ``binary_name``
    is the on-disk ``<binary>`` prefix a :class:`BinaryDataset` needs.
    """

    qualified_name: str
    memmap_dir: Path
    binary_name: str


class MissingIndexPolicy(Enum):
    """Policy for a binary that exists but lacks its requested ``.idx``.

    A missing index means that binary cannot contribute to the sample,
    which is a sampling bias -- so the default refuses it loudly.

    * :attr:`RAISE` -- abort discovery with a :class:`ValueError`
      listing every ``(dir, binary)`` missing its index file.
    * :attr:`SKIP_WITH_ERROR_LOG` -- emit an ERROR-level log record per
      excluded binary (the exclusion is a bias and must be LOUD), then
      exclude it from the collection.
    """

    RAISE = auto()
    SKIP_WITH_ERROR_LOG = auto()
