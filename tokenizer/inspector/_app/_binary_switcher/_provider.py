"""Loader-provider discriminator + switch-target dataclass.

Single concern: typed pure-data records the binary-switcher dialog
returns. The :class:`InspectorApp` side reads ``SwitchTarget.provider``
to pick the right opener (memmap → :func:`make_batch_decode_factory`,
csv → :func:`make_ftl_factory`), and ``SwitchTarget.binary`` to pick
the binary name (``None`` means "auto-detect / multi-binary load all
the directory's binaries" — represented as the ``[open this folder]``
entry).

No string-typed discriminators cross the boundary: :class:`LoaderProvider`
is an enum, and the App's switch dispatch table maps each enum value to
its opener function.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


__all__ = [
    "LoaderProvider",
    "SwitchTarget",
]


class LoaderProvider(Enum):
    """Discriminator for the two backend openers.

    * :attr:`MEMMAP` — per-binary memmap directory (``*_sections.bin``,
      ``*_function_names.txt``, ``unified_vocab.csv``); opened via
      :func:`tokenizer.inspector._render._backend_factory.make_batch_decode_factory`.
    * :attr:`CSV` — per-variant ``<base>_output.csv`` files; opened via
      :func:`tokenizer.inspector._render._backend_factory.make_ftl_factory`.
    """

    MEMMAP = "memmap"
    CSV = "csv"


@dataclass(frozen=True)
class SwitchTarget:
    """One user-selected switch target.

    ``provider`` discriminates between the two openers. ``path`` is the
    directory the opener consumes (memmap dir or csv dir). ``binary``
    is the binary name to focus on; ``None`` means "no specific binary
    picked — let the opener auto-detect if it can, otherwise fail
    loud". The dialog produces ``binary=None`` for the
    ``[open this folder]`` entry of a single-binary directory (auto-
    detect succeeds) and refuses to surface ``binary=None`` for multi-
    binary directories.
    """

    provider: LoaderProvider
    path: Path
    binary: Optional[str] = None
