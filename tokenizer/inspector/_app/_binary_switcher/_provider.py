"""Loader-provider discriminator + switch-target dataclass.

Single concern: typed pure-data records the binary-switcher dialog
returns. The :class:`InspectorApp` side reads ``SwitchTarget.provider``
to pick the right opener (memmap → :func:`make_batch_decode_factory`,
csv → :func:`make_ftl_factory`), and ``SwitchTarget.binary`` to pick
the binary name. After the binary-first dialog refactor the dialog
ALWAYS supplies a concrete binary name, so ``binary=None`` is reserved
for programmatic callers (e.g. headless tests stubbing the opener
table) and short-circuits to the resolver's auto-detect path — which
raises :class:`SystemExit` on multi-binary directories. Surfacing
``None`` from the UI is no longer possible.

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
    loud". The binary-first dialog refactor closed the UI path that
    used to surface ``binary=None`` (the old ``[open this folder]``
    row), so end-user dismissals always carry a concrete name; ``None``
    is only seen by programmatic / test callers.
    """

    provider: LoaderProvider
    path: Path
    binary: Optional[str] = None
