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

Provider-subdir policy: the typical corpus layout publishes
per-provider artefacts into a same-named subdirectory of the corpus
root (memmap bins land under ``<corpus>/memmap/`` per
:mod:`dynrunner.full_pipeline.phase_routing`; stage-1 CSVs sit at the
corpus root in the same pipeline but a future producer may put them in
``<corpus>/csv/``). :func:`resolve_provider_dirs` formalises the
"either in-place or in the provider's subdir" search so every
consumer (dialog scan, CLI resolver, opener) uses the same policy
keyed off the typed :class:`LoaderProvider` value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


__all__ = [
    "LoaderProvider",
    "SwitchTarget",
    "resolve_provider_dirs",
]


class LoaderProvider(Enum):
    """Discriminator for the two backend openers.

    * :attr:`MEMMAP` — per-binary memmap directory (``*_sections.bin``,
      ``*_function_names.txt``, ``unified_vocab.csv``); opened via
      :func:`tokenizer.inspector._render._backend_factory.make_batch_decode_factory`.
    * :attr:`CSV` — per-variant ``<base>_output.csv`` files; opened via
      :func:`tokenizer.inspector._render._backend_factory.make_ftl_factory`.

    The enum ``value`` doubles as the provider's canonical subdir name
    under the corpus root (see :func:`resolve_provider_dirs`); a typed
    member ``subdir_name`` property surfaces the contract without
    forcing callers to read the raw enum value.
    """

    MEMMAP = "memmap"
    CSV = "csv"

    @property
    def subdir_name(self) -> str:
        """Canonical subdirectory name for this provider's artefacts.

        Convention: the corpus pipeline writes provider X's files into
        ``<corpus_root>/<X>/``. ``MEMMAP`` -> ``memmap`` per
        :func:`dynrunner.full_pipeline.phase_routing._route_build_memmap`;
        ``CSV`` -> ``csv`` reserved for stage-1 outputs in a future
        producer that segregates them. Equal to the enum's ``value``
        so the mapping is self-consistent.
        """
        return self.value


def resolve_provider_dirs(path: Path, provider: LoaderProvider) -> List[Path]:
    """Ordered candidate locations for ``provider``'s artefacts.

    Search policy: the in-place location (``path`` itself) takes
    precedence over the provider's subdir (``path/<provider.subdir_name>/``)
    so an explicit per-dir copy wins over an inherited corpus-root one,
    mirroring :func:`tokenizer.aligned_data.loader.unified_vocab_gate.resolve_unified_vocab_path`.
    Both candidates are returned (without existence filtering) so the
    caller decides what counts as "data present" — discovery callers
    union over the list, resolvers stop at the first hit.

    Returning a list (not a single ``Optional[Path]``) preserves the
    "binaries may live in BOTH locations simultaneously" case: corpus
    builders can publish in one or the other and the dialog should
    surface either. Order matters: in-place first so the resolver's
    first-hit semantics mirror the unified-vocab gate.
    """
    return [path, path / provider.subdir_name]


@dataclass(frozen=True)
class SwitchTarget:
    """One user-selected switch target.

    ``provider`` discriminates between the two openers. ``path`` is the
    EFFECTIVE directory the opener consumes (memmap dir or csv dir) —
    when the user anchored at a corpus root whose bins live in a
    ``memmap/`` subdir, ``path`` is the subdir, not the root.
    ``anchor_path`` is the user's anchor (the dir they were browsing in
    the picker); the App stores it as ``_current_path`` so the next
    dialog opens at the same anchor regardless of where the loader
    actually reads from. ``binary`` is the binary name to focus on;
    ``None`` means "no specific binary picked — let the opener
    auto-detect if it can, otherwise fail loud". The binary-first
    dialog refactor closed the UI path that used to surface
    ``binary=None`` (the old ``[open this folder]`` row), so end-user
    dismissals always carry a concrete name; ``None`` is only seen by
    programmatic / test callers.

    ``anchor_path`` defaults to ``path`` so legacy callers that don't
    know about the in-place/subdir split (e.g. headless tests
    stubbing the opener with a flat directory) get the prior behaviour
    by construction.
    """

    provider: LoaderProvider
    path: Path
    binary: Optional[str] = None
    anchor_path: Optional[Path] = field(default=None)

    def __post_init__(self) -> None:
        # Default anchor_path to path so the legacy two-argument
        # ``SwitchTarget(provider, path, binary)`` construction in
        # existing tests keeps working without a second positional.
        if self.anchor_path is None:
            object.__setattr__(self, "anchor_path", self.path)
