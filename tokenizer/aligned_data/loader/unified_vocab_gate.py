"""Single-concern helper: load + validate the corpus-wide unified vocab.

Hard cutover: the variant-aware dataloader REQUIRES
``unified_vocab.format_version == MEMMAP_FORMAT_VERSION`` (currently v1).
Any other on-disk vocab is rejected up-front, before any per-binary state
materialises, so the caller sees a single clean ``ValueError`` rather than
a downstream decode mismatch. The reader has NO knowledge of specific
legacy version numbers — it simply requires v1 and treats every other
value identically as "not v1, regenerate".

Path-resolution policy lives next to the gate so every consumer that
defaults a vocab path from a memmap directory shares one search order:
the vocab can sit alongside the memmap bins OR one level up (corpus
root with bins in a ``memmap/`` subdir). See
``resolve_unified_vocab_path``.

Kept out of ``aligned_data_loader.py`` so the loader file stays under the
300 LOC project cap and so the gate is independently testable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tokenizer.aligned_data.memmap_format import (
    MEMMAP_FORMAT_VERSION,
    _PRELUDE_RESERVED_SIZE,
)
from tokenizer.token_manager import VocabularyManager


def compute_vocab_fingerprint(vocab_path: Path) -> bytes:
    """Return the 8-byte identity fingerprint of a ``unified_vocab.csv``.

    The first 8 bytes of ``sha256(file bytes)`` — enough to distinguish any
    two distinct corpus vocabs (e.g. an arm64 vs a mips64 unify) with
    negligible collision risk. The memmap builder stamps this into each
    ``_data.bin`` prelude; the loader recomputes it from the vocab it
    actually loaded and HARD-FAILS on mismatch (catalog built against a
    different vocab than the one being used to decode it). Computed from the
    raw CSV bytes so the BUILDER (which has the co-located file) and the
    LOADER (which reads the same file to parse it) agree by construction.
    """
    digest = hashlib.sha256(Path(vocab_path).read_bytes()).digest()
    return digest[:_PRELUDE_RESERVED_SIZE]

# ``load_unified_vocab_manager`` is imported lazily inside the function
# body to break the import cycle: ``tokenizer.vocab_unifier`` (its
# package __init__ pulls .loader, which imports
# ``tokenizer.aligned_data.memmap_format`` and so triggers the
# ``aligned_data/__init__`` cascade that lands back here). A top-level
# import here resolves into a partially-initialised vocab_unifier.loader
# and raises ``ImportError`` on every ``python -m tokenizer.vocab_unifier``
# entry point.


# Required on-disk layout version for ``unified_vocab.csv``. Sourced from
# the single ``MEMMAP_FORMAT_VERSION`` constant so future bumps cascade
# through every consumer without a touch here.
REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION = MEMMAP_FORMAT_VERSION


# On-disk basename for the corpus-wide unified vocab. Centralised so the
# search policy below and every caller agree on one spelling.
UNIFIED_VOCAB_BASENAME = "unified_vocab.csv"


def _unified_vocab_candidates(memmap_dir: Path) -> list[Path]:
    """Ordered candidate locations for ``unified_vocab.csv``.

    The search policy is symmetric: the vocab can live alongside the
    memmap bins (``<memmap_dir>/unified_vocab.csv``) or one level up at
    the corpus root (``<memmap_dir>/../unified_vocab.csv``) when the bins
    sit in a subdirectory of the corpus root. Both layouts are produced
    by valid build pipelines; pick the first that exists.

    Order matters: the in-directory location takes precedence so an
    explicit per-memmap-dir copy wins over an inherited corpus-root one.
    """
    return [
        memmap_dir / UNIFIED_VOCAB_BASENAME,
        memmap_dir.parent / UNIFIED_VOCAB_BASENAME,
    ]


def resolve_unified_vocab_path(memmap_dir: Path) -> Path:
    """Search for ``unified_vocab.csv`` near ``memmap_dir`` and return it.

    Single source of truth for "where does the unified vocab live when
    the caller only knows the memmap directory?". Callers that have an
    explicit user-supplied path (e.g. via a CLI flag) bypass this and
    feed the path straight to :func:`load_and_validate_unified_vocab`.

    Args:
        memmap_dir: Directory containing the per-binary memmap bins.

    Returns:
        The first candidate location that exists on disk.

    Raises:
        ValueError: None of the candidate locations contain a file.
            The message enumerates every probed path so the operator can
            copy the vocab to whichever location they prefer.
    """
    candidates = _unified_vocab_candidates(Path(memmap_dir))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    listed = ", ".join(str(c) for c in candidates)
    raise ValueError(
        f"unified_vocab.csv not found near memmap dir {memmap_dir}; "
        f"searched: {listed}. The variant-aware dataloader requires the "
        "SAME corpus-wide unified vocab that was used to build this "
        "memmap (the vocab is part of the memmap's identity — building "
        "a new one would produce ids that disagree with the bin data). "
        "Copy the unified_vocab.csv from the build-pipeline output to "
        "one of the searched locations."
    )


def load_and_validate_unified_vocab(vocab_path: Path) -> VocabularyManager:
    """Load ``unified_vocab.csv`` and enforce the memmap-chain version.

    Args:
        vocab_path: Path to the corpus-wide unified vocab CSV.

    Returns:
        The validated ``VocabularyManager``.

    Raises:
        ValueError: missing file, unparseable contents, or
            ``format_version != REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION``.
            The exception message names the path and the version mismatch
            loudly so an operator can resolve it without grepping logs.
    """
    if not vocab_path.exists():
        raise ValueError(
            f"unified_vocab.csv not found at {vocab_path}; the "
            "variant-aware dataloader requires the SAME corpus-wide "
            "unified vocab that was used to build this memmap (the "
            "vocab is part of the memmap's identity — building a new "
            "one would produce ids that disagree with the bin data). "
            "Copy the unified_vocab.csv from the build-pipeline output "
            "alongside the rest of the memmap files."
        )

    from tokenizer.vocab_unifier.loader import load_unified_vocab_manager

    vocab_manager = load_unified_vocab_manager(vocab_path)
    if vocab_manager is None:
        raise ValueError(
            f"unified_vocab.csv at {vocab_path} failed to parse "
            "(see logs for the underlying error). The variant-aware "
            "dataloader cannot proceed without a usable unified vocab."
        )

    if vocab_manager.format_version != REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION:
        raise ValueError(
            f"unified_vocab.format_version={vocab_manager.format_version}; "
            f"v{REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION} required for the "
            "memmap-output chain. Re-run `python -m tokenizer.vocab_unifier` "
            "followed by `python -m tokenizer.memmap_builder` on the "
            "per-binary CSVs to regenerate."
        )

    # Stamp the vocab's identity fingerprint onto the VM so the per-binary
    # session can verify a catalog was built against THIS vocab before
    # decoding it -- catching a wrong-but-same-format-version vocab that
    # would silently mis-decode the variant-axis band (#27 safety net).
    vocab_manager._vocab_fingerprint = compute_vocab_fingerprint(vocab_path)
    return vocab_manager
