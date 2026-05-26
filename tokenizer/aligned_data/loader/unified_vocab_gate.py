"""Single-concern helper: load + validate the corpus-wide unified vocab.

Hard cutover: the variant-aware dataloader REQUIRES
``unified_vocab.format_version == MEMMAP_FORMAT_VERSION`` (currently v1).
Any other on-disk vocab is rejected up-front, before any per-binary state
materialises, so the caller sees a single clean ``ValueError`` rather than
a downstream decode mismatch. The reader has NO knowledge of specific
legacy version numbers — it simply requires v1 and treats every other
value identically as "not v1, regenerate".

Kept out of ``aligned_data_loader.py`` so the loader file stays under the
300 LOC project cap and so the gate is independently testable.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.token_manager import VocabularyManager

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

    return vocab_manager
