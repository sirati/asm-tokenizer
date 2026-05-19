"""Single-concern helper: load + validate the corpus-wide unified vocab.

Hard cutover: the variant-aware dataloader REQUIRES
``unified_vocab.format_version == 3``. v2 unified vocabs predate variant-axis
tokens, and a v2-shaped dataset's section CSVs cite ``variant_ref`` cells
that the v3-aware loader would otherwise misinterpret as row indices. The
gate refuses the dataset up-front, before any per-binary state materialises,
so the caller sees a single clean ``ValueError`` rather than a downstream
decode mismatch.

Kept out of ``aligned_data_loader.py`` so the loader file stays under the
300 LOC project cap and so the gate is independently testable.
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.token_manager import VocabularyManager
from tokenizer.vocab_unifier.loader import load_unified_vocab_manager


# Required on-disk layout version for ``unified_vocab.csv``. Bumped to 3 when
# the unifier began registering variant-axis tokens (see plan
# memoized-booping-wren.md §"Format-version policy").
REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION = 3


def load_and_validate_unified_vocab(vocab_path: Path) -> VocabularyManager:
    """Load ``unified_vocab.csv`` and enforce ``format_version == 3``.

    Args:
        vocab_path: Path to the corpus-wide unified vocab CSV.

    Returns:
        The validated ``VocabularyManager``.

    Raises:
        ValueError: missing file, unparseable contents, or
            ``format_version != 3``. The exception message names the path
            and the version mismatch loudly so an operator can resolve it
            without grepping logs.
    """
    if not vocab_path.exists():
        raise ValueError(
            f"unified_vocab.csv not found at {vocab_path}; the "
            "variant-aware dataloader requires a corpus-wide unified "
            "vocab. Run `python -m tokenizer.vocab_unifier --source "
            "<dir> --output <dir>` to produce one."
        )

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
            f"v{REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION} required for "
            "variant-aware dataloader. Re-run tokenizer.vocab_unifier to "
            f"produce a v{REQUIRED_UNIFIED_VOCAB_FORMAT_VERSION} vocab."
        )

    return vocab_manager
