"""Dynrunner entry point for vocabulary unification.

This is a single-process aggregation (`unify_vocab` merges all CSVs into
one VocabularyManager sequentially); we skip the runner subprocess
machinery entirely and delegate to the standalone CLI.
"""
import logging

from tokenizer.vocab_unifier.__main__ import main as _standalone_main


def main() -> None:
    logging.getLogger().info(
        "vocab unification runs in single-process mode (no runner parallelism)"
    )
    _standalone_main()


if __name__ == "__main__":
    main()
