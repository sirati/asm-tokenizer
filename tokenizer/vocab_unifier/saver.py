import csv

import numpy as np

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType


def save_vocabulary(vocab_manager: VocabularyManager, csv_writer: csv.writer) -> None:
    token_count = len(vocab_manager.id_to_token)
    # Under format_version=2 the first `_V2_RESERVED_DIGIT_COUNT` IDs are
    # protocol-reserved digit slots — never emitted on the wire (see plan
    # vivid-tinkering-wilkes.md: "no entries are written for these
    # positions"). Slice them off the serialized vocab + accompanying
    # per-ID arrays; the loader reconstitutes them from the protocol
    # convention when `format_version=2` is detected in the trailer.
    if vocab_manager.format_version == 2:
        start = VocabularyManager._V2_RESERVED_DIGIT_COUNT
    else:
        start = 0

    row = [
        "vocabulary",
        ",".join(vocab_manager.id_to_token[start:]),
        f"_id_to_token_type norm:{0 + TokenType.UNRESOLVED}",
        ndarray_to_base64(vocab_manager._id_to_token_type[start:token_count] - TokenType.UNRESOLVED),
        f"_platform_instruction_type_cache norm:{0 + PlatformInstructionTypes.UNRESOLVED}",
        ndarray_to_base64(
            vocab_manager._platform_instruction_type_cache[start:token_count] - PlatformInstructionTypes.UNRESOLVED
        ),
        "_lit_start_cache",
        ndarray_to_base64(vocab_manager._lit_start_cache[: vocab_manager._lit_start_count]),
        "_lit_end_cache",
        ndarray_to_base64(vocab_manager._lit_end_cache[: vocab_manager._lit_end_count]),
    ]

    if vocab_manager.platform is None:
        platform_norm = -1
        extra = [
            f"platforms norm:{platform_norm}",
            ",".join(vocab_manager.platform_list),
            ndarray_to_base64(vocab_manager.token_to_platform[start:token_count] - platform_norm),
        ]
        row += extra

    if vocab_manager.format_version == 2:
        # Append the v2 trailer pair after the v1-shaped row so v1 readers
        # (which assert the legacy column count) reject the file cleanly
        # rather than silently mis-decoding.
        row += ["format_version", str(vocab_manager.format_version)]

    csv_writer.writerow(row)
