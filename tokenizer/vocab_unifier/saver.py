import csv

import numpy as np

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType


def save_vocabulary(vocab_manager: VocabularyManager, csv_writer: csv.writer) -> None:
    token_count = len(vocab_manager.id_to_token)
    row = [
        "vocabulary",
        ",".join(vocab_manager.id_to_token),
        f"_id_to_token_type norm:{0 + TokenType.UNRESOLVED}",
        ndarray_to_base64(vocab_manager._id_to_token_type[:token_count] - TokenType.UNRESOLVED),
        f"_platform_instruction_type_cache norm:{0 + PlatformInstructionTypes.UNRESOLVED}",
        ndarray_to_base64(
            vocab_manager._platform_instruction_type_cache[:token_count] - PlatformInstructionTypes.UNRESOLVED
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
            ndarray_to_base64(vocab_manager.token_to_platform[:token_count] - platform_norm),
        ]
        row += extra

    csv_writer.writerow(row)
