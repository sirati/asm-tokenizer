import csv
import io
from pathlib import Path

import numpy as np

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.compact_base64_utils import base64_to_ndarray
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

from .types import Platform


def load_vocab_manager_csv_row_bytes(csv_row: bytes, platform: Platform) -> VocabularyManager:
    csv_data = io.BytesIO(csv_row)
    reader = csv.reader(io.TextIOWrapper(csv_data, encoding="ascii"), quotechar='"')
    row = next(reader)
    assert len(row) == 10 or (platform == "unified" and len(row) == 13), f"Expected 10 or 13 columns, got {len(row)}"
    assert row[0] == "vocabulary"
    assert row[2].startswith("_id_to_token_type")
    assert row[4].startswith("_platform_instruction_type_cache")
    assert row[6] == "_lit_start_cache"
    assert row[8] == "_lit_end_cache"
    if platform == "unified":
        assert row[9].startswith("platforms")

    vocabulary = row[1].strip('"').split(",")
    id_to_token_type_offset = int(row[2].partition("norm:")[2])
    platform_instruction_type_cache_offset = int(row[4].partition("norm:")[2])
    id_to_token_type = base64_to_ndarray(row[3]).astype(np.int8) + id_to_token_type_offset
    platform_instruction_type_cache = base64_to_ndarray(row[5]).astype(np.int8) + platform_instruction_type_cache_offset
    lit_start_cache = base64_to_ndarray(row[7]).astype(np.int_)
    lit_end_cache = base64_to_ndarray(row[9]).astype(np.int_)
    platform_offset = int(row[10].partition("norm:")[2]) if platform == "unified" else None
    platform_list = row[11].strip('"').split(",") if platform == "unified" else None
    token_to_platform = base64_to_ndarray(row[12]).astype(np.int8) + platform_offset if platform == "unified" else None

    platform = platform if platform != "unified" else None

    return VocabularyManager.from_vocab(
        platform=platform,
        vocab_list=vocabulary,
        id_to_token_type=id_to_token_type,
        platform_instruction_type_cache=platform_instruction_type_cache,
        lit_start_cache=lit_start_cache,
        lit_end_cache=lit_end_cache,
        platform_list=platform_list,
        token_to_platform=token_to_platform,
    )


def load_vocab_manager(csv_path: Path, platform: Platform | None = None) -> VocabularyManager:
    if platform is None:
        platform_options = Platform.__args__
        file_name = csv_path.name
        for option in platform_options:
            if file_name.startswith(option):
                platform = option
                break

    assert platform is not None, f"Could not determine platform from file name: {csv_path.name}"

    data = np.memmap(csv_path, dtype=np.uint8, mode="r")
    search_area = data[:-64]
    chunk_size = 1 << 14

    num_chunks = (np.size(search_area) + chunk_size - 1) // chunk_size

    last_line_chunk = None
    for i in range(num_chunks):
        start = max(-(i + 1) << 14, -np.size(search_area))
        end = -(i << 14) if (i << 14) != 0 else None
        chunk = search_area[start:end]

        mask = (chunk == 10) | (chunk == 13)

        if np.any(mask):
            last_local_index = np.where(mask)[0][-1]
            last_global_index = (np.size(search_area) + start) + last_local_index + 1
            last_line_chunk = data[last_global_index:]
            break

    return load_vocab_manager_csv_row_bytes(last_line_chunk.tobytes(), platform)
