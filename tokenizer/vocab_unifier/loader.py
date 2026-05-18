import csv
import io
import logging
from pathlib import Path

import numpy as np

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.compact_base64_utils import base64_to_ndarray
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

from .types import Platform


def assert_valid_vocab_def(row: list[str], platform: Platform) -> None:
    # v1 layout: 10 cells (per-platform) or 13 cells (unified).
    # v2 / v3 layout: appends `("format_version", "<int>")` at the tail —
    # 12 cells (per-platform) or 15 cells (unified). Wire-format is
    # otherwise byte-identical to v1; the trailing pair is the *only*
    # delta on the vocab tail row (the per-binary entries for IDs 0..255
    # are stripped by the saver because they are protocol-reserved
    # digit slots — see plan vivid-tinkering-wilkes.md). v3 is the
    # unified-vocab-only additive superset that registers variant-axis
    # tokens at IDs >= 256 (plan memoized-booping-wren.md) — same wire
    # column count as v2, only the trailer integer differs.
    base_cols = 13 if platform == "unified" else 10
    assert len(row) == base_cols or len(row) == base_cols + 2, (
        f"Expected {base_cols} (v1) or {base_cols + 2} (v2/v3) columns, got {len(row)}"
    )
    assert row[0] == "vocabulary"
    assert row[2].startswith("_id_to_token_type")
    assert row[4].startswith("_platform_instruction_type_cache")
    assert row[6] == "_lit_start_cache"
    assert row[8] == "_lit_end_cache"
    # Real unified-row layout puts the "platforms norm:..." header cell at
    # position 10 (row[9] is the lit_end base64 payload). The pre-existing
    # assertion checked row[9] which never matched any real layout;
    # `is_vocab_def` swallowed the AssertionError via its bare-except so
    # the typo was silently dead code (the only caller of this validator
    # via the read-back path, `load_unified_vocab_manager`, was already
    # returning None for an unrelated readline bug). Fixed here because
    # the v2 path now exercises this validation more rigorously.
    if platform == "unified":
        assert row[10].startswith("platforms"), (
            f"Expected 'platforms ...' header cell at position 10, got {row[10]!r}"
        )
    if len(row) == base_cols + 2:
        assert row[base_cols] == "format_version", (
            f"Expected v2 trailer cell 'format_version' at position {base_cols}, got {row[base_cols]!r}"
        )


def is_vocab_def(csv_row: bytes, platform: Platform) -> tuple[bool, list[str]]:
    try:
        csv_data = io.BytesIO(csv_row)
        reader = csv.reader(io.TextIOWrapper(csv_data, encoding="ascii"), quotechar='"')
        row = next(reader)
        assert_valid_vocab_def(row, platform)
        return True, row
    except Exception:
        return False, None


def read_last_line_of_file(csv_path: Path) -> bytes:
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

    assert last_line_chunk is not None, f"Could not find last line in {csv_path}"
    return last_line_chunk.tobytes()


def load_vocab_manager_csv_row_bytes(csv_row: bytes, platform: Platform) -> VocabularyManager | None:
    valid, row = is_vocab_def(csv_row, platform)
    if not valid:
        return None

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

    # v2 / v3 trailer: ("format_version", "<int>") appended after the v1
    # tail. Saver strips the protocol-reserved digit slots (IDs 0..255)
    # from the serialized vocab; the loader reconstitutes them here so
    # downstream absolute-ID lookups (lit caches,
    # register_on_vocab_manager, etc.) all stay valid. v3 is the additive
    # unified-vocab superset (variant-axis tokens at IDs >= 256; wire
    # layout identical to v2) per plan memoized-booping-wren.md, so both
    # versions take this branch.
    base_cols = 13 if platform == "unified" else 10
    format_version = 1
    if len(row) == base_cols + 2:
        format_version = int(row[base_cols + 1])

    if format_version in (2, 3):
        reserved = VocabularyManager._V2_RESERVED_DIGIT_COUNT
        digit_names = [f"digit_{i:02X}" for i in range(reserved)]
        vocabulary = digit_names + vocabulary
        id_to_token_type = np.concatenate(
            [
                np.full(reserved, TokenType.UNRESOLVED, dtype=id_to_token_type.dtype),
                id_to_token_type,
            ]
        )
        platform_instruction_type_cache = np.concatenate(
            [
                np.full(
                    reserved,
                    PlatformInstructionTypes.AGNOSTIC,
                    dtype=platform_instruction_type_cache.dtype,
                ),
                platform_instruction_type_cache,
            ]
        )
        if token_to_platform is not None:
            token_to_platform = np.concatenate(
                [
                    np.full(reserved, -1, dtype=token_to_platform.dtype),
                    token_to_platform,
                ]
            )
        # Lit caches reference absolute IDs >= 256 (legacy stubs registered
        # for class-stability under a v2 VM, if any); the saver writes them
        # unshifted, so no adjustment is needed here.

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
        format_version=format_version,
    )


def load_vocab_manager(csv_path: Path, platform: Platform | None = None) -> VocabularyManager | None:
    if platform is None:
        # Auto-detect the canonical Platform from the filename prefix.
        # Filename prefix may be either a canonical Platform literal
        # (legacy: `arm32-...`, `x64-...`) or a sidecar arch alias
        # (`armv7l-hf-...`, `x86_64-...`). The arch_translation table
        # accepts both shapes and collapses them onto the canonical
        # Platform — single source of truth shared with the tokenize
        # worker.
        #
        # Sort prefix candidates by length DESC so longer prefixes
        # match first (e.g. `mips64` wins over `mips`; `armv7l-hf`
        # wins over `arm`); without this `mips64-...`.startswith("mips")
        # would silently mis-detect.
        from tokenizer.arch_translation import (
            all_known_arch_strings,
            arch_to_platform,
        )
        platform_options = sorted(all_known_arch_strings(), key=len, reverse=True)
        file_name = csv_path.name
        for option in platform_options:
            if file_name.startswith(option + "-"):
                platform = arch_to_platform(option)
                break

    assert platform is not None, f"Could not determine platform from file name: {csv_path.name}"

    try:
        last_line = read_last_line_of_file(csv_path)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error reading last line of file: {csv_path}")
        logger.error(f"Error message: {e}")
        return None

    return load_vocab_manager_csv_row_bytes(last_line, platform)


def load_unified_vocab_manager(csv_path: Path) -> VocabularyManager | None:
    """Load vocabulary manager from unified_vocab.csv file.

    The unified-vocab file is exactly one CSV row (the vocab definition
    line, written by ``unifier.unify_vocab`` via a single
    ``csv.writer.writerow`` call). Earlier revisions of this loader
    assumed a leading header line and used ``readline`` to skip it,
    which always returned ``None`` against the real writer output
    (called out in ``assert_valid_vocab_def``'s comment as the
    "unrelated readline bug"). Reading the whole file as the row
    avoids that mismatch and naturally tolerates an optional trailing
    newline — ``is_vocab_def`` already pipes the bytes through
    ``csv.reader`` which strips it.

    Args:
        csv_path: Path to unified_vocab.csv file

    Returns:
        VocabularyManager instance or None if loading fails
    """
    try:
        with open(csv_path, "rb") as f:
            vocab_line = f.read()
        return load_vocab_manager_csv_row_bytes(vocab_line, "unified")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error reading vocab from file: {csv_path}")
        logger.error(f"Error message: {e}")
        return None
