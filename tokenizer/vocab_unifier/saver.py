import csv

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

# Unified-vocab artifacts always carry the single in-tree memmap-chain
# version (``MEMMAP_FORMAT_VERSION``). The per-binary CSV (out-of-scope
# tokenize-output, see plan memoized-booping-wren.md §"Out of scope")
# keeps its own version=2 trailer; this writer stamps whichever value
# the manager declares.
_PER_BINARY_FORMAT_VERSION = 2

_SUPPORTED_FORMAT_VERSIONS = (MEMMAP_FORMAT_VERSION, _PER_BINARY_FORMAT_VERSION)


def save_vocabulary(vocab_manager: VocabularyManager, csv_writer: csv.writer) -> None:
    # Two callers exist: the vocab_unifier produces unified-vocab
    # format_version=1 (see plan memoized-booping-wren.md §"Format-version
    # coupling"), and the per-binary tokenize worker produces per-binary
    # CSV format_version=2. Both share the wire layout: the first
    # `_V2_RESERVED_TOKEN_COUNT` (= 257) IDs are protocol-reserved and
    # never written on the wire — that span covers the 256 inline-digit
    # slots PLUS the `value_negative` postfix marker pinned at slot 256.
    # The loader reconstitutes the full reserved prefix from the protocol
    # convention based on the trailer integer. Any other format_version
    # is a programmer error — the legacy v1-no-trailer and v3 paths were
    # removed in the memoized-booping-wren.md cleanup.
    if vocab_manager.format_version not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"save_vocabulary supports unified vocab format_version="
            f"{MEMMAP_FORMAT_VERSION} or per-binary CSV format_version="
            f"{_PER_BINARY_FORMAT_VERSION}; got "
            f"{vocab_manager.format_version}"
        )

    token_count = len(vocab_manager.id_to_token)
    # Strip the protocol-reserved prefix (`_V2_RESERVED_TOKEN_COUNT` = 257
    # slots: digits 0..255 PLUS `value_negative` at slot 256). Neither the
    # digit names nor the value_negative marker is written on the wire —
    # the loader reconstructs both from the protocol convention based on
    # the trailer integer. Both supported versions (v1 unified, v2 per-
    # binary) share this encoding and converge on the same strip boundary.
    start = VocabularyManager._V2_RESERVED_TOKEN_COUNT

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

    # Trailer pair always written. The loader keys decode behavior on the
    # integer — it does not branch on per-version layouts because both
    # supported versions share the wire format byte-for-byte.
    row += ["format_version", str(vocab_manager.format_version)]

    csv_writer.writerow(row)
