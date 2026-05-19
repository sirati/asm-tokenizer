import csv

from tokenizer.architecture import PlatformInstructionTypes
from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenType

# Unified-vocab artifacts always carry the single in-tree memmap-chain
# version (memmap_format.MEMMAP_FORMAT_VERSION). The per-binary CSV
# (out-of-scope tokenize-output, see plan memoized-booping-wren.md
# §"Out of scope") keeps its own version=2 trailer; this writer stamps
# whichever value the manager declares. The constant is duplicated here
# rather than imported from tokenizer.aligned_data.memmap_format because
# that package eagerly imports vocab_unifier.loader (via the unified-vocab
# gate), forming an import cycle. Bumps stay coupled because the gate
# also raises on any version other than its imported constant.
_UNIFIED_FORMAT_VERSION = 1
_PER_BINARY_FORMAT_VERSION = 2

_SUPPORTED_FORMAT_VERSIONS = (_UNIFIED_FORMAT_VERSION, _PER_BINARY_FORMAT_VERSION)


def save_vocabulary(vocab_manager: VocabularyManager, csv_writer: csv.writer) -> None:
    # Two callers exist: the vocab_unifier produces unified-vocab
    # format_version=1 (see plan memoized-booping-wren.md §"Format-version
    # coupling"), and the per-binary tokenize worker produces per-binary
    # CSV format_version=2. Both share the wire layout: the first
    # `_V2_RESERVED_DIGIT_COUNT` IDs are protocol-reserved digit slots,
    # not written on the wire (the loader reconstitutes them from the
    # protocol convention based on the trailer integer). Any other value
    # is a programmer error — the legacy v1-no-trailer and v3 paths were
    # removed in the memoized-booping-wren.md cleanup.
    if vocab_manager.format_version not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"save_vocabulary supports unified vocab format_version="
            f"{_UNIFIED_FORMAT_VERSION} or per-binary CSV format_version="
            f"{_PER_BINARY_FORMAT_VERSION}; got "
            f"{vocab_manager.format_version}"
        )

    token_count = len(vocab_manager.id_to_token)
    # Strip the protocol-reserved digit slots; no entries are written for
    # these positions. Both supported versions share this encoding.
    start = VocabularyManager._V2_RESERVED_DIGIT_COUNT

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
