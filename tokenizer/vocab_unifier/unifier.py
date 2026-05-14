import csv
import logging
from pathlib import Path

import numpy as np

from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager

from .loader import load_vocab_manager
from .saver import save_vocabulary

logger = logging.getLogger(__name__)


def unify_vocab(
    csv_files: list[Path],
    unified_vocab_file: Path,
    mapping_output_dir: Path | None = None,
    mapping_source_root: Path | None = None,
) -> None:
    """Build a single unified vocabulary across `csv_files` and emit
    one `.mapping.b64c` file per input CSV (the per-binary local-ID →
    unified-ID translation table).

    `mapping_output_dir` controls where mapping files are written:
      - None (default): next to each CSV, via `csv_file.with_suffix`.
        This matches the standalone CLI's historic behaviour and is
        what the local-mode dynrunner dispatch expects.
      - Set: mapping files land under `<mapping_output_dir>` mirroring
        the relative layout of the CSVs under `mapping_source_root`.
        The SLURM-dispatch worker uses this so mapping files land on
        the writable bind-mount (`/app/out-network`) instead of the
        read-only source bind-mount (`/app/src-network`), while still
        preserving subdirectory structure so build_memmap's identifier-
        equality match between `<rel>/<binary>_output.csv` and
        `<rel>/<binary>_output.mapping.b64c` succeeds.

    `mapping_source_root`: required when `mapping_output_dir` is set.
    Each `csv_file` must be reachable from this root; the relative
    subdir is preserved when computing the mapping path.
    """
    if mapping_output_dir is not None and mapping_source_root is None:
        raise ValueError(
            "unify_vocab: mapping_source_root is required when "
            "mapping_output_dir is set, so per-CSV subdir layout can be "
            "preserved on the output side."
        )

    # `unified_vm` is constructed lazily once we know the format_version of
    # the inputs. Mixed v1/v2 inputs are rejected (a unified vocab would
    # have inconsistent reserved-id semantics — IDs 0..255 are real entries
    # under v1 but protocol-reserved digits under v2). Either-version is
    # acceptable in isolation; the all-same check happens on first load.
    unified_vm: VocabularyManager | None = None
    unified_format_version: int | None = None

    if mapping_output_dir is not None:
        mapping_output_dir.mkdir(parents=True, exist_ok=True)

    loaded_count = 0
    for csv_file in csv_files:
        print(f"Loading vocabulary from {csv_file}")
        current_vocab_manager = load_vocab_manager(csv_file)
        if current_vocab_manager is None:
            logger.error(f"Failed to load vocabulary from {csv_file}. Missing or incomplete (no vocab def in last line).")
            continue
        loaded_count += 1

        current_format_version = current_vocab_manager.format_version
        if unified_format_version is None:
            unified_format_version = current_format_version
            unified_vm = VocabularyManager(
                platform=None, format_version=unified_format_version
            )
        elif current_format_version != unified_format_version:
            raise ValueError(
                f"unify_vocab: cannot mix vocab format versions in one run. "
                f"Earlier inputs were format_version={unified_format_version}; "
                f"{csv_file} reports format_version={current_format_version}. "
                f"Re-tokenize the older corpus to v2, or run the unifier on "
                f"each version separately."
            )

        mappings = np.full_like(current_vocab_manager.id_to_token_type, -1, dtype=np.int32)

        # Under format_version=2, IDs 0..255 are protocol-reserved digit
        # slots. The plan requires explicit identity remap for that range —
        # both per-binary VM and unified VM agree on those positions by
        # construction, so the inline-digit stream survives `mapping[tokens]`
        # unchanged. Filling here before the representative loop is
        # idempotent (the loop only touches IDs >= 256 under v2 since
        # `iter_representative_tokens` skips reserved digits).
        if current_format_version == 2:
            reserved = VocabularyManager._V2_RESERVED_DIGIT_COUNT
            mappings[:reserved] = np.arange(reserved, dtype=mappings.dtype)

        for tokens in current_vocab_manager.iter_representative_tokens():
            original = tokens.get_token_ids()
            mapped = tokens.register_on_vocab_manager(unified_vm).get_token_ids()
            assert len(original) == len(mapped)
            for original_id, mapped_id in zip(original, mapped):
                mappings[original_id] = mapped_id

        if not np.all(mappings >= 0):
            logger.error(f"Invalid mappings found in {csv_file}, skipping file")
            continue

        if mapping_output_dir is None:
            mapping_file_path = csv_file.with_suffix(".mapping.b64c")
        else:
            # Mirror the CSV's subdir under mapping_output_dir so
            # build_memmap's identifier-equality match between
            # `<rel>/<binary>_output.csv` and
            # `<rel>/<binary>_output.mapping.b64c` lines up.
            rel = csv_file.relative_to(mapping_source_root)
            mapping_file_path = (
                mapping_output_dir / rel.parent / (csv_file.stem + ".mapping.b64c")
            )
            mapping_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_file_path, "w", newline="", encoding="ascii") as mapping_file:
            mapping_file.write(ndarray_to_base64(mappings))

    if loaded_count == 0:
        # Every CSV failed to load — there's no vocab to unify. Emitting
        # an empty unified_vocab.csv would silently propagate garbage to
        # phase 3. Surface this as a hard failure so the framework
        # marks unify-vocab unrecoverable instead of advancing.
        raise FileNotFoundError(
            f"unify_vocab: 0 of {len(csv_files)} input CSVs were loadable. "
            f"Phase 1 (tokenize) likely produced no usable output."
        )

    print(f"Saving unified vocabulary to {unified_vocab_file} "
          f"({loaded_count}/{len(csv_files)} CSVs loaded)")
    with open(unified_vocab_file, "w", newline="", encoding="ascii") as csvfile:
        writer = csv.writer(csvfile)
        save_vocabulary(unified_vm, writer)
