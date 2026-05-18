import csv
import logging
from pathlib import Path

import numpy as np

from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager

from .loader import load_vocab_manager
from .saver import save_vocabulary
from .variant_registration import discover_and_register_variants

logger = logging.getLogger(__name__)

# uint16 ceiling for the wire-format vocab id (matches the bin formats'
# `_data.bin` / `_variants.bin` layout — IDs > 0xFFFF would silently
# truncate). Surfaced as a hard post-condition on the unified vocab
# size; if a corpus actually hits this, the remediation is to split it
# by arch and unify per slice rather than silently corrupt the dataset.
_UINT16_CEILING = 0xFFFF


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

    if mapping_output_dir is not None:
        mapping_output_dir.mkdir(parents=True, exist_ok=True)

    # v3 unified vocab: variant-axis tokens occupy a low contiguous
    # block starting at id 256 (the v2 reserved-digit boundary);
    # instruction representatives from the per-binary v2 CSVs land
    # above that block via pass 2. The unified VM is v3 from the
    # start so `_private_add_token` recognises the reserved-digit
    # layout for both variant and instruction registrations.
    unified_vm = VocabularyManager(platform=None, format_version=3)

    # Pass 1 — sidecar-only walk to register every distinct variant-
    # axis token. No CSV body is read in this pass; the input CSVs'
    # filenames + optional ``_meta.json`` siblings carry all the
    # variant identity the unifier needs.
    n_variants = discover_and_register_variants(csv_files, unified_vm)

    # Pass 2 — instruction-representative walk against each per-binary
    # vocab CSV. The mapping array translates the per-binary id space
    # into the unified id space; v3 reuses v2's reserved-digit semantics
    # so the identity remap for IDs 0..255 stays valid.
    loaded_count = 0
    for csv_file in csv_files:
        print(f"Loading vocabulary from {csv_file}")
        current_vocab_manager = load_vocab_manager(csv_file)
        if current_vocab_manager is None:
            logger.error(f"Failed to load vocabulary from {csv_file}. Missing or incomplete (no vocab def in last line).")
            continue
        loaded_count += 1

        current_format_version = current_vocab_manager.format_version
        # Per-binary CSVs are always v2 under the plan (variant tokens
        # live ONLY in the unified vocab). A v1 or v3 input here would
        # mean someone fed the unifier a stale or mis-built CSV; refuse
        # explicitly rather than silently mis-decoding the reserved-id
        # band.
        if current_format_version != 2:
            raise ValueError(
                f"unify_vocab: per-binary CSV must be format_version=2; "
                f"{csv_file} reports format_version={current_format_version}. "
                f"Re-tokenize the corpus or check whether the input was "
                f"accidentally a unified vocab instead of a per-binary one."
            )

        mappings = np.full_like(current_vocab_manager.id_to_token_type, -1, dtype=np.int32)

        # Under format_version=2, IDs 0..255 are protocol-reserved digit
        # slots. The plan requires explicit identity remap for that range —
        # both per-binary VM and unified VM agree on those positions by
        # construction, so the inline-digit stream survives `mapping[tokens]`
        # unchanged. Filling here before the representative loop is
        # idempotent (the loop only touches IDs >= 256 under v2 since
        # `iter_representative_tokens` skips reserved digits).
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

    # Hard post-condition: the unified vocab must fit in uint16 because
    # `_data.bin` and `_variants.bin` both serialise IDs as uint16.
    # Silent overflow would corrupt the dataset; surface as ValueError
    # with a remediation hint so the operator can split the corpus by
    # arch and unify per slice instead.
    total_ids = len(unified_vm.id_to_token)
    if total_ids > _UINT16_CEILING:
        raise ValueError(
            f"unify_vocab: unified vocab has {total_ids} tokens "
            f"({n_variants} variant + "
            f"{total_ids - VocabularyManager._V2_RESERVED_DIGIT_COUNT - n_variants} "
            f"instruction), exceeds uint16 ceiling ({_UINT16_CEILING}). "
            f"Split the corpus by architecture and unify each slice "
            f"separately, or prune low-frequency tokens before unifying."
        )

    print(f"Saving unified vocabulary to {unified_vocab_file} "
          f"({loaded_count}/{len(csv_files)} CSVs loaded; "
          f"{n_variants} variant-axis tokens registered)")
    with open(unified_vocab_file, "w", newline="", encoding="ascii") as csvfile:
        writer = csv.writer(csvfile)
        save_vocabulary(unified_vm, writer)
