import csv
import logging
from pathlib import Path

import numpy as np

from tokenizer.aligned_data.memmap_format import MEMMAP_FORMAT_VERSION
from tokenizer.compact_base64_utils import ndarray_to_base64
from tokenizer.token_manager import VocabularyManager
from tokenizer.variant_tokens import VariantInventory

from .era_detect import detect_legacy_no_value_negative
from .loader import load_vocab_manager
from .saver import save_vocabulary
from .variant_registration import iter_variant_infos

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
    *,
    insert_value_negative: bool = False,
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

    `insert_value_negative`: DEFAULT (tiebreak) era for per-binary CSVs,
    NOT a uniform override. Each CSV's era is resolved per-file by
    :func:`era_detect.detect_legacy_no_value_negative` from its own
    token-stream carrier coherence; ``insert_value_negative`` only
    decides the era when detection is inconclusive (legacy data is
    carrier-blind, and degenerate / too-short CSVs carry no signal). A
    confident MODERN detection always wins over this default, so a
    MIXED-era corpus (legacy untouched @256 + modern re-tokenized @257)
    unifies correctly: pass the legacy default (``True``) and the modern
    files self-upgrade to 257 while the legacy files keep 256.

    When a CSV is resolved to the legacy era, it is loaded with only 256
    reserved slots (digits only) — the first entry of its per-binary
    vocabulary is a real token at per-binary id 256, NOT
    ``value_negative``. The unified vocab still gets the canonical
    257-reserved layout (``value_negative`` pinned at unified slot 256),
    and the per-binary real tokens 256+ are remapped via
    :meth:`register_on_vocab_manager` into the unified id space (legacy
    id 256 = ``block_v2`` typically lands at unified id 264). The emitted
    mapping.b64c sidecars carry the shifted ids, so downstream consumers
    (memmap_builder) see canonical-layout unified ids regardless of which
    era each CSV was tokenized in.
    """
    if mapping_output_dir is not None and mapping_source_root is None:
        raise ValueError(
            "unify_vocab: mapping_source_root is required when "
            "mapping_output_dir is set, so per-CSV subdir layout can be "
            "preserved on the output side."
        )

    if mapping_output_dir is not None:
        mapping_output_dir.mkdir(parents=True, exist_ok=True)

    # Unified vocab layout (plan immutable-whistling-twilight.md):
    #   * IDs 0..255             — protocol-reserved digit slots.
    #   * ID  256                — `value_negative` postfix marker.
    #   * IDs 257..263 (NUMBER)  — `valued_const_v2`, `float16`, ...,
    #                              `float128` (source-declaration order).
    #   * IDs 264..271 (IDENTITY)— `block_v2`, `local_func`, `plt_func`,
    #                              `ext_func`, `string_ptr`, `jump_table`,
    #                              `ro_data_ptr`, `rw_data_ptr`.
    #   * IDs 272..X             — instruction representatives merged from
    #                              the per-binary CSVs.
    #   * IDs X+1..Y (TAIL)      — metadata-variant block in axis-grouped
    #                              order (positional axes first, then
    #                              sidecar-key axes alphabetical).
    # The unified VM is built at ``MEMMAP_FORMAT_VERSION`` from the start
    # so ``_private_add_token`` recognises the reserved-digit layout for
    # both canonical-block and instruction registrations.
    unified_vm = VocabularyManager(platform=None, format_version=MEMMAP_FORMAT_VERSION)
    assert len(unified_vm.id_to_token) == VocabularyManager._V2_RESERVED_TOKEN_COUNT

    # Pre-register the canonical NUMBER + IDENTITY blocks at fixed slots
    # 257..271. After this, the next caller-driven `_private_add_token`
    # lands at id ``_V2_EAGER_BLOCK_END`` (= 272).
    unified_vm._register_v2_canonical_blocks()
    assert len(unified_vm.id_to_token) == VocabularyManager._V2_EAGER_BLOCK_END

    # Variant collection runs inline alongside the instruction merge
    # below; registration is deferred to Pass 3 so the variant block lands
    # as a contiguous tail above the last instruction representative.
    variant_inventory = VariantInventory()

    # Pass 2 — instruction-representative walk against each per-binary
    # vocab CSV. The mapping array translates the per-binary id space
    # into the unified id space; the unified vocab reuses v2's
    # reserved-prefix semantics so the identity remap for IDs 0..256
    # (digits 0..255 + `value_negative` at slot 256) stays valid. Variant
    # collection from the same CSV's filename + optional sidecar runs
    # alongside the merge so the sidecar walk does not need a second pass.
    loaded_count = 0
    for csv_file in csv_files:
        print(f"Loading vocabulary from {csv_file}")
        # Per-CSV era resolution. `insert_value_negative` is only the
        # DEFAULT/tiebreak here — the detector positively confirms the
        # modern (offset-257) era from the CSV's own token-stream carrier
        # coherence and overrides the default in that case, so a MIXED-era
        # corpus (legacy untouched + modern re-tokenized) unifies
        # correctly regardless of the single global flag. Legacy and
        # degenerate CSVs keep the default. See `era_detect` for why the
        # signal is one-directional (modern is positively detectable;
        # legacy is carrier-blind and must default).
        legacy_no_value_negative = detect_legacy_no_value_negative(
            csv_file, default=insert_value_negative,
        )
        current_vocab_manager = load_vocab_manager(
            csv_file, legacy_no_value_negative=legacy_no_value_negative,
        )
        if current_vocab_manager is None:
            logger.error(f"Failed to load vocabulary from {csv_file}. Missing or incomplete (no vocab def in last line).")
            continue
        loaded_count += 1

        current_format_version = current_vocab_manager.format_version
        # Per-binary CSVs are always v2 under the plan (variant tokens
        # live ONLY in the unified vocab). Anything else here would
        # mean someone fed the unifier a stale or mis-built CSV — most
        # commonly a unified vocab file mistaken for a per-binary one;
        # refuse explicitly rather than silently mis-decoding the
        # reserved-id band.
        if current_format_version != 2:
            raise ValueError(
                f"unify_vocab: per-binary CSV must be format_version=2; "
                f"{csv_file} reports format_version={current_format_version}. "
                f"Re-tokenize the corpus or check whether the input was "
                f"accidentally a unified vocab instead of a per-binary one."
            )

        # Collect variant-axis tokens implied by this CSV's filename +
        # optional `_meta.json` sidecar. `iter_variant_infos` already
        # owns the skip-on-parse-error policy; the inventory deduplicates
        # across the corpus and `iter_tokens_axis_grouped` (Pass 3 below)
        # owns the deterministic registration order.
        variant_inventory.update(iter_variant_infos([csv_file]))

        mappings = np.full_like(current_vocab_manager.id_to_token_type, -1, dtype=np.int32)

        # Modern path: per-binary IDs 0..256 are protocol-reserved (256
        # digits + value_negative at slot 256). Both per-binary VM and
        # unified VM agree by construction, so identity remap is the
        # correct translation across the prefix.
        #
        # Legacy path (``legacy_no_value_negative=True``): per-binary IDs
        # 0..255 are digits, and slot 256 is the FIRST REAL TOKEN (the
        # per-binary VM has 256 reserved, not 257). Only the digit
        # prefix identity-remaps; slot 256+ is filled by
        # ``register_on_vocab_manager`` below. The flag is the PER-CSV
        # detected era (see the load above), not the global default, so
        # the prefix translation matches the era each CSV was loaded with.
        if legacy_no_value_negative:
            reserved = VocabularyManager._V2_RESERVED_DIGIT_COUNT
            mappings[:reserved] = np.arange(reserved, dtype=mappings.dtype)
        else:
            reserved = VocabularyManager._V2_RESERVED_TOKEN_COUNT
            mappings[:reserved] = np.arange(reserved, dtype=mappings.dtype)
            assert (
                mappings[VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID]
                == VocabularyManager._V2_VALUE_NEGATIVE_TOKEN_ID
            ), "value_negative identity remap mismatch"

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

    # Pass 3 — register every collected variant-axis token at the tail of
    # the unified vocab in axis-grouped order. `iter_tokens_axis_grouped`
    # yields positional axes (`arch`, `comp`, `cver`, `opt`) in declared
    # order first, then sidecar-key axes alphabetical-by-prefix. The
    # `Variant_Axis` Inner registers each string at the next free id; the
    # multiset is identical to alphabetical iteration but the order is the
    # one the dataloader's positional decode wants.
    n_variants = 0
    for token_str in variant_inventory.iter_tokens_axis_grouped():
        unified_vm.Variant_Axis(token_str)
        n_variants += 1

    # Hard post-condition: the unified vocab must fit in uint16 because
    # `_data.bin` and `_variants.bin` both serialise IDs as uint16.
    # Silent overflow would corrupt the dataset; surface as ValueError
    # with a remediation hint so the operator can split the corpus by
    # arch and unify per slice instead.
    total_ids = len(unified_vm.id_to_token)
    if total_ids > _UINT16_CEILING:
        # `_V2_EAGER_BLOCK_END` (= 272) covers the 256-entry reserved digit
        # band, the `value_negative` marker at slot 256, and the eagerly-
        # registered canonical number+identity blocks at slots 257..271.
        # Instruction representatives + variant tokens together fill
        # everything past that prefix, so the remaining count after
        # subtracting eager-block and variant is purely instruction-
        # representative tokens.
        raise ValueError(
            f"unify_vocab: unified vocab has {total_ids} tokens "
            f"({n_variants} variant + "
            f"{total_ids - VocabularyManager._V2_EAGER_BLOCK_END - n_variants} "
            f"instruction), exceeds uint16 ceiling ({_UINT16_CEILING}). "
            f"Split the corpus by architecture and unify each slice "
            f"separately, or prune low-frequency tokens before unifying."
        )

    print(f"Saving unified vocabulary to {unified_vocab_file} "
          f"({loaded_count}/{len(csv_files)} CSVs loaded; "
          f"{n_variants} variant-axis tokens registered)")
    with open(unified_vocab_file, "w", newline="", encoding="ascii") as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        save_vocabulary(unified_vm, writer)
