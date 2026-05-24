import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from dynamic_runner.worker import Task

from dynrunner.tokenize import TokenizerPhase
from tokenizer.compact_base64_utils import base64_to_ndarray_vec, ndarray_to_base64
from tokenizer.fill_constant_candidates import fill_constant_candidates
from tokenizer.function_data_manager import FunctionData, FunctionDataManager
from tokenizer.function_deduper import FunctionDeduper, canonical_function_name
from tokenizer.function_filter import FunctionFilter
from tokenizer.function_token_list import FunctionTokenList
from tokenizer.opaque_remapping import (
    apply_opaque_mapping,
    apply_opaque_mapping_raw_optimized,
)
from tokenizer.string_sidecar import StringSidecar
from tokenizer.tokens import Category, TokenResolver
from tokenizer.vocab_unifier import save_vocabulary

VERIFICATION: bool = False


# v2 metadata-column ordering. The keys appear in this fixed order in the
# emitted JSON so byte-identical input produces byte-identical output (no
# reliance on Python's dict-iteration order quirks across versions). A
# category whose per-function list is empty is OMITTED from the JSON to
# keep the column compact for the common case (most functions touch only
# a handful of categories). Block identities are intentionally absent —
# blocks carry no human-readable metadata in v2 (see ``_emit_block``),
# and per-function block ranges are recoverable from the token stream.
_V2_METADATA_KEY_ORDER: list[tuple[Category, str]] = [
    (Category.LOCAL_FUNC,  "local_funcs"),
    (Category.PLT_FUNC,    "plt_funcs"),
    (Category.EXT_FUNC,    "ext_funcs"),
    (Category.RO_DATA_PTR, "ro_data_ptr"),
    (Category.RW_DATA_PTR, "rw_data_ptr"),
    (Category.STRING_PTR,  "string_ptr"),
    (Category.JUMP_TABLE,  "jump_tables"),
]


def _strings_path_for(csv_path: Path) -> Path:
    """Derive the per-binary ``_strings.bin`` sidecar path from the CSV path.

    Both paths share a ``<base>`` prefix (see ``tokenizer/output_filename.py``);
    the CSV ends in ``_output.csv`` and the sidecar ends in ``_strings.bin``.
    """
    csv_path = Path(csv_path)
    name = csv_path.name
    suffix = "_output.csv"
    if name.endswith(suffix):
        base = name[: -len(suffix)]
    else:
        # Defensive fallback — strip a trailing extension only.
        base = csv_path.stem
    return csv_path.with_name(f"{base}_strings.bin")


def _build_v2_string_ptr_entry(
    entry: dict, sidecar: StringSidecar
) -> dict:
    """Register a ``string_ptr`` entry's bytes in the sidecar and return
    the cleaned per-row metadata dict ``{line, start_offset, encoding}``.

    The classifier (``constant_handler._emit_string_ptr``) stashes the
    raw string bytes + the originating string's base address under
    underscore-prefixed internal keys (``_string_bytes``,
    ``_string_encoding``, ``_start_addr``); this writer is the single
    place that consumes them and rewrites the entry to the wire shape
    documented in the plan ("Wire format > Per-function CSV"). The
    underscore-prefix convention keeps the placeholder fields obviously
    internal and lets the cleanup step here be a single pass.

    ``start_offset`` semantics: when the constant ``addr`` equals the
    string's base ``_start_addr``, the offset is 0 (whole-string load).
    When ``addr > _start_addr`` the constant is a substring access N
    bytes into the string and ``start_offset = N``. If ``_start_addr``
    is missing from the classifier-side metadata (provider lookup did
    not surface a base), we fall back to ``start_offset = 0`` —
    conservative but lossy for genuine substring accesses; the loss is
    visible in the strings sidecar (the full string is still recorded
    once at its true base; the offset just doesn't point at the right
    substring start).
    """
    raw_bytes = entry.get("_string_bytes")
    encoding = entry.get("_string_encoding") or entry.get("encoding") or "unknown"
    if raw_bytes is None:
        # No bytes captured — emit a sentinel triplet so the column shape
        # is preserved. A missing string is a classifier-side bug; we
        # don't want to silently drop the entry (which would shift
        # downstream identity-indexed lookups).
        line = -1
        start_offset = 0
    else:
        line = sidecar.add(raw_bytes, encoding)
        addr_hex = entry.get("addr")
        start_addr = entry.get("_start_addr")
        if addr_hex is not None and start_addr is not None:
            try:
                value = int(addr_hex, 16) if isinstance(addr_hex, str) else int(addr_hex)
                start_offset = max(0, value - int(start_addr))
            except (TypeError, ValueError):
                start_offset = 0
        else:
            start_offset = 0
    return {"line": line, "start_offset": start_offset, "encoding": encoding}


def _build_v2_metadata_json(
    resolver: TokenResolver, sidecar: StringSidecar
) -> str:
    """Convert the resolver's per-category metadata into the v2 wire-format
    JSON string for a single function's CSV row.

    Categories with an empty list are omitted (compactness; the reader
    treats a missing key as an empty list per the plan). The output is
    a single-line compact JSON string (no whitespace separators) so the
    CSV column stays narrow.
    """
    out: dict[str, list[dict]] = {}
    for category, key in _V2_METADATA_KEY_ORDER:
        entries = resolver.metadata.get(category, [])
        if not entries:
            continue
        if category is Category.STRING_PTR:
            cleaned = [_build_v2_string_ptr_entry(e, sidecar) for e in entries]
        else:
            # Drop any leading-underscore placeholder fields so we don't
            # leak classifier-side internals into the wire format. The
            # convention is general: keys starting with ``_`` are
            # internal-only across all emitters.
            cleaned = [
                {k: v for k, v in e.items() if not (isinstance(k, str) and k.startswith("_"))}
                for e in entries
            ]
        out[key] = cleaned
    return json.dumps(out, separators=(",", ":"))


def build_vocab_tokenize_and_index(
    func_tokens: FunctionTokenList,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if func_tokens.last_index == 0:
        return (
            np.array([], dtype=np.int_),
            np.array([], dtype=np.int_),
            np.array([], dtype=np.int_),
        )

    (token_ids, _, _, _, insn_idx_run_lengths, _, block_insn_run_lengths, _, _) = func_tokens.get_used_arrays()

    block_insn_split_start_indicies = np.cumsum(np.insert(block_insn_run_lengths[:-1], 0, 0))
    block_idx_run_lengths = np.add.reduceat(insn_idx_run_lengths, block_insn_split_start_indicies)

    return token_ids, block_idx_run_lengths, insn_idx_run_lengths


def main_loop(
    instr_sets,
    provider,
    constant_list,
    func_addr_range,
    func_disas,
    func_disas_token,
    func_name_addr,
    func_names,
    lookup,
    resolver,
    text_end,
    text_start,
    vocab_manager,
    csv_path,
    arch_provider,
    logger: logging.Logger,
    task: Task,
    **_kwargs,
) -> tuple[FunctionDataManager, int]:
    logger.info("Preparing main loop")

    filter = FunctionFilter(logger)

    total_functions = provider.function_count()
    function_manager = FunctionDataManager(total_functions) if VERIFICATION else FunctionDataManager(0)
    # Semantic-merge gate consulted before every CSV row write + every
    # ``FunctionDataManager.add_function_data`` call (see
    # ``tokenizer/function_deduper.py``). Per-binary state; instantiated
    # once here so the same gate covers the whole iter_functions pass.
    function_deduper = FunctionDeduper()

    exceptions = []
    filtered_count = 0
    last_keepalive_time = time.time()

    is_v2 = getattr(vocab_manager, "format_version", 1) == 2
    metadata_column_name = "metadata" if is_v2 else "opaque_metadata"

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        print("WRITING OUTPUT")
        # v2-only: string-bytes sidecar opened alongside the CSV.
        # Created unconditionally under v2 (even if the binary has zero
        # string constants) so the framework's per-binary publishing
        # step has a stable file set. Closed at the bottom of this
        # ``with`` (after the CSV flush) — the main loop's own broad
        # ``try/except`` (below, around the per-function body) absorbs
        # every per-function exception and the outer ``except`` around
        # the iteration absorbs unrecoverable iteration failures, so
        # control always reaches the ``sidecar.close()`` call. No
        # ``try/finally`` needed here.
        sidecar: Optional[StringSidecar] = (
            StringSidecar(_strings_path_for(Path(csv_path))) if is_v2 else None
        )
        writer = csv.writer(csvfile, lineterminator='\n')
        if is_v2:
            # Prelude row: single cell announcing the wire-format
            # version. Readers MUST consume this row before parsing
            # the header. v1 outputs lack the prelude; readers
            # detect v1 by absence (the header row's first cell is
            # ``function_name`` directly).
            writer.writerow(["version=2"])
        writer.writerow(
            [
                "function_name",
                "occurrence",
                "tokens_base64",
                "block_runlength_base64",
                "instruction_runlength_base64",
                metadata_column_name,
            ]
        )

        occurence = 0
        prev_func_name = ""

        logger.info("Starting main loop")
        task.set_phase(TokenizerPhase.TOKENIZATION.value)

        try:
            pbar = tqdm(
                iterable=provider.iter_functions(),
                total=total_functions,
                desc="Retrieving data from alllll functions. Like a big boy.",
            )
            for i, (func_addr, func_name, func) in enumerate(pbar):
                current_time = time.time()
                # 5s cadence (was 200ms in the hand-rolled era) — the
                # framework's stage_timeouts are minute-scale and the
                # new manager-side check_timeouts only needs proof of
                # life within the configured window.
                if (current_time - last_keepalive_time) >= 5.0:
                    task.keepalive()
                    last_keepalive_time = current_time

                resolver.reset()

                try:
                    (function_analysis) = fill_constant_candidates(
                        func_addr=func_addr,
                        func=func,
                        instr_sets=instr_sets,
                        constant_dict=constant_list,
                        lookup=lookup,
                        text_start=text_start,
                        text_end=text_end,
                        resolver=resolver,
                        vocab_manager=vocab_manager,
                        arch_provider=arch_provider,
                        disasm_provider=provider,
                    )
                except Exception as e:
                    logger.warning(f"Error processing {func_name}: {e}. Skipping function.")
                    exceptions.append(e)
                    continue

                if function_analysis is None:
                    continue

                (temp_bbs, block_list, block_dict, constant_handler, func_tokens) = function_analysis

                func_addr_range[func_addr] = sorted(block_list, key=lambda d: list(d.values())[0][0])

                # v1: legacy frequency-sort remapping pass. v2 identities
                # are monotonic at allocation time (no global re-sort), so
                # ``create_opaque_mapping`` and ``apply_opaque_mapping_*``
                # are v1-only. The v2 stubs in ConstantHandler raise
                # NotImplementedError, so calling them in v2 would crash —
                # the explicit branch keeps the v1 path verbatim while
                # making the v2 short-circuit explicit.
                if not is_v2:
                    opaque_mapping = constant_handler.create_opaque_mapping()

                    if len(opaque_mapping) > 0:
                        func_tokens = apply_opaque_mapping_raw_optimized(
                            func_tokens, opaque_mapping, vocab_manager, constant_handler
                        )
                        if VERIFICATION:
                            temp_bbs = apply_opaque_mapping(temp_bbs, opaque_mapping, constant_handler=None)

                if VERIFICATION:
                    for x, y in zip(
                        [token for (_, block) in temp_bbs for insn in block for token in insn],
                        func_tokens.iter_raw_tokens(),
                    ):
                        if x != y:
                            print(f"Token mismatch: {x} != {y}")
                            raise ValueError("Token mismatch in disassembly list")

                # v1 aggregates a single per-function metadata structure
                # via ``get_metadata_list_by_opaque_id`` (a single ordered
                # list indexed by opaque id). v2 reads the per-category
                # metadata lists off the TokenResolver and serializes them
                # as JSON via ``_build_v2_metadata_json`` (which also
                # registers any string_ptr bytes in the sidecar).
                if is_v2:
                    meta_result = None  # not used in the v2 row build
                else:
                    meta_result = constant_handler.get_metadata_list_by_opaque_id()

                tokenized_instructions, block_run_lengths, insn_run_lengths = build_vocab_tokenize_and_index(
                    func_tokens
                )

                if len(tokenized_instructions) == 0:
                    continue

                try:
                    tokens_base64 = ndarray_to_base64(tokenized_instructions)
                    block_base64 = ndarray_to_base64(block_run_lengths)
                    insn_base64 = ndarray_to_base64(insn_run_lengths)
                    # Canonical-name derivation (cross-ISA-stable): the
                    # CSV column 0, the occurrence sentinel, and the
                    # function-names sidecar all consume the canonical
                    # name produced from the same three identity axes
                    # the deduper consults. ``func_name`` (the raw
                    # provider name) is preserved only for log /
                    # diagnostic call sites below; never written to the
                    # output.
                    identity_key = getattr(func, "identity_key", None)
                    comment = getattr(func, "comment", None)
                    canonical_name = canonical_function_name(
                        func_name, comment, identity_key
                    )
                    if prev_func_name == canonical_name:
                        occurence += 1
                    else:
                        occurence = 0
                    writer = csv.writer(csvfile, lineterminator='\n')
                    if filter.filter_fns(func_tokens, func_name, vocab_manager):
                        occurence -= 1
                        filtered_count += 1
                        continue

                    # Semantic-merge gate: the deduper consumes the
                    # four-axis identity tuple ``(name, comment,
                    # identity_key, body)`` and returns
                    # ``is_duplicate=True`` ⇒ this function was already
                    # written (fold: no CSV row, no FDM record, no
                    # occurrence bump). Otherwise the row is written
                    # with the CANONICAL name as column 0 (so caller-side
                    # metadata.name and callee-side function-definition
                    # name are byte-identical for the downstream
                    # function_lookup[(name, vkey)] resolver). The
                    # legacy ``occurrence`` column still disambiguates
                    # the body-divergence diagnostic case (same canonical
                    # name + different body).
                    resolution = function_deduper.resolve(
                        func_name, comment, identity_key, tokens_base64
                    )
                    if resolution.is_duplicate:
                        # Roll back the occurrence bump performed above:
                        # the duplicate never reaches the writer, so the
                        # next same-name function should land at the
                        # same occurrence the duplicate would have taken
                        # (legacy ordering preserved for non-folded
                        # entries). When ``occurence`` was reset to 0
                        # for a first-seen name, the rollback restores
                        # the prev_func_name sentinel.
                        if prev_func_name == canonical_name:
                            occurence -= 1
                        continue
                    if resolution.body_divergence_warning:
                        logger.warning(
                            "Body-divergence under same identity tuple for "
                            "%r (comment=%r, identity_key=%r); allocating "
                            "fresh slot",
                            func_name,
                            comment,
                            identity_key,
                        )

                    if is_v2:
                        # v2 metadata column: JSON-serialized per-category
                        # metadata. ``_build_v2_metadata_json`` is also
                        # responsible for registering each ``string_ptr``
                        # entry's bytes in the sidecar (and rewriting
                        # placeholder ``line``/``start_offset`` fields).
                        metadata_cell = _build_v2_metadata_json(resolver, sidecar)
                    else:
                        # v1 verbatim: ``repr()`` of the aggregated
                        # per-function metadata list.
                        metadata_cell = str(repr(meta_result))

                    row = [
                        canonical_name,
                        occurence,
                        tokens_base64,
                        block_base64,
                        insn_base64,
                        metadata_cell,
                    ]

                    writer.writerow(row)
                    prev_func_name = canonical_name

                    if i & 16383 == 16383:
                        save_vocabulary(vocab_manager, writer)

                    if i & 255 == 255:
                        csvfile.flush()

                    if VERIFICATION:
                        assert np.all(base64_to_ndarray_vec(tokens_base64) == tokenized_instructions), (
                            "Base64 conversion failed for tokens"
                        )
                        assert np.all(base64_to_ndarray_vec(block_base64) == block_run_lengths), (
                            "Base64 conversion failed for block run lengths"
                        )
                        assert np.all(base64_to_ndarray_vec(insn_base64) == insn_run_lengths), (
                            "Base64 conversion failed for instruction run lengths"
                        )
                        function_data = FunctionData(
                            tokens=func_tokens,
                            tokens_base64=tokens_base64,
                            block_runlength_base64=block_base64,
                            instruction_runlength_base64=insn_base64,
                            # ``metadata_cell`` carries whichever
                            # metadata payload was written to the CSV:
                            # the v1 ``opaque_metadata`` repr or the v2
                            # ``metadata`` JSON cell. The wire-format
                            # column name (chosen above via
                            # ``metadata_column_name``) is the only
                            # version-dependent thing; the in-memory
                            # field is just the cell's content.
                            metadata_cell=metadata_cell,
                        )
                        fdm_final_name = function_manager.add_function_data(
                            func_name,
                            func_addr,
                            temp_bbs,
                            func_tokens,
                            function_data,
                            identity_key=identity_key,
                            comment=comment,
                        )

                        func_name_addr[fdm_final_name] = func_addr
                        func_disas[fdm_final_name] = temp_bbs
                        func_disas_token[fdm_final_name] = func_tokens
                        func_names.append(fdm_final_name)
                except Exception as e:
                    logger.warning(
                        f"Error saving {func_name}: {e}.\n"
                        f"Tokenstream: {func_tokens}\n"
                        f"Tokens: {tokenized_instructions}\n"
                        f"Block encoding: {block_run_lengths}\n"
                        f"Instructions: {insn_run_lengths}\n"
                        f"MetaData: {str(meta_result)}"
                    )
                    exceptions.append(e)
                    continue
        except Exception as e:
            print(f"Unrecoverable error in main loop: {e}, writing what we have at least")
            exceptions.append(e)

        task.keepalive()

        save_vocabulary(vocab_manager, writer)
        csvfile.flush()

        # v2 sidecar close — paired with the open at the top of this
        # ``with`` block. Idempotent if already closed.
        if sidecar is not None:
            sidecar.close()

    if len(exceptions) > 0:
        # Per-function errors were already log+continue (see inner
        # except clauses). Don't re-raise here — that would invalidate
        # an otherwise-complete CSV and cause the framework to retry
        # the whole binary. The warning_handler's count is the signal
        # the caller actually uses to surface degraded data quality.
        logger.info(
            f"main_loop completed with {len(exceptions)} caught exceptions; "
            f"CSV written with surviving functions."
        )

    if VERIFICATION:
        function_manager.compact_arrays()

    return function_manager, filtered_count
