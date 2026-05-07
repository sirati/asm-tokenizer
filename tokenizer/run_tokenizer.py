import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import cast

from dynamic_runner.worker import Task

from dynrunner.tokenize import TokenizerPhase
from shared import setup_logger
from tokenizer.arch import Platform, get_provider
from tokenizer.disasm import DisassemblyProvider, get_disassembly_provider
from tokenizer.hash_checked_pickles import (
    has_valid_pickle,
    save_pickle,
    try_load_pickle,
)
from tokenizer.main_loop import main_loop
from tokenizer.output_filename import format_output_basename
from tokenizer.output_staging import staged_publish
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenResolver
from tokenizer.variant_info import VariantInfo
from tokenizer.vocab_unifier.loader import load_vocab_manager

DO_PICKLES: bool = True

# Output-filename suffixes appended to the canonical-format
# ``<base>`` (computed in ``tokenizer.output_filename``). The CSV
# carries the token stream, the meta sidecar carries the variant's
# canonical-4 + extra_metadata; ``build_memmap``'s pairing walk
# matches them by their shared ``<base>`` prefix.
_OUTPUT_CSV_SUFFIX = "_output.csv"
_META_SIDECAR_SUFFIX = "_meta.json"


def _write_meta_sidecar(
    stage_dir: Path, output_basename: str, variant_info: VariantInfo
) -> None:
    """Persist the per-variant metadata next to the staged CSV.

    The file is written into the staging dir so ``staged_publish``'s
    rglob picks it up alongside the CSV + ``_consts.txt`` and atomic-
    publishes all three under one transaction. ``output_basename`` is
    the canonical-format shared prefix produced by
    ``tokenizer.output_filename.format_output_basename``; the meta
    filename is ``<base>_meta.json`` — paired with ``<base>_output.csv``
    by ``build_memmap``'s discovery walk.

    The serialised payload is a verbatim ``dataclasses.asdict`` of the
    VariantInfo. ``extra_metadata`` flows through opaque (the dataclass
    field type is already ``dict[str, Any]``, JSON-serialisable by
    construction).
    """
    meta_path = stage_dir / f"{output_basename}{_META_SIDECAR_SUFFIX}"
    meta_path.write_text(json.dumps(dataclasses.asdict(variant_info), indent=2))


class NonRecoverableTokenizerError(Exception):
    """Marker exception for binary-deterministic tokenizer failures
    (e.g. known-bad inputs that trip an angr CFG resolver bug). The
    worker maps this to `ErrorType.NON_RECOVERABLE` so the framework
    skips the retry pass on the affected task.
    """


def disassemble_to_tokens(
    out_folder: Path,
    binary_name: str,
    platform: Platform,
    provider: DisassemblyProvider,
    constant_list: dict[str, list[str]],
    csv_path: Path,
    binary_path: Path,
    pickle_mainloop_file_path: Path,
    logger: logging.Logger,
    task: Task,
    backend: str = "angr",
    with_pickled=False,
    do_pickles=True,
    **kwargs,
):
    if not with_pickled:
        task.set_phase(TokenizerPhase.ANGR_2.value)
        func_names = []
        block_runlength_dict = {}
        insn_runlength_dict = {}
        opaque_meta_dict = {}

        opaque_const_meta: dict[str, list[str]] = {}

        func_addr_range: dict[int, list[dict[str, tuple[str, str]]]] = {}
        func_disas: dict[str, list[dict[str, list[str]]]] = {}

        text_start, text_end = provider.get_text_section_bounds()
        lookup = provider.create_metadata_lookup()

        func_disas_token: dict[str, list[dict[str, list[str]]]] = {}

        func_name_addr = {}

        kwargs = dict(
            block_runlength_dict=block_runlength_dict,
            provider=provider,
            constant_list=constant_list,
            func_addr_range=func_addr_range,
            func_disas=func_disas,
            func_disas_token=func_disas_token,
            func_name_addr=func_name_addr,
            func_names=func_names,
            insn_runlength_dict=insn_runlength_dict,
            lookup=lookup,
            opaque_const_meta=opaque_const_meta,
            opaque_meta_dict=opaque_meta_dict,
            text_end=text_end,
            text_start=text_start,
        )

        if do_pickles:
            save_pickle(pickle_mainloop_file_path, kwargs)

    else:
        kwargs.update(dict(provider=provider, constant_list=constant_list))
        func_names = kwargs["func_names"]

    vocab_manager = VocabularyManager(platform)

    resolver = TokenResolver()
    arch_provider = get_provider(platform, backend)
    instr_sets = arch_provider.load_instruction_sets()

    kwargs.update(
        dict(resolver=resolver, instr_sets=instr_sets, arch_provider=arch_provider, csv_path=csv_path, logger=logger)
    )

    function_manager, filtered = main_loop(vocab_manager=vocab_manager, task=task, **kwargs)

    return (func_names, function_manager, vocab_manager, filtered)


def run_tokenizer(
    binary_path: Path,
    platform: Platform | str,
    skip_existing_csv: bool,
    source_dir: Path,
    output_dir: Path,
    task: Task,
    backend: str = "angr",
    variant_info: VariantInfo | None = None,
    source_relative_path: Path | None = None,
    output_basename: str | None = None,
) -> tuple[int, int]:
    """Tokenize one binary; return ``(warnings, filtered)``.

    Returns ``(-1, -1)`` for the skip-existing fast-path so the caller
    can map that to the framework's "already done" Done envelope (a
    convention the local manager treats as a no-warning skip).

    Per-variant metadata flows through three optional parameters that
    the standalone CLI doesn't need to wire:

    * ``variant_info``: drives the per-variant metadata
      ``<base>_meta.json`` sidecar AND (when ``output_basename`` is
      ``None``) the canonical-format output filename composed by
      ``tokenizer.output_filename`` (the single source of truth shared
      with
      ``dynrunner.tokenize.tokenizer_task.get_output_filename_pattern``).
      ``None`` → recovered from ``binary_path.name`` via
      ``VariantInfo.from_legacy_filename``, preserving byte-identical
      output for the standalone CLI's legacy 4-axis filenames (the
      canonical reconstruction round-trips legacy filenames bit-for-bit
      because ``pkg`` equals the parsed ``binary_name`` slot).
    * ``source_relative_path``: subdir under ``output_dir`` to write
      into (mirrors the source-tree layout). ``None`` → derived from
      ``binary_path.relative_to(source_dir)`` like before. Sidecar
      mode passes the JSON sidecar's path-under-source so output
      lands next to where the binary "logically" lives in the source
      tree, not under the per-task scratch dir where the tarball got
      extracted.
    * ``output_basename``: explicit ``<base>`` (without
      ``_output.csv``/``_meta.json`` suffix) for the emitted files.
      Multi-binary sidecar tarballs need this because all N binaries
      in one tarball share a single ``VariantInfo`` (per-package
      metadata), but each emits its own pair of files distinguished
      by the binary's archive-member basename in the filename slot.
      ``None`` → composed from ``variant_info`` via
      ``format_output_basename`` (the legacy 1:1 pkg-equals-binary
      case). The meta sidecar's serialised payload is unchanged either
      way — it's always ``dataclasses.asdict(variant_info)`` so all N
      meta sidecars in a multi-binary tarball carry identical
      per-package metadata.
    """
    logger, warning_handler = setup_logger("tokenizer")
    logger.info("STARTING DISASSEMBLY")
    task.set_phase(TokenizerPhase.ANGR_1.value)

    file_path: Path = binary_path.absolute()

    binary_name = file_path.name

    if platform == "auto":
        # Order: longer prefixes first so `mips64-...` matches "mips64"
        # before "mips32" (and similarly for ppc, riscv). Within each
        # family, the 64-bit variant comes first.
        _ALL_PLATFORMS: list[Platform] = [
            "x86",
            "x64",
            "arm64",
            "arm32",
            "mips64",
            "mips32",
            "ppc64",
            "ppc32",
            "riscv64",
            "riscv32",
        ]
        detected_platform: Platform | None = None
        for p in _ALL_PLATFORMS:
            if binary_name.startswith(p):
                detected_platform = p
                break

        if detected_platform is None:
            raise ValueError(
                f"Could not detect platform from binary name '{binary_name}'. "
                f"Expected one of: {', '.join(_ALL_PLATFORMS)}"
            )
        platform = cast(Platform, detected_platform)
        logger.info(f"[*] Detected platform: {platform}")
    elif variant_info is None:
        # The "filename starts with platform" guard exists to catch a
        # mis-routed legacy binary (where the filename encodes the
        # platform). It only applies to legacy callers (standalone
        # CLI): no variant_info → identity is filename-derived → the
        # startswith check is meaningful. Sidecar callers pass
        # ``variant_info`` from a JSON sidecar; the extracted binary's
        # on-disk name (``hello``, ``busybox``, ...) carries no
        # platform prefix, so the variant's arch is the authoritative
        # source instead and the filename-startswith guard would
        # spuriously fire.
        assert binary_name.startswith(platform), (
            f"Binary name '{binary_name}' must start with platform '{platform}'. Wrong platform's file in queue?"
        )

    # Route ARM and MIPS binaries to Ghidra: angr's CFG resolvers
    # have known bugs on these platforms (arm_elf_fast.py:89 IndexError
    # on ARM ELF indirect jumps; missing-VEX `NotImplementedError:
    # report bug to @rhelmot` on MIPS). Ghidra's analyzer handles
    # both classes cleanly. Other platforms keep whatever backend
    # was requested (default angr — fast, stable for x86/x64/ppc/riscv).
    if platform in {"arm32", "arm64", "mips32", "mips64"} and backend == "angr":
        logger.info(
            f"[*] Switching to ghidra backend for {platform} "
            f"(angr CFG bugs on ARM/MIPS would cause permanent failures)."
        )
        backend = "ghidra"

    if source_relative_path is not None:
        # Caller supplied the canonical source-relative path (sidecar
        # mode: the JSON sidecar's path-under-source, NOT the
        # per-task scratch dir where the binary got extracted). Use
        # it verbatim so output mirrors source-tree layout.
        relative_path = source_relative_path
    else:
        try:
            relative_path = file_path.relative_to(source_dir.absolute())
        except ValueError:
            relative_path = Path(file_path.parent.name) / file_path.name

    out_folder = output_dir / relative_path.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    # Pickle files are angr CFG-state caches used to skip stage-1 prep
    # on a retry of the same binary. They're debug artifacts, NOT
    # durable outputs. In SLURM dispatch they should live on the
    # ephemeral `/app/out-tmp` mount (auto-cleaned by the wrapper trap
    # on container exit) so they don't fill the gateway-durable
    # `/app/out-network` mount with ~2 MB × 2 pickles per binary.
    # Outside the wrapper container (`/app/out-tmp` doesn't exist),
    # fall back to writing pickles next to the CSV — preserves the
    # standalone CLI's existing layout.
    out_tmp_root = Path("/app/out-tmp")
    if out_tmp_root.exists():
        pickle_folder = out_tmp_root / relative_path.parent
        pickle_folder.mkdir(parents=True, exist_ok=True)
    else:
        pickle_folder = out_folder
    pickle_file_path = pickle_folder / f"{binary_name}.pkl"
    pickle_mainloop_file_path = pickle_folder / f"{binary_name}.mainloop.pkl"

    # Resolve the VariantInfo that drives the meta sidecar (and, when
    # the caller doesn't supply an explicit ``output_basename``, also
    # drives the canonical-format output filename). Standalone CLI
    # callers pass nothing → recover from the binary's legacy 4-axis
    # filename; worker handler passes the decoded payload. The default
    # output basename is composed from the variant's canonical-4 + pkg
    # + variant_id via ``tokenizer.output_filename`` so the task-side
    # ``get_output_filename_pattern`` and this writeout agree on the
    # filename byte-for-byte. Multi-binary sidecar callers override
    # ``output_basename`` per archive member so each binary in the
    # same tarball lands at its own filename (their VariantInfo —
    # including ``pkg`` — stays unchanged because per-package metadata
    # applies identically to every binary in the package).
    if variant_info is None:
        variant_info = VariantInfo.from_legacy_filename(binary_path)
    if output_basename is None:
        output_basename = format_output_basename(
            variant_info.arch,
            variant_info.compiler,
            variant_info.compiler_version,
            variant_info.opt,
            variant_info.pkg,
            variant_info.variant_id,
        )
    csv_filename = f"{output_basename}{_OUTPUT_CSV_SUFFIX}"
    csv_final_path = out_folder / csv_filename

    if csv_final_path.exists() and skip_existing_csv:
        # Validate the existing CSV is complete (last line is the
        # vocab def). Pre-atomicity attempts may have written partial
        # CSVs to the final path; without this check, a partial CSV
        # would be treated as done and propagate through the unifier
        # as None (see vocab_unifier.unifier: "Probably incomplete
        # from crashed stage-1").
        if load_vocab_manager(csv_final_path) is not None:
            logger.info(f"File {f'{binary_path.name}_output.csv'} already exists: {csv_final_path}.")
            return (-1, -1)
        else:
            logger.warning(
                f"Existing CSV {csv_final_path} is incomplete (no vocab def "
                f"in last line); re-tokenizing."
            )

    with_pickled = False
    kvargs: dict | None = None
    start_time = time.time()
    do_pickles = DO_PICKLES and backend == "angr"

    if do_pickles:
        if has_valid_pickle(pickle_mainloop_file_path):
            logger.info("loading existing mainloop pickle to speed up")
            task.set_phase(TokenizerPhase.ANGR_2.value)
            kvargs = try_load_pickle(pickle_mainloop_file_path, logger)
            if kvargs is not None:
                if "path" not in kvargs:
                    kvargs["path"] = file_path
                with_pickled = True
                logger.info(f"Pickle loading time: {time.time() - start_time:.2f} seconds")
        elif has_valid_pickle(pickle_file_path):
            logger.info("loading existing pickle to speed up")
            kvargs = try_load_pickle(pickle_file_path, logger)
            if kvargs is not None:
                logger.info(f"Pickle loading time: {time.time() - start_time:.2f} seconds")

    # Stage every output file (CSV + sidecar `_consts.txt`) under
    # `/app/out-tmp/<rel>/` and atomic-publish to `out_folder` only
    # after a fully successful tokenization. If the worker is killed
    # mid-write (SLURM time-out, OOM, container SIGKILL), the stage
    # dir is rm-rf'd by the wrapper trap and `out_folder` never sees
    # a partial file — the next retry's skip-existing check trusts
    # whatever is at the canonical path because nothing partial can
    # land there.
    #
    # The provider try/finally lives INSIDE the staged_publish `with`
    # so per-binary backend resources (Ghidra JVM Project + analysis
    # threads) are released regardless of success or failure, before
    # the staging-dir cleanup runs.
    provider: DisassemblyProvider | None = None
    with staged_publish(task, out_folder, scope=str(relative_path)) as stage_dir:
        csv_path = stage_dir / csv_filename
        try:
            if kvargs is None:
                provider = get_disassembly_provider(backend, file_path)
                # `_consts.txt` is derived from `csv_path.parent /
                # f"{stem.replace('_output', '')}_consts.txt"` in both
                # providers — by passing the staged csv path here, the
                # consts.txt lands inside `stage_dir` and gets published
                # alongside the CSV on clean exit.
                constants: dict[str, list[str]] = provider.parse_data_sections(output_csv_path=str(csv_path))
                try:
                    provider.build_cfg()
                except (IndexError, AssertionError, NotImplementedError) as e:
                    # Known angr CFG-resolver bugs (e.g. arm_elf_fast.py:89
                    # IndexError on certain ARM ELF indirect jumps; the
                    # `NotImplementedError: Ummmmm... not sure what goes here.
                    # report bug to @rhelmot` from missing-instruction VEX
                    # paths) are binary-deterministic — retry won't help.
                    # Re-raise as a NonRecoverable so the framework's retry-pass
                    # doesn't burn cycles on the same task.
                    raise NonRecoverableTokenizerError(
                        f"angr build_cfg failed deterministically for {binary_name}: "
                        f"{type(e).__name__}: {e}"
                    ) from e

                kvargs = dict(provider=provider, constant_list=constants)
                logger.info(f"Preparation stage 1 time: {time.time() - start_time:.2f} seconds")
                start_time = time.time()
                if do_pickles:
                    save_pickle(pickle_file_path, kvargs)
                    logger.info(f"Pickle (prep only) saving time: {time.time() - start_time:.2f} seconds")
            else:
                # Pickle-loaded kvargs: surface the provider for cleanup
                # symmetry. angr's pickled provider holds only Python state
                # so close() is a no-op, but routing it through the same
                # path keeps the lifecycle invariant clean.
                provider = kvargs.get("provider")

            start_time = time.time()
            logger.info("Calling lowlevel_disas")
            kvargs.update(
                dict(
                    with_pickled=with_pickled,
                    do_pickles=do_pickles,
                    out_folder=out_folder,
                    binary_name=binary_name,
                    platform=platform,
                    csv_path=csv_path,
                    binary_path=binary_path,
                    pickle_mainloop_file_path=pickle_mainloop_file_path,
                    task=task,
                    logger=logger,
                    backend=backend,
                )
            )
            (func_names, function_manager, vocab_manager, filtered) = disassemble_to_tokens(**kvargs)
            disassembly_time = time.time() - start_time
            warning_handler.unregister()
            logger.info(
                f"Disassembly time: {disassembly_time:.2f} seconds, \
                warnings={warning_handler.warning_count}, \
                errors={warning_handler.error_count}, \
                filtered={filtered}"
            )

            # Persist the per-variant metadata next to the CSV so the
            # build_memmap phase can reconstruct VariantInfo for
            # `_versions.json`. Lives inside `staged_publish` so the
            # rglob-walk picks it up and atomic-publishes it together
            # with the CSV + `_consts.txt`. Written after a successful
            # `disassemble_to_tokens` only — partial runs have no
            # metadata sidecar, matching the staging contract that
            # nothing reaches the durable destination on failure.
            _write_meta_sidecar(stage_dir, output_basename, variant_info)
        finally:
            if provider is not None:
                provider.close()

    return (warning_handler.warning_count, filtered)
