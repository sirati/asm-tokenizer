import logging
import time
from pathlib import Path
from typing import cast

from dynamic_runner.comm import CommunicationInterface, DoneResponse, KeepaliveResponse, PhaseUpdateResponse
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
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenResolver

DO_PICKLES: bool = True


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
    comm: CommunicationInterface,
    backend: str = "angr",
    with_pickled=False,
    do_pickles=True,
    **kwargs,
):
    if not with_pickled:
        comm.send_response(PhaseUpdateResponse(phase_name=TokenizerPhase.ANGR_2.value))
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

    function_manager, filtered = main_loop(vocab_manager=vocab_manager, comm=comm, **kwargs)

    return (func_names, function_manager, vocab_manager, filtered)


def run_tokenizer(
    binary_path: Path,
    platform: Platform | str,
    skip_existing_csv: bool,
    source_dir: Path,
    output_dir: Path,
    comm: CommunicationInterface,
    backend: str = "angr",
):
    logger, warning_handler = setup_logger("tokenizer")
    logger.info("STARTING DISASSEMBLY")
    comm.send_response(PhaseUpdateResponse(phase_name=TokenizerPhase.ANGR_1.value))

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
    else:
        assert binary_name.startswith(platform), (
            f"Binary name '{binary_name}' must start with platform '{platform}'. Wrong platform's file in queue?"
        )

    try:
        relative_path = file_path.relative_to(source_dir.absolute())
    except ValueError:
        relative_path = Path(file_path.parent.name) / file_path.name

    out_folder = output_dir / relative_path.parent
    out_folder.mkdir(parents=True, exist_ok=True)

    pickle_file_path = out_folder / f"{binary_name}.pkl"
    pickle_mainloop_file_path = out_folder / f"{binary_name}.mainloop.pkl"
    csv_path = out_folder / f"{binary_name}_output.csv"

    if csv_path.exists() and skip_existing_csv:
        logger.info(f"File {f'{binary_path.name}_output.csv'} already exists: {csv_path}.")
        comm.send_response(DoneResponse(warnings=-1, filtered=-1))
        return

    with_pickled = False
    kvargs: dict | None = None
    start_time = time.time()
    do_pickles = DO_PICKLES and backend == "angr"

    if do_pickles:
        if has_valid_pickle(pickle_mainloop_file_path):
            logger.info("loading existing mainloop pickle to speed up")
            comm.send_response(PhaseUpdateResponse(phase_name=TokenizerPhase.ANGR_2.value))
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

    if kvargs is None:
        provider = get_disassembly_provider(backend, file_path)
        constants: dict[str, list[str]] = provider.parse_data_sections(output_csv_path=str(csv_path))
        provider.build_cfg()

        kvargs = dict(provider=provider, constant_list=constants)
        logger.info(f"Preparation stage 1 time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        if do_pickles:
            save_pickle(pickle_file_path, kvargs)
            logger.info(f"Pickle (prep only) saving time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    logger.info("Calling lowlevel_disas")
    if kvargs is not None:
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
                comm=comm,
                logger=logger,
                backend=backend,
            )
        )
    else:
        raise RuntimeError("Failed to initialize kvargs")
    (func_names, function_manager, vocab_manager, filtered) = disassemble_to_tokens(**kvargs)
    disassembly_time = time.time() - start_time
    warning_handler.unregister()
    logger.info(
        f"Disassembly time: {disassembly_time:.2f} seconds, \
        warnings={warning_handler.warning_count}, \
        errors={warning_handler.error_count}, \
        filtered={filtered}"
    )

    comm.send_response(DoneResponse(warnings=warning_handler.warning_count, filtered=filtered))
