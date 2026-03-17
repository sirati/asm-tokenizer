import logging
import time
from pathlib import Path
from typing import Literal, cast

import angr

from dynamic_batch.comm import CommunicationInterface, DoneResponse, KeepaliveResponse, PhaseUpdateResponse
from dynamic_batch.task.tokenizer import TokenizerPhase
from shared import setup_logger
from tokenizer.address_meta_data_lookup import AddressMetaDataLookup
from tokenizer.arch import get_provider
from tokenizer.csv_files import parse_and_save_data_sections
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
    platform: Literal["x86", "arm64", "arm32", "x64"],
    cfg: angr.analyses.cfg.cfg_fast.CFGFast,
    constant_list: dict[str, list[str]],
    csv_path: Path,
    binary_path: Path,
    pickle_mainloop_file_path: Path,
    logger: logging.Logger,
    comm: CommunicationInterface,
    with_pickled=False,
    project=None,
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

        project = angr.Project(binary_path, auto_load_libs=False) if project is None else project
        obj = project.loader.main_object
        text_start: int = 0
        text_end: int = 0

        for section in obj.sections:
            if section.name == ".text":
                text_start = section.vaddr
                text_size = section.memsize
                text_end = text_start + text_size

        lookup = AddressMetaDataLookup(binary_path)

        func_disas_token: dict[str, list[dict[str, list[str]]]] = {}

        func_name_addr = {}

        kwargs = dict(
            block_runlength_dict=block_runlength_dict,
            cfg=cfg,
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

        if DO_PICKLES:
            save_pickle(pickle_mainloop_file_path, kwargs)

    else:
        kwargs.update(dict(cfg=cfg, constant_list=constant_list))
        func_names = kwargs["func_names"]

    vocab_manager = VocabularyManager(platform)

    resolver = TokenResolver()
    arch_provider = get_provider(platform)
    instr_sets = arch_provider.load_instruction_sets()

    kwargs.update(
        dict(resolver=resolver, instr_sets=instr_sets, arch_provider=arch_provider, csv_path=csv_path, logger=logger)
    )

    function_manager, filtered = main_loop(vocab_manager=vocab_manager, comm=comm, **kwargs)

    return (func_names, function_manager, vocab_manager, filtered)


def run_tokenizer(
    binary_path: Path,
    platform: Literal["x86", "arm64", "arm32", "x64", "auto"],
    skip_existing_csv: bool,
    source_dir: Path,
    output_dir: Path,
    comm: CommunicationInterface,
):
    logger, warning_handler = setup_logger("tokenizer")
    logger.info("STARTING DISASSEMBLY")
    comm.send_response(PhaseUpdateResponse(phase_name=TokenizerPhase.ANGR_1.value))

    file_path: Path = binary_path.absolute()

    binary_name = file_path.name

    if platform == "auto":
        possible_platforms: list[Literal["x86", "arm64", "arm32", "x64"]] = [
            "x86",
            "arm64",
            "arm32",
            "x64",
        ]
        detected_platform: Literal["x86", "arm64", "arm32", "x64"] | None = None
        for p in possible_platforms:
            if binary_name.startswith(p):
                detected_platform = p
                break

        if detected_platform is None:
            raise ValueError(
                f"Could not detect platform from binary name '{binary_name}'. "
                f"Expected one of: {', '.join(possible_platforms)}"
            )
        platform = cast(Literal["x86", "arm64", "arm32", "x64"], detected_platform)
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

    if DO_PICKLES:
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
        project: angr.Project = angr.Project(file_path, auto_load_libs=False)
        constants: dict[str, list[str]] = parse_and_save_data_sections(project, output_csv_path=str(csv_path))
        cfg: angr.analyses.cfg.cfg_fast.CFGFast = project.analyses.CFGFast(normalize=True)

        kvargs = dict(project=project, cfg=cfg, constant_list=constants)
        logger.info(f"Preparation stage 1 time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        if DO_PICKLES:
            save_pickle(pickle_file_path, kvargs)
            logger.info(f"Pickle (prep only) saving time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    logger.info("Calling lowlevel_disas")
    if kvargs is not None:
        kvargs.update(
            dict(
                with_pickled=with_pickled,
                out_folder=out_folder,
                binary_name=binary_name,
                platform=platform,
                csv_path=csv_path,
                binary_path=binary_path,
                pickle_mainloop_file_path=pickle_mainloop_file_path,
                comm=comm,
                logger=logger,
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
