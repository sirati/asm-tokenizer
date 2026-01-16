import pickle
import time
from pathlib import Path
from typing import Literal, cast

import angr

from tokenizer.address_meta_data_lookup import AddressMetaDataLookup
from tokenizer.csv_files import parse_and_save_data_sections
from tokenizer.instruction_sets import InstructionSets
from tokenizer.main_loop import main_loop
from tokenizer.token_manager import VocabularyManager
from tokenizer.tokens import TokenResolver

SCRIPT_FOLDER: Path = Path(__file__).parent.resolve()


def disassemble_to_tokens(
    out_folder: Path,
    binary_name: str,
    platform: Literal["x86", "arm64", "arm32", "x64"],
    cfg: angr.analyses.cfg.cfg_fast.CFGFast,
    constant_list: dict[str, list[str]],
    csv_path: Path,
    binary_path: Path,
    pickle_mainloop_file_path: Path,
    with_pickled=False,
    project=None,
    **kwargs,
):
    if not with_pickled:
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

        with open(pickle_mainloop_file_path, "wb") as f:
            pickle.dump(kwargs, f)

    else:
        kwargs.update(dict(cfg=cfg, constant_list=constant_list))
        func_names = kwargs["func_names"]

    vocab_manager = VocabularyManager(platform)

    resolver = TokenResolver()
    instr_sets = InstructionSets(SCRIPT_FOLDER / "./data_store.json")
    kwargs.update(dict(resolver=resolver, instr_sets=instr_sets, csv_path=csv_path))

    function_manager = main_loop(vocab_manager=vocab_manager, **kwargs)

    return (func_names, function_manager, vocab_manager)


def run_tokenizer(
    binary_path: Path,
    platform: Literal["x86", "arm64", "arm32", "x64", "file_prefix"],
    skip_existing_csv: bool,
    source_dir: Path,
    output_dir: Path,
) -> None:
    print("STARTING DISASSEMBLY")

    file_path: Path = binary_path.absolute()

    binary_name = file_path.name

    if platform == "file_prefix":
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
        print(f"[*] Detected platform: {platform}")
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
        print(f"File {f'{binary_path.name}_output.csv'} already exists: {csv_path}.")
        return None

    with_pickled = False
    start_time = time.time()

    if pickle_mainloop_file_path.exists():
        print("loading existing mainloop pickle to speed up")
        with open(pickle_mainloop_file_path, "rb") as f:
            kvargs = pickle.load(f)
            if "path" not in kvargs:
                kvargs["path"] = file_path
            with_pickled = True
        print(f"Pickle loading time: {time.time() - start_time:.2f} seconds")
    elif pickle_file_path.exists():
        print("loading existing pickle to speed up")
        with open(pickle_file_path, "rb") as f:
            kvargs = pickle.load(f)

        print(f"Pickle loading time: {time.time() - start_time:.2f} seconds")
    else:
        project: angr.Project = angr.Project(file_path, auto_load_libs=False)
        constants: dict[str, list[str]] = parse_and_save_data_sections(project)
        cfg: angr.analyses.cfg.cfg_fast.CFGFast = project.analyses.CFGFast(normalize=True)

        kvargs: dict = dict(project=project, cfg=cfg, constant_list=constants)
        print(f"Preparation stage 1 time: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        with open(pickle_file_path, "wb") as f:
            pickle.dump(kvargs, f)

        print(f"Pickle (prep only) saving time: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print("Calling lowlevel_disas")
    kvargs.update(
        dict(
            with_pickled=with_pickled,
            out_folder=out_folder,
            binary_name=binary_name,
            platform=platform,
            csv_path=csv_path,
            binary_path=binary_path,
            pickle_mainloop_file_path=pickle_mainloop_file_path,
        )
    )
    (func_names, function_manager, vocab_manager) = disassemble_to_tokens(**kvargs)
    disassembly_time = time.time() - start_time
    print(f"Disassembly time: {disassembly_time:.2f} seconds")
