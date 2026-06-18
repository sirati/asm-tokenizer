import argparse
import json
import logging
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dynamic_runner.worker import (
    NonRecoverableError,
    RecoverableError,
    Task,
    WorkerOutput,
    run,
    task_function,
)

from shared import (
    increase_csv_field_size_limit,
    remove_stream_handlers,
    resilient_file_handler,
)
from tokenizer.arch import Platform
from tokenizer.disasm import configure_worker_jvm_processor_cap
from tokenizer.arch_translation import arch_to_platform
from tokenizer.output_filename import format_output_basename
from tokenizer.run_tokenizer import NonRecoverableTokenizerError, run_tokenizer
from tokenizer.variant_info import VariantInfo

# Payload-key constants mirror the task-side encoder
# (``dynrunner.tokenize.tokenizer_task._build_payload``); kept verbatim
# so a rename on either side surfaces as a missing-key fail rather
# than a silent default. The variant sub-dict exactly matches
# ``VariantInfo``'s field names — see ``_decode_variant``.
_PAYLOAD_VARIANT_KEY = "variant"
# Per-binary identity slot (``handle.binary_name``); the output basename
# uses it instead of ``variant.pkg`` so a sidecar folder's several
# binaries (sharing one ``variant``) emit distinct, non-colliding files.
_PAYLOAD_BINARY_NAME_KEY = "binary_name"

def _decode_variant(variant_dict: dict) -> VariantInfo:
    """Reconstruct ``VariantInfo`` from the JSON-decoded payload.

    The payload encoder (``tokenizer_task._variant_to_payload_dict``)
    serialises every dataclass field verbatim, so the decoder can
    round-trip via the constructor — no field-by-field defensive
    rebuild needed. ``extra_metadata`` is forwarded as-is (a JSON dict
    decoded by the runtime), preserving the opaque pass-through
    contract for downstream consumers.
    """
    return VariantInfo(
        arch=variant_dict["arch"],
        compiler=variant_dict["compiler"],
        compiler_version=variant_dict["compiler_version"],
        opt=variant_dict["opt"],
        pkg=variant_dict["pkg"],
        variant_id=int(variant_dict.get("variant_id", 0)),
        extra_metadata=variant_dict.get("extra_metadata", {}) or {},
    )


@dataclass(frozen=True)
class _TokenizeJob:
    """One binary to tokenize: its on-disk path and the canonical
    output basename it should emit under.

    ``output_basename`` is precomputed at job-resolution time (legacy:
    derived from the variant's pkg slot; sidecar: per-archive-member
    basename in the binary slot) so the worker handler doesn't need
    to know which pathway produced the job — both flows converge on
    the same shape and ``run_tokenizer`` sees an explicit basename
    for each emit.
    """

    binary_path: Path
    output_basename: str


def _job_for_task(
    source_path: Path, variant: VariantInfo, binary_name: str
) -> _TokenizeJob:
    """Build the single ``_TokenizeJob`` the worker runs for one task.

    The on-disk file at ``source_path`` is always a binary (no
    extraction step) — discovery (``walk_dataset``) emits the binary
    path directly for both legacy 4-axis files and sidecar-folder
    variants. The basename is composed from the variant's canonical-4
    + per-binary ``binary_name`` + variant_id.

    ``binary_name`` is the per-binary identity slot the discovery side
    threads through the payload (``handle.binary_name``), NOT
    ``variant.pkg``: a sidecar folder's several binaries share one
    ``variant`` so ``pkg`` would collide them onto one basename. For
    legacy / single-binary corpora ``binary_name == pkg``, so this
    round-trips the source filename byte-for-byte (preserving legacy
    output paths).
    """
    return _TokenizeJob(
        binary_path=source_path,
        output_basename=format_output_basename(
            variant.arch,
            variant.compiler,
            variant.compiler_version,
            variant.opt,
            binary_name,
            variant.variant_id,
        ),
    )


@dataclass(frozen=True)
class DebugDefaults:
    arg_abbr: str  # e.g. "s"
    arg_name: str  # e.g. "small"
    binary: str
    folder: str
    compiler: str
    version: str
    optimisation: str

    @property
    def choices(self) -> tuple[str, str]:
        return self.arg_abbr, self.arg_name

    def path(self) -> str:
        return f"{{source}}/{self.folder}/x86-{self.compiler}-{self.version}-{self.optimisation}_{self.binary}"

    def help_line(self) -> str:
        return f"default for {self.arg_name}: {self.binary} -> {self.path()}"


debug_defaults = [
    DebugDefaults(
        arg_abbr="s",
        arg_name="small",
        binary="minigzipsh",
        folder="zlib",
        compiler="gcc",
        version="5",
        optimisation="O3",
    ),
    DebugDefaults(
        arg_abbr="m",
        arg_name="medium",
        binary="sigtool",
        folder="clamav",
        compiler="clang",
        version="5.0",
        optimisation="O1",
    ),
    DebugDefaults(
        arg_abbr="l",
        arg_name="large",
        binary="sigtool",
        folder="nmap",
        compiler="clang",
        version="5.0",
        optimisation="Os",
    ),
    DebugDefaults(
        arg_abbr="g",
        arg_name="giantic",
        binary="z3",
        folder="z3",
        compiler="gcc",
        version="5",
        optimisation="O0",
    ),
]
debug_defaults_loopup = {d.arg_abbr: d for d in debug_defaults} | {d.arg_name: d for d in debug_defaults}


# Module-level config populated by `_on_args` before the run-loop
# starts; the @task_function handler reads it for each task. The
# runtime's contract: on_args owns CLI-arg → config translation,
# the handler reads from there (the Task object carries only
# per-task data — relative_path + payload).
_PLATFORM: str
_SKIP_EXISTING: bool
_SOURCE_DIR: Path
_OUTPUT_DIR: Path
_BACKEND: str
_SIMULATE_ERRORS: float
_DUMP_DUPLICATE_FUNCTION_METADATA: bool
_DEBUG_RENDER: bool


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenize binaries for BinAI.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--batch",
        type=str,
        metavar="QUEUE_FILE",
        help="Process a batch of binaries from a queue file",
    )
    group.add_argument(
        "--single",
        type=str,
        metavar="BINARY_FILE",
        help="Process a single binary file",
    )
    group.add_argument(
        "--dynamic_queue",
        type=int,
        metavar="SOCKET_FD",
        help="Worker mode: receive tasks via socket file descriptor (anonymous socket)",
    )
    group.add_argument(
        "--socket-path",
        type=str,
        metavar="SOCKET_PATH",
        help="Worker mode: receive tasks via named Unix socket at this path",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file instead of stdout/err",
    )
    group.add_argument(
        "--debug",
        choices=([d.arg_abbr for d in debug_defaults] + [d.arg_name for d in debug_defaults]),
        help=(
            "Debug mode: process a debug file. Possible to override platform, "
            "compiler, version, and optimisation-level.\n" + "\n".join(d.help_line() for d in debug_defaults)
        ),
    )

    parser.add_argument(
        "--platform",
        type=str,
        help="Specify the platform (e.g., x86, arm64) for the tokenizer. Use 'auto' to auto-detect from binary name. Default is auto.",
        default="auto",
        choices=["x86", "x64", "arm32", "arm64", "mips32", "mips64", "ppc32", "ppc64", "riscv32", "riscv64", "auto"],
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing csv files.")
    parser.add_argument(
        "--source",
        type=str,
        default="./src",
        help="Source directory containing binaries (default: ./src)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./out",
        help="Output directory for results (default: ./out)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="ghidra",
        choices=["angr", "ghidra"],
        help="Disassembly backend to use (default: ghidra)",
    )
    parser.add_argument(
        "--workers-per-node",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of tokenize workers sharing this node (the dispatch's "
            "per-machine worker count). The disassembly JVM caps its "
            "thread pools to ceil(machine_cores / N) so N co-located "
            "workers stop oversubscribing the node's CPU. Framework "
            "dispatch supplies it from --cores; omit (default) for no cap."
        ),
    )
    parser.add_argument(
        "--simulate-errors",
        type=float,
        metavar="PERCENTAGE",
        help="Simulate random worker crashes with given percentage chance (0-100)",
    )
    parser.add_argument(
        "--dump-duplicate-function-metadata",
        action="store_true",
        help=(
            "Debug: write a 5-layer-deep pickle snapshot of every Ghidra Function "
            "whose name collides with another function in the same disassembly. "
            "Lands at <output_dir>/.../<base>_duplicate_function_dump.pkl. "
            "Ghidra backend only (angr ignores). Off by default; zero overhead "
            "when off."
        ),
    )
    return parser


def _setup_logging(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.log_file:
        if args.dynamic_queue:
            remove_stream_handlers(logger)
        # A vanished/stale log mount must never abort the worker before it
        # signals Ready; resilient_file_handler degrades to stderr on OSError.
        logger.addHandler(resilient_file_handler(Path(args.log_file), level=logging.INFO))
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    cwd = Path.cwd()
    return (cwd / args.source).resolve(), (cwd / args.output).resolve()


def _on_args(args: argparse.Namespace) -> None:
    """Hook invoked by `run()` before the loop starts. Sets up
    logging, CSV field-size limits, and the module-level config
    the handler closes over.
    """
    global _PLATFORM, _SKIP_EXISTING, _SOURCE_DIR, _OUTPUT_DIR, _BACKEND, _SIMULATE_ERRORS
    global _DUMP_DUPLICATE_FUNCTION_METADATA, _DEBUG_RENDER

    increase_csv_field_size_limit()
    _setup_logging(args)
    _SOURCE_DIR, _OUTPUT_DIR = _resolve_dirs(args)
    _PLATFORM = args.platform
    _SKIP_EXISTING = bool(args.skip_existing)
    _BACKEND = args.backend
    _SIMULATE_ERRORS = float(args.simulate_errors) if args.simulate_errors is not None else 0.0
    _DUMP_DUPLICATE_FUNCTION_METADATA = bool(args.dump_duplicate_function_metadata)
    # ``--debug`` is a mode in the mutually-exclusive entry group, so
    # worker / batch / single runs always leave it None → production
    # keeps the per-instruction debug-label rendering OFF.
    _DEBUG_RENDER = args.debug is not None

    # Size the disassembly JVM's thread pools to this worker's fair share
    # of the node BEFORE the first task constructs a provider (the JVM is
    # process-global per worker and boots once). The framework passes the
    # per-node worker count via ``--workers-per-node``; absent (standalone
    # runs / a dispatch that omits it) leaves the JVM uncapped with a WARN.
    configure_worker_jvm_processor_cap(args.workers_per_node)

    logger = logging.getLogger()
    logger.info(f"[*] Source directory: {_SOURCE_DIR}")
    logger.info(f"[*] Output directory: {_OUTPUT_DIR}")
    if _SIMULATE_ERRORS > 0:
        logger.info(f"[*] Error simulation enabled: {_SIMULATE_ERRORS}% chance per task")


@task_function
def handle(task: Task) -> WorkerOutput | None:
    """Per-task body — runtime owns the read/run/respond cycle.

    The runtime's exception → wire mapping turns:
      * MemoryError                       → OUT_OF_MEMORY (exit)
      * RecoverableError                  → RECOVERABLE
      * NonRecoverableError               → NON_RECOVERABLE
      * NonRecoverableTokenizerError      → routed to NonRecoverableError
        below because angr CFG-resolver bugs (arm_elf_fast.py:89
        IndexError, missing-VEX NotImplementedError) are
        binary-deterministic — retry won't help.
      * subprocess.CalledProcessError     → RECOVERABLE (default)
      * everything else                   → WorkerExceptionResponse
                                            (default: RECOVERABLE).
    """
    logger = logging.getLogger()
    # ``task.relative_path`` is the wire identifier emitted by the
    # primary's ``discover_items`` — strictly relative to the source
    # root (see tokenizer_task._iter_local_pairs / _to_taskinfo where
    # absolute on-disk paths get stripped to <rel>). When the
    # framework's extraction cache stages the file at a different
    # local location, ``task.resolved_path`` is set to that absolute
    # path and ``open_path`` returns it; otherwise ``open_path`` is
    # the relative wire identifier and we join it against
    # ``_SOURCE_DIR``.
    raw_open = Path(task.open_path) if hasattr(task, "open_path") else Path(task.relative_path)
    if raw_open.is_absolute():
        source_path = raw_open
    else:
        source_path = _SOURCE_DIR / raw_open
    logger.info(f"[*] Processing: {source_path}")

    if _SIMULATE_ERRORS > 0 and random.random() * 100 < _SIMULATE_ERRORS:
        logger.warning(f"[!] SIMULATED Error for task {source_path.name}")
        raise NonRecoverableError(f"Simulated error ({_SIMULATE_ERRORS}% chance)")

    # Decode the task payload (encoded by tokenizer_task._build_payload)
    # into a VariantInfo. The wire identifier (``source_path``) always
    # points at the binary file itself for both transport flavors
    # ``walk_dataset`` emits (legacy 4-axis files and sidecar-folder
    # ``<variant_dir>/<pkg>`` binaries) — no extraction step exists.
    payload = json.loads(task.payload_str) if task.payload_str else {}
    variant = _decode_variant(payload[_PAYLOAD_VARIANT_KEY])
    # Per-binary identity slot for the output basename. Falls back to
    # ``variant.pkg`` for any payload predating the key (legacy /
    # single-binary corpora, where ``binary_name == pkg`` holds).
    binary_name = payload.get(_PAYLOAD_BINARY_NAME_KEY) or variant.pkg

    # Source-tree-relative path used for output layout + staged_publish
    # scope. ``task.relative_path`` is wire-supplied relative to the
    # source root; we use it verbatim for both legacy and sidecar
    # modes so outputs mirror the source tree's layout.
    source_relative_path = Path(task.relative_path)

    # Sidecar tasks (``variant_id != 0``) carry a distro-style ``arch``
    # string (``x86_64``, ``aarch64``, ``armv7l-hf``, ...) that the
    # tokenizer's filename auto-detect can't read off the extracted
    # binary's bare name (``hello`` / ``busybox``). The translator owns
    # the mapping and raises on unknown arches; legacy tasks
    # (``variant_id == 0``) keep the auto-detect-from-filename pathway
    # because their on-disk filename still encodes the platform.
    if variant.variant_id != 0:
        platform: Platform | str = arch_to_platform(variant.arch)
    else:
        platform = cast(Platform | str, _PLATFORM)

    try:
        # ``_job_for_task`` builds the single ``_TokenizeJob`` the
        # worker runs. ``run_tokenizer`` returns warnings + filtered
        # counts; skip-existing emits return ``(-1, -1)`` which clamp
        # to 0 below — preserves the original semantic that ``0/0``
        # after a full skip is indistinguishable from a clean run.
        job = _job_for_task(source_path, variant, binary_name)
        warnings, filtered = run_tokenizer(
            job.binary_path,
            platform=platform,
            skip_existing_csv=_SKIP_EXISTING,
            source_dir=_SOURCE_DIR,
            output_dir=_OUTPUT_DIR,
            task=task,
            backend=_BACKEND,
            variant_info=variant,
            source_relative_path=source_relative_path,
            output_basename=job.output_basename,
            dump_duplicate_function_metadata=_DUMP_DUPLICATE_FUNCTION_METADATA,
            debug_render=_DEBUG_RENDER,
        )
        warnings_total = max(0, warnings)
        filtered_total = max(0, filtered)
    except NonRecoverableTokenizerError as e:
        # Binary-deterministic tokenizer failures (e.g.
        # angr arm_elf_fast.py:89 IndexError on certain
        # ARM ELF indirect jumps). Retry won't help.
        tb_str = traceback.format_exc()
        logger.error(f"[!] Non-recoverable tokenizer error processing {source_path}:\n{tb_str}")
        raise NonRecoverableError(str(e)) from e
    except (RecoverableError, NonRecoverableError, MemoryError):
        raise
    except Exception:
        # All other exceptions surface as WorkerExceptionResponse
        # (RECOVERABLE per c455835's default — the retry-pass
        # exhaustion logic catches truly-permanent cases). The
        # traceback ships with the response.
        tb_str = traceback.format_exc()
        logger.error(f"[!] Error processing {source_path}:\n{tb_str}")
        raise

    return WorkerOutput(warnings=warnings_total, filtered=filtered_total)


def _run_standalone(args: argparse.Namespace) -> None:
    """Standalone (non-framework) entry points: --single, --batch,
    --debug. These don't go through the runtime's read/run/respond
    loop; they iterate locally and pass a default ``Task()`` to
    ``run_tokenizer`` (its keepalive/set_phase methods are no-ops
    when ``_emit`` is None).
    """
    increase_csv_field_size_limit()
    _setup_logging(args)
    source_dir, output_dir = _resolve_dirs(args)
    logger = logging.getLogger()
    logger.info(f"[*] Source directory: {source_dir}")
    logger.info(f"[*] Output directory: {output_dir}")

    if args.debug is not None:
        args.platform = args.platform if args.platform != "auto" else "x86"
        debug_default = debug_defaults_loopup[args.debug]
        if debug_default is None:
            raise NotImplementedError(f"Debug option '{args.debug}' not found")
        if args.skip_existing:
            logger.warning("Skipping existing file in debug mode!! - probably not what you want")
        args.single = (
            f"{debug_default.folder}/x86-"
            f"{debug_default.compiler}-{debug_default.version}-{debug_default.optimisation}_{debug_default.binary}"
        )

    common_params = dict(
        platform=args.platform,
        skip_existing_csv=args.skip_existing,
        source_dir=source_dir,
        output_dir=output_dir,
        backend=args.backend,
        dump_duplicate_function_metadata=bool(args.dump_duplicate_function_metadata),
        # ``--debug`` runs render full per-instruction debug labels;
        # --single / --batch production runs skip the rendering.
        debug_render=args.debug is not None,
    )

    if args.batch:
        cwd = Path.cwd()
        queue_file_path = (cwd / args.batch).resolve()
        logger.info(f"[*] Reading queue file: {queue_file_path}")

        with open(queue_file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        logger.info(f"[*] Total lines in queue: {len(lines)}")

        absolute_lines = []
        for line in lines:
            if line.startswith("./"):
                line = line[2:]
            absolute_lines.append(str(source_dir / line))

        from tokenizer.utils import filter_queue

        filtered_lines = filter_queue(absolute_lines, out_dir=str(output_dir), source_dir=str(source_dir))
        logger.info(f"[*] Filtered queue: {len(filtered_lines)} items to process")

        failures: list[str] = []
        for idx, binary_path_str in enumerate(filtered_lines, 1):
            logger.info(f"\n[*] Processing binary {idx}/{len(filtered_lines)}: {binary_path_str}")
            binary_path = Path(binary_path_str).resolve()
            try:
                run_tokenizer(binary_path, task=Task(relative_path=str(binary_path)), **common_params)
            except Exception as e:
                logger.error(f"[!] Error processing {binary_path}: {e}")
                logger.info("Continuing with next binary in queue...")
                failures.append(binary_path_str)
                continue

        # A binary the operator put in the queue that did not produce its
        # output is a hard failure, not a skip: the explicit skip-existing
        # path returns cleanly from ``run_tokenizer`` (logged by
        # ``filter_queue``), so anything reaching ``failures`` is a real
        # no-output run. Exit non-zero so a standalone batch that silently
        # produced nothing can never be mistaken for success.
        logger.info(
            f"\n[*] Batch processing complete: "
            f"{len(filtered_lines) - len(failures)}/{len(filtered_lines)} succeeded."
        )
        if failures:
            logger.error(
                f"[!] {len(failures)} of {len(filtered_lines)} queued binaries "
                f"failed to produce output: {', '.join(failures)}"
            )
            sys.exit(1)
        return

    # --single (also reached after --debug's expansion above).
    binary_path_input = Path(args.single)
    cwd = Path.cwd()

    if binary_path_input.is_absolute():
        binary_path = binary_path_input.resolve()
    else:
        source_was_explicit = "--source" in sys.argv
        if source_was_explicit:
            binary_path = (source_dir / binary_path_input).resolve()
        else:
            cwd_path = cwd / binary_path_input
            exists_in_cwd = cwd_path.exists()
            source_path = source_dir / binary_path_input
            exists_in_source = source_path.exists()

            if exists_in_cwd and exists_in_source:
                logger.error(
                    f"[!] Ambiguous path: '{args.single}' exists in both current directory and source directory."
                )
                logger.error(f"    Found at: {cwd_path}")
                logger.error(f"    Found at: {source_path}")
                logger.error(
                    "    Suggestion: Use --source ./ or --source ./src/ to explicitly set the directory, or provide an absolute path."
                )
                sys.exit(1)
            elif exists_in_cwd:
                binary_path = cwd_path.resolve()
            elif exists_in_source:
                binary_path = source_path.resolve()
            else:
                binary_path = cwd_path.resolve()

    logger.info(f"[*] Processing single binary: {binary_path}")
    run_tokenizer(binary_path, task=Task(relative_path=str(binary_path)), **common_params)


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.dynamic_queue or args.socket_path:
        # Worker mode: hand control to the framework's runtime,
        # which owns the read/run/respond loop and exception →
        # wire mapping. Our `_on_args` populates module-level
        # config that the @task_function handler reads.
        run(args=args, on_args=_on_args)
        return

    _run_standalone(args)


if __name__ == "__main__":
    main()
