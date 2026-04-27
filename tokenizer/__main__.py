import argparse
import csv
import logging
import os
import random
import socket
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dynamic_batch_rs.comm import (
    ErrorResponse,
    ErrorType,
    NoopInterface,
    PickledErrorResponse,
    ReadyResponse,
    StopCommand,
    UnixSocketInterface,
    parse_command,
)
from shared import increase_csv_field_size_limit, remove_stream_handlers
from tokenizer.arch import Platform
from tokenizer.run_tokenizer import run_tokenizer


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


def main():
    sock = None
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        increase_csv_field_size_limit()

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
            choices=["x86", "x64", "arm32", "arm64", "mips", "mips64", "ppc", "ppc64", "riscv32", "riscv64", "auto"],
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
            default="angr",
            choices=["angr", "ghidra"],
            help="Disassembly backend to use (default: angr)",
        )
        parser.add_argument(
            "--simulate-errors",
            type=float,
            metavar="PERCENTAGE",
            help="Simulate random worker crashes with given percentage chance (0-100)",
        )

        args = parser.parse_args()

        cwd = Path.cwd()
        source_dir = (cwd / args.source).resolve()
        output_dir = (cwd / args.output).resolve()

        if args.log_file:
            if args.dynamic_queue:
                remove_stream_handlers(logger)

            log_file = Path(args.log_file)
            if not log_file.parent.exists():
                log_file.parent.mkdir(parents=True)

            log_file.touch()

            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

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
        )

        if args.dynamic_queue or args.socket_path:
            if args.socket_path:
                from dynamic_batch_rs.comm import NamedSocketInterface

                logger.info(f"[*] Worker connecting to named socket: {args.socket_path}")
                comm = NamedSocketInterface(args.socket_path, is_server=False)
                logger.info(f"[*] Worker started (PID {os.getpid()}), sending ready signal...")
            else:
                sock = socket.socket(fileno=args.dynamic_queue)
                comm = UnixSocketInterface(sock)
                logger.info(f"[*] Worker started (PID {os.getpid()}), sending ready signal...")

            comm.send_response(ReadyResponse())
            logger.info(f"[*] Ready signal sent, waiting for tasks...")

            # Check if crash simulation is enabled
            simulate_errors_chance = args.simulate_errors if args.simulate_errors is not None else 0.0
            if simulate_errors_chance > 0:
                logger.info(f"[*] Error simulation enabled: {simulate_errors_chance}% chance per task")

            while True:
                try:
                    command = comm.receive_command(blocking=True)
                    if not command:
                        break

                    if isinstance(command, StopCommand):
                        logger.info("[*] Received stop command, shutting down")
                        break

                    binary_path = source_dir / command.relative_path
                    logger.info(f"[*] Processing: {binary_path}")

                    # Simulate crash if enabled
                    if simulate_errors_chance > 0:
                        if random.random() * 100 < simulate_errors_chance:
                            logger.warning(f"[!] SIMULATED Error for task {binary_path.name}")
                            response = ErrorResponse(
                                error_type=ErrorType.NON_RECOVERABLE,
                                error_message=f"Simulated error ({simulate_errors_chance}% chance)",
                            )
                            comm.send_response(response)
                            break

                    try:
                        run_tokenizer(
                            binary_path,
                            platform=cast(Platform | str, common_params["platform"]),
                            skip_existing_csv=cast(bool, common_params["skip_existing_csv"]),
                            source_dir=cast(Path, common_params["source_dir"]),
                            output_dir=cast(Path, common_params["output_dir"]),
                            comm=comm,
                            backend=cast(str, common_params["backend"]),
                        )
                    except MemoryError as e:
                        response = ErrorResponse(error_type=ErrorType.OUT_OF_MEMORY, error_message=str(e))
                        comm.send_response(response)
                        break
                    except (KeyboardInterrupt, SystemExit) as e:
                        response = ErrorResponse(error_type=ErrorType.NON_RECOVERABLE, error_message=type(e).__name__)
                        comm.send_response(response)
                        logger.info(f"[!] Non-recoverable error: {e}")
                        break
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        logger.error(f"[!] Error processing {binary_path}:\n{tb_str}")
                        response = ErrorResponse(
                            error_type=ErrorType.RECOVERABLE, error_message=f"{type(e).__name__}: {str(e)}"
                        )
                        comm.send_response(response)

                except (KeyboardInterrupt, SystemExit) as e:
                    logger.info(f"[!] Worker interrupted: {e}")
                    break
                except Exception as e:
                    logger.info(f"[!] Worker error: {e}")
                    try:
                        response = PickledErrorResponse(
                            exception_type=type(e).__name__,
                            exception_message=str(e),
                            traceback_str=traceback.format_exc(),
                        )
                        comm.send_response(response)
                    except Exception:
                        fallback = ErrorResponse(
                            error_type=ErrorType.NON_RECOVERABLE,
                            error_message=f"Failed to send error: {type(e).__name__}: {str(e)[:100]}",
                        )
                        comm.send_response(fallback)
                    break

            comm.close()
            logger.info("[*] Worker shutdown complete")

        elif args.batch:
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

            comm = NoopInterface()
            for idx, binary_path_str in enumerate(filtered_lines, 1):
                logger.info(f"\n[*] Processing binary {idx}/{len(filtered_lines)}: {binary_path_str}")
                binary_path = Path(binary_path_str).resolve()
                try:
                    run_tokenizer(binary_path, comm=comm, **common_params)
                except Exception as e:
                    logger.info(f"[!] Error processing {binary_path}: {e}")
                    logger.info("Continuing with next binary in queue...")
                    continue

            logger.info("\n[*] Batch processing complete.")
        elif args.single:
            binary_path_input = Path(args.single)

            # If the path is absolute, use it directly
            if binary_path_input.is_absolute():
                binary_path = binary_path_input.resolve()
            else:
                # Check if --source was explicitly provided by the user
                source_was_explicit = "--source" in sys.argv

                if source_was_explicit:
                    # If source was explicitly set, always use file relative to source_dir
                    binary_path = (source_dir / binary_path_input).resolve()
                else:
                    # Check if path exists relative to cwd
                    cwd_path = cwd / binary_path_input
                    exists_in_cwd = cwd_path.exists()

                    # Check if path exists relative to source_dir
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
                        # Neither exists, just resolve relative to cwd (will fail later with clear error)
                        binary_path = cwd_path.resolve()
            logger.info(f"[*] Processing single binary: {binary_path}")
            comm = NoopInterface()
            run_tokenizer(binary_path, comm=comm, **common_params)

    except Exception as e:
        if sock is not None:
            try:
                tb_str = traceback.format_exc()
                error_info = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": tb_str,
                }
                pickled_error = pickle.dumps(error_info)
                error_msg = f"error:pickle:{pickled_error.decode('latin-1')}\n"
                sock.sendall(error_msg.encode("utf-8"))
            except Exception as pickle_error:
                try:
                    fallback_msg = f"error:non_recoverable:Failed to pickle error: {type(e).__name__}: {str(e)[:100]}\n"
                    sock.sendall(fallback_msg.encode("utf-8"))
                except Exception:
                    pass
        raise


if __name__ == "__main__":
    main()
