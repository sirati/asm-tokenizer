"""Worker subprocess for `dynrunner.unify_vocab`.

Receives one ``ProcessBinaryCommand`` per run. The wire's
``relative_path`` is the literal "unify_vocab" sentinel and
``payload`` is a JSON string carrying the list of discovered
per-binary CSV relative paths. The worker:

1. Parses ``command.payload`` into a ``csv_paths`` list.
2. Joins each entry against ``--source`` (the bind-mount root inside
   the container, or the local source dir for non-SLURM dispatch).
3. Calls ``tokenizer.vocab_unifier.unifier.unify_vocab(csv_files,
   unified_vocab_path)`` and replies ``done``.

The standalone CLI at ``tokenizer.vocab_unifier.__main__`` does its
own discovery; this worker doesn't. Discovery is the starting
instance's concern under the dynrunner Protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import traceback
from pathlib import Path

from dynamic_runner.comm import (
    ErrorResponse,
    ErrorType,
    PickledErrorResponse,
    ReadyResponse,
    StopCommand,
    UnixSocketInterface,
)
from shared import remove_stream_handlers
from tokenizer.vocab_unifier.unifier import unify_vocab


def _process_payload(
    payload_json: str,
    source_dir: Path,
    output_dir: Path,
    unified_vocab_filename: str,
) -> None:
    data = json.loads(payload_json)
    csv_paths_raw = data["csv_paths"]
    csv_files = [source_dir / rel for rel in csv_paths_raw]
    unified_vocab_path = output_dir / unified_vocab_filename
    output_dir.mkdir(parents=True, exist_ok=True)
    unify_vocab(csv_files, unified_vocab_path)


def main() -> None:
    sock = None
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        parser = argparse.ArgumentParser(
            description="Vocab-unifier worker (single-process aggregation).",
        )

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--dynamic_queue",
            type=int,
            metavar="SOCKET_FD",
            help="Receive tasks via socket file descriptor (anonymous socket)",
        )
        group.add_argument(
            "--socket-path",
            type=str,
            metavar="SOCKET_PATH",
            help="Receive tasks via named Unix socket at this path",
        )

        parser.add_argument(
            "--source",
            type=str,
            required=True,
            help=(
                "Source directory. Per-CSV relative paths in the worker's "
                "task payload resolve against this; in SLURM bind-mount "
                "deployments this is `/app/src-network`."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="Output directory for the unified vocab CSV.",
        )
        parser.add_argument(
            "--out-unified-vocab",
            type=str,
            default="unified_vocab.csv",
            help="Filename for the unified vocab CSV (default: unified_vocab.csv).",
        )
        parser.add_argument(
            "--log-file",
            type=str,
            help="Log file instead of stdout/err",
        )
        parser.add_argument(
            "--skip_existing",
            action="store_true",
            help=(
                "Accepted for framework compatibility; the discover_items "
                "filter on the starting instance owns the skip-existing "
                "check, not the worker."
            ),
        )

        args = parser.parse_args()

        source_dir = Path(args.source).resolve()
        output_dir = Path(args.output).resolve()
        unified_vocab_filename = args.out_unified_vocab

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
                "%(levelname)s | %(asctime)s,%(msecs)03d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
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

        if args.socket_path:
            from dynamic_runner.comm import NamedSocketInterface

            logger.info(f"[*] Worker connecting to named socket: {args.socket_path}")
            comm = NamedSocketInterface(args.socket_path, is_server=False)
        else:
            sock = socket.socket(fileno=args.dynamic_queue)
            comm = UnixSocketInterface(sock)

        logger.info(f"[*] Worker started (PID {os.getpid()}), sending ready signal...")
        comm.send_response(ReadyResponse())
        logger.info("[*] Ready signal sent, waiting for tasks...")

        while True:
            try:
                command = comm.receive_command(blocking=True)
                if not command:
                    break

                if isinstance(command, StopCommand):
                    logger.info("[*] Received stop command, shutting down")
                    break

                if not command.payload:
                    response = ErrorResponse(
                        error_type=ErrorType.NON_RECOVERABLE,
                        error_message=(
                            "vocab_unifier worker received task without "
                            "payload; discover_items must emit non-empty "
                            "TaskInfo.payload for this task type."
                        ),
                    )
                    comm.send_response(response)
                    continue
                logger.info("[*] Processing unify_vocab task")

                try:
                    _process_payload(
                        command.payload,
                        source_dir,
                        output_dir,
                        unified_vocab_filename,
                    )
                    from dynamic_runner.comm import DoneResponse

                    comm.send_response(DoneResponse())
                except MemoryError as e:
                    response = ErrorResponse(
                        error_type=ErrorType.OUT_OF_MEMORY, error_message=str(e)
                    )
                    comm.send_response(response)
                    break
                except (KeyboardInterrupt, SystemExit) as e:
                    response = ErrorResponse(
                        error_type=ErrorType.NON_RECOVERABLE,
                        error_message=type(e).__name__,
                    )
                    comm.send_response(response)
                    logger.info(f"[!] Non-recoverable error: {e}")
                    break
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
                    # Filesystem-shape errors are permanent for this
                    # input set — re-running won't make a missing CSV
                    # appear. Mark NonRecoverable so the framework
                    # surfaces the misconfiguration instead of looping.
                    tb_str = traceback.format_exc()
                    logger.error(f"[!] Non-recoverable filesystem error:\n{tb_str}")
                    print(
                        f"[!] Non-recoverable filesystem error:\n{tb_str}",
                        file=sys.stderr,
                        flush=True,
                    )
                    response = ErrorResponse(
                        error_type=ErrorType.NON_RECOVERABLE,
                        error_message=f"{type(e).__name__}: {str(e)}",
                    )
                    comm.send_response(response)
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"[!] Error processing unify_vocab:\n{tb_str}")
                    # Also dump to stderr so the traceback survives
                    # ephemeral worker logs and reaches the SLURM stderr
                    # capture (slurm_<jobid>.err).
                    print(
                        f"[!] Error processing unify_vocab:\n{tb_str}",
                        file=sys.stderr,
                        flush=True,
                    )
                    response = ErrorResponse(
                        error_type=ErrorType.RECOVERABLE,
                        error_message=f"{type(e).__name__}: {str(e)}",
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
                        error_message=(
                            f"Failed to send error: {type(e).__name__}: {str(e)[:100]}"
                        ),
                    )
                    comm.send_response(fallback)
                break

        comm.close()
        logger.info("[*] Worker shutdown complete")

    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
