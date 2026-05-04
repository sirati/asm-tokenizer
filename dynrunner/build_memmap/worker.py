"""Worker subprocess for `dynrunner.build_memmap`.

Receives one ``ProcessBinaryCommand`` per group. The wire's
``relative_path`` is the binary_name (an opaque identifier; not a
filesystem path) and ``payload`` is a JSON string containing the
per-version pairing data the starting instance prepared. The worker:

1. Parses ``command.payload`` into a ``versions`` list of
   ``{csv_path, mapping_path, arch, compiler, compilerversion, opt}``.
2. Reconstructs ``BinaryVersionInfo`` instances.
3. Calls ``tokenizer.memmap_builder.builder.build_memmap_files`` once
   for the group and replies ``done``.

The worker scans nothing, reads no manifest from disk, and does no
pairing or grouping — those concerns live entirely in
``MemmapBuilderTask.discover_items`` on the starting instance.
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
from shared import increase_csv_field_size_limit, remove_stream_handlers
from tokenizer.memmap_builder.builder import BinaryVersionInfo, build_memmap_files


def _process_payload(
    binary_name: str, payload_json: str, output_dir: Path
) -> None:
    """Parse the inline payload, reconstruct BinaryVersionInfo, build memmap."""
    data = json.loads(payload_json)
    versions_raw = data["versions"]
    versions = [
        BinaryVersionInfo(
            path=Path(entry["csv_path"]),
            mapping_path=Path(entry["mapping_path"]),
            arch=entry["arch"],
            compiler=entry["compiler"],
            compilerversion=entry["compilerversion"],
            opt=entry["opt"],
        )
        for entry in versions_raw
    ]
    build_memmap_files(versions, output_dir, binary_name)


def main() -> None:
    sock = None
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        increase_csv_field_size_limit()

        parser = argparse.ArgumentParser(
            description="Memmap-builder worker (per-binary-group).",
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
                "Source directory. Informational only at this layer — "
                "the per-version csv/mapping paths come absolute inside "
                "the wire's task payload."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="Output directory for memmap files",
        )
        parser.add_argument(
            "--vocab-source",
            type=str,
            default=None,
            help=(
                "Vocab source directory (informational; the wire payload "
                "carries absolute mapping paths)."
            ),
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
                "Accepted for framework compatibility; per-group skip "
                "is governed by the starting instance's discover_items "
                "filter, not the worker."
            ),
        )

        args = parser.parse_args()

        source_dir = Path(args.source).resolve()
        output_dir = Path(args.output).resolve()

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

                # `relative_path` is the binary_name; `payload` is the
                # per-version pairing JSON. Both come from the wire's
                # `task:` form (see ProcessBinaryCommand).
                binary_name = command.relative_path
                if not command.payload:
                    response = ErrorResponse(
                        error_type=ErrorType.NON_RECOVERABLE,
                        error_message=(
                            f"build_memmap worker received task without "
                            f"payload for binary_name={binary_name!r}; "
                            f"discover_items must emit non-empty "
                            f"TaskInfo.payload for this task type."
                        ),
                    )
                    comm.send_response(response)
                    continue
                logger.info(f"[*] Processing group: {binary_name}")

                try:
                    _process_payload(binary_name, command.payload, output_dir)
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
                except Exception as e:
                    tb_str = traceback.format_exc()
                    logger.error(f"[!] Error processing {binary_name}:\n{tb_str}")
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
