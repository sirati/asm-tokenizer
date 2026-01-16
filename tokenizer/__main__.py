import argparse
import csv
import logging
import socket
import sys
import traceback
from pathlib import Path
from typing import Literal, cast

from tokenizer.run_tokenizer import run_tokenizer


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    maxInt = sys.maxsize
    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    parser = argparse.ArgumentParser(description="Tokenize binaries for BinAI.")
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
        help="Worker mode: receive tasks via socket file descriptor",
    )
    group.add_argument(
        "--debugs",
        action="store_true",
        help="Debug mode: process ../src/clamav/x86-gcc-5-O3_minigzipsh",
    )
    group.add_argument(
        "--debugl",
        action="store_true",
        help="Debug mode: process ../src/clamav/x86-clang-5.0-O1_sigtool",
    )

    parser.add_argument(
        "--platform",
        type=str,
        help="Specify the platform (e.g., x86, arm64) for the tokenizer. Use 'file_prefix' to auto-detect from binary name. Default is file_prefix.",
        default="file_prefix",
        choices=["x86", "arm64", "arm32", "x64", "file_prefix"],
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

    args = parser.parse_args()

    cwd = Path.cwd()
    source_dir = (cwd / args.source).resolve()
    output_dir = (cwd / args.output).resolve()

    logger.info(f"[*] Source directory: {source_dir}")
    logger.info(f"[*] Output directory: {output_dir}")

    if (args.debugs or args.debugl) and args.platform == "file_prefix":
        args.platform = "x86"

    common_params = dict(
        platform=args.platform,
        skip_existing_csv=args.skip_existing,
        source_dir=source_dir,
        output_dir=output_dir,
    )

    if args.dynamic_queue:
        sock = socket.socket(fileno=args.dynamic_queue)
        sock_file = sock.makefile("r")

        logger.info("[*] Worker started, waiting for tasks...")

        while True:
            try:
                line = sock_file.readline()
                if not line:
                    break

                command = line.strip()

                if command == "stop":
                    logger.info("[*] Received stop command, shutting down")
                    break

                binary_path = source_dir / command
                logger.info(f"[*] Processing: {binary_path}")

                try:
                    run_tokenizer(
                        binary_path,
                        platform=cast(
                            Literal["x86", "arm64", "arm32", "x64", "file_prefix"], common_params["platform"]
                        ),
                        skip_existing_csv=cast(bool, common_params["skip_existing_csv"]),
                        source_dir=cast(Path, common_params["source_dir"]),
                        output_dir=cast(Path, common_params["output_dir"]),
                        sock=sock,
                    )
                except MemoryError as e:
                    error_msg = f"error:oom:{str(e)}\n"
                    sock.sendall(error_msg.encode("utf-8"))
                    break
                except (KeyboardInterrupt, SystemExit) as e:
                    error_msg = f"error:non_recoverable:{type(e).__name__}\n"
                    sock.sendall(error_msg.encode("utf-8"))
                    logger.info(f"[!] Non-recoverable error: {e}")
                    break
                except Exception as e:
                    tb_str = traceback.format_exc().replace("\n", " ")[:200]
                    error_msg = f"error:recoverable:{type(e).__name__}: {str(e)[:100]} | {tb_str}\n"
                    sock.sendall(error_msg.encode("utf-8"))

            except (KeyboardInterrupt, SystemExit) as e:
                logger.info(f"[!] Worker interrupted: {e}")
                break
            except Exception as e:
                logger.info(f"[!] Worker error: {e}")
                break

        sock.close()
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

        for idx, binary_path_str in enumerate(filtered_lines, 1):
            logger.info(f"\n[*] Processing binary {idx}/{len(filtered_lines)}: {binary_path_str}")
            binary_path = Path(binary_path_str).resolve()
            try:
                run_tokenizer(binary_path, **common_params)
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
        run_tokenizer(binary_path, **common_params)
    elif args.debugs:
        binary_path = source_dir / f"clamav/{args.platform}-gcc-5-O3_minigzipsh"
        logger.info(f"[*] Debug mode (gcc): {binary_path}")
        debug_params = common_params.copy()
        debug_params.update(dict(skip_existing_csv=False))
        run_tokenizer(binary_path, **debug_params)
    elif args.debugl:
        binary_path = source_dir / f"clamav/{args.platform}-clang-5.0-O1_sigtool"
        logger.info(f"[*] Debug mode (clang): {binary_path}")
        debug_params = common_params.copy()
        debug_params.update(dict(skip_existing_csv=False))
        run_tokenizer(binary_path, **debug_params)


if __name__ == "__main__":
    main()
