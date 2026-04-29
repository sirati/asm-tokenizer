"""CLI dispatcher for dynrunner specializations.

    python -m dynrunner --task tokenize   [...task args]
    python -m dynrunner --task unify-vocab [...task args]
    python -m dynrunner --task build-memmap [...task args]
    python -m dynrunner --task all         [...shared args]

`--task all` runs the three tasks sequentially in the order
tokenize → unify-vocab → build-memmap. Each subtask's argparse sees
the unchanged remaining argv, so any flag it doesn't recognize is a
hard error. Use `--task all` only with arguments every subtask accepts
(`--source`, `--output`, the file-discovery filters, etc.).
"""

from __future__ import annotations

import argparse
import importlib
import sys


_TASK_TO_MODULE: dict[str, str] = {
    "tokenize": "dynrunner.tokenize",
    "unify-vocab": "dynrunner.unify_vocab",
    "build-memmap": "dynrunner.build_memmap",
}

_PIPELINE_ORDER: tuple[str, ...] = ("tokenize", "unify-vocab", "build-memmap")


def _dispatch(task: str, rest: list[str]) -> None:
    module_name = _TASK_TO_MODULE[task]
    module = importlib.import_module(module_name + ".__main__")
    sys.argv = [f"python -m {module_name}", *rest]
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dynrunner",
        description="Dispatcher for asm-tokenizer's dynrunner specializations.",
        add_help=False,
    )
    parser.add_argument(
        "--task",
        choices=[*_TASK_TO_MODULE.keys(), "all"],
        help=(
            "Which specialization to run. `all` runs the full "
            "tokenize -> unify-vocab -> build-memmap pipeline."
        ),
    )
    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show dispatcher help; with --task, forwards --help to that task too.",
    )

    args, rest = parser.parse_known_args()

    if args.task is None:
        parser.print_help()
        sys.exit(0 if args.help else 2)

    if args.help:
        parser.print_help()
        rest = [*rest, "--help"]

    if args.task == "all":
        for sub in _PIPELINE_ORDER:
            _dispatch(sub, rest)
    else:
        _dispatch(args.task, rest)


if __name__ == "__main__":
    main()
