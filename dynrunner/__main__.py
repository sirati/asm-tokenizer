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

from tokenizer.arch import Platform as _CanonicalPlatform
from tokenizer.arch_translation import all_known_arch_strings


_TASK_TO_MODULE: dict[str, str] = {
    "tokenize": "dynrunner.tokenize",
    "unify-vocab": "dynrunner.unify_vocab",
    "build-memmap": "dynrunner.build_memmap",
}

_PIPELINE_ORDER: tuple[str, ...] = ("tokenize", "unify-vocab", "build-memmap")


# Framework's `--platform` default is `["x86", "x64"]` (see
# `dynamic_runner._shared.selection_args.add_selection_arguments`).
# That silently drops arm/mips/ppc/riscv CSVs at discovery in phases
# 2 and 3, leaving the user with a half-dataset and no warning. We
# inject the full set here when the user didn't pass `--platform`, so
# the asm-tokenizer dispatcher's default is "process everything" and
# users opt INTO a subset rather than out.
#
# The full set is the union of the canonical Platform literals
# (matching legacy filenames like `x64-clang-7-O0_minigzip`) and the
# sidecar arch aliases the translator knows about (`x86_64`,
# `armv7l-hf`, ...). The sidecar canonical-format outputs preserve
# the sidecar's verbose arch in the filename (e.g.
# `x86_64-clang-10.0.1-Oz_hello__cf70c518_output.csv`), so the
# discovery walks in phases 2 and 3 must accept BOTH spellings.
_ALL_PLATFORMS: tuple[str, ...] = tuple(
    sorted({*_CanonicalPlatform.__args__, *all_known_arch_strings()})
)


def _ensure_full_platform_default(rest: list[str]) -> list[str]:
    """If --platform isn't in `rest`, append the full ISA list.

    Detecting "user passed --platform" purely from `rest` is robust:
    argparse hasn't run yet, so any presence of the flag (with or
    without `=`-form) means the user is making an explicit choice and
    we leave it alone.
    """
    has_platform = any(
        arg == "--platform" or arg.startswith("--platform=")
        for arg in rest
    )
    if has_platform:
        return rest
    return [*rest, "--platform", *_ALL_PLATFORMS]


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

    rest = _ensure_full_platform_default(rest)

    if args.task == "all":
        for sub in _PIPELINE_ORDER:
            _dispatch(sub, rest)
    else:
        _dispatch(args.task, rest)


if __name__ == "__main__":
    main()
