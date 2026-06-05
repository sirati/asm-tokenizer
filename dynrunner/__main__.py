"""CLI dispatcher for dynrunner specializations.

    python -m dynrunner --task tokenize       [...task args]
    python -m dynrunner --task unify-vocab    [...task args]
    python -m dynrunner --task build-memmap   [...task args]
    python -m dynrunner --task full-pipeline  [...task args]
    python -m dynrunner --task all            (alias for full-pipeline)

``--task full-pipeline`` hands the framework a composite
:class:`~dynrunner.full_pipeline.FullPipelineTask` that declares all
three phases with dep edges; the framework drives the chain on one
persistent secondary mesh (one sbatch wave on SLURM, one mesh
formation round). The historical ``--task all`` form — three
independent framework dispatches with per-phase rewrites of
``--source``/``--output`` — has been removed.

``--task all`` is kept as a verbatim alias for ``full-pipeline`` so
ops scripts that already pin the older name keep working without a
behaviour change.
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
    "full-pipeline": "dynrunner.full_pipeline",
}

# `--task all` is a verbatim alias kept for ops-compat with the
# pre-composite shell scripts. The alias is resolved upfront so the
# dispatch path is the same one-shot ``_dispatch`` invocation as every
# other task.
_TASK_ALIASES: dict[str, str] = {
    "all": "full-pipeline",
}


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
    # Thread the remaining argv to the task entry point explicitly; the
    # framework's run() no longer reads process-global sys.argv. (The
    # task module is also the secondary entry point — there main(argv=None)
    # falls back to sys.argv[1:] for the framework-forwarded argv.)
    module.main(argv=rest)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m dynrunner",
        description="Dispatcher for asm-tokenizer's dynrunner specializations.",
        add_help=False,
    )
    parser.add_argument(
        "--task",
        choices=[*_TASK_TO_MODULE.keys(), *_TASK_ALIASES.keys()],
        help=(
            "Which specialization to run. `full-pipeline` (alias: `all`) "
            "runs the composite tokenize → unify-vocab → build-memmap "
            "task as a single framework dispatch."
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

    resolved_task = _TASK_ALIASES.get(args.task, args.task)
    _dispatch(resolved_task, rest)


if __name__ == "__main__":
    main()
