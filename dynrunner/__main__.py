"""CLI dispatcher for dynrunner specializations.

    python -m dynrunner --task tokenize   [...task args]
    python -m dynrunner --task unify-vocab [...task args]
    python -m dynrunner --task build-memmap [...task args]
    python -m dynrunner --task all         [...shared args]

`--task all` runs the three tasks sequentially in the order
tokenize → unify-vocab → build-memmap, **chaining outputs** so phase
2 and phase 3 read from phase 1's output rather than the user's
original source tree:

* tokenize:     ``--source <user-source> --output <user-output>``
* unify-vocab:  ``--source <user-output> --output <user-output>``
                (mapping files land alongside the CSVs they describe)
* build-memmap: ``--source <user-output> --output <user-output>/memmap``
                (memmap files are flat-named per binary group; the
                ``memmap/`` subdir keeps them separate from the
                per-binary CSVs/mappings in the tokenize tree)

The user only specifies ``--source`` and ``--output`` once. Other
flags every subtask accepts (file-discovery filters, ``--platform``,
``--max-memory``, etc.) are forwarded verbatim to each phase.
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


def _extract_flag(rest: list[str], flag: str) -> str | None:
    """Read the value of ``--<flag>`` from ``rest`` without removing it.

    Accepts both ``--flag value`` and ``--flag=value`` forms. Returns
    the first occurrence (argparse semantics) or ``None`` if absent.
    """
    long = f"--{flag}"
    eq_prefix = f"--{flag}="
    for i, arg in enumerate(rest):
        if arg == long and i + 1 < len(rest):
            return rest[i + 1]
        if arg.startswith(eq_prefix):
            return arg[len(eq_prefix):]
    return None


def _replace_flag(rest: list[str], flag: str, new_value: str) -> list[str]:
    """Return ``rest`` with the value of ``--<flag>`` replaced by
    ``new_value``. If the flag is absent, append ``--<flag> <new_value>``.
    Leaves other args (their order) untouched.
    """
    long = f"--{flag}"
    eq_prefix = f"--{flag}="
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(rest):
        arg = rest[i]
        if arg == long and i + 1 < len(rest):
            out.append(long)
            out.append(new_value)
            i += 2
            replaced = True
            continue
        if arg.startswith(eq_prefix):
            out.append(f"{eq_prefix}{new_value}")
            i += 1
            replaced = True
            continue
        out.append(arg)
        i += 1
    if not replaced:
        out.extend([long, new_value])
    return out


def _chain_for_phase(rest: list[str], phase: str) -> list[str]:
    """Rewrite ``--source`` / ``--output`` for phase 2 and phase 3 of
    ``--task all`` so each phase reads from the previous phase's output.

    Phase 1 (``tokenize``) keeps the user's ``--source`` and ``--output``
    verbatim. Phase 2 (``unify-vocab``) reads the tokenize outputs by
    pointing ``--source`` at the user's ``--output``; its own
    ``--output`` stays the same dir so mapping files land alongside the
    CSVs they describe. Phase 3 (``build-memmap``) reads from the
    same dir but writes its flat-named memmap artefacts into a sibling
    ``memmap/`` subdir, preserving the tokenize tree's structure.

    See module docstring for the full layout. Aborts cleanly if the
    user didn't provide ``--output`` (we can't synthesise a chain
    target without it).
    """
    if phase == "tokenize":
        return rest
    user_output = _extract_flag(rest, "output")
    if user_output is None:
        raise SystemExit(
            "dynrunner --task all: --output is required to chain "
            f"phase '{phase}' onto the previous phase's output. "
            "Pass --output <dir> alongside --source."
        )
    rest = _replace_flag(rest, "source", user_output)
    if phase == "build-memmap":
        rest = _replace_flag(rest, "output", f"{user_output}/memmap")
    return rest


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
            _dispatch(sub, _chain_for_phase(rest, sub))
    else:
        _dispatch(args.task, rest)


if __name__ == "__main__":
    main()
