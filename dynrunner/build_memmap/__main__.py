from __future__ import annotations

import sys

from dynamic_runner import TaskDeploymentSpec, run

from .memmap_builder_task import MemmapBuilderTask


def main(argv: list[str] | None = None) -> None:
    run(
        task=MemmapBuilderTask(),
        deployment=TaskDeploymentSpec(
            secondary_module="dynrunner.build_memmap",
            image_name="asm-tokenizer",
        ),
        description="Memory-mapped binary file builder (per-binary-group parallel).",
        argv=argv if argv is not None else sys.argv[1:],
    )


if __name__ == "__main__":
    main()
