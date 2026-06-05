from __future__ import annotations

import sys

from dynamic_runner import TaskDeploymentSpec, run

from .tokenizer_task import TokenizerTask


def main(argv: list[str] | None = None) -> None:
    run(
        task=TokenizerTask(),
        deployment=TaskDeploymentSpec(
            secondary_module="dynrunner.tokenize",
            image_name="asm-tokenizer",
        ),
        description="Dynamic batch processing for binary tokenization with memory-aware parallel execution",
        argv=argv if argv is not None else sys.argv[1:],
    )


if __name__ == "__main__":
    main()
