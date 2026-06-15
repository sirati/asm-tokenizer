from __future__ import annotations

import sys

from dynamic_runner import TaskDeploymentSpec, run

from .build_index_task import BuildIndexTask


def main(argv: list[str] | None = None) -> None:
    run(
        task=BuildIndexTask(),
        deployment=TaskDeploymentSpec(
            secondary_module="dynrunner.build_index",
            image_name="asm-tokenizer",
        ),
        description=(
            "Per-binary index build (realized-length sidecars + "
            "sorted-index .idx; sorted-index depends on realized-length "
            "per binary)."
        ),
        argv=argv if argv is not None else sys.argv[1:],
    )


if __name__ == "__main__":
    main()
