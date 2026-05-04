from dynamic_runner import TaskDeploymentSpec, run

from .memmap_builder_task import MemmapBuilderTask


def main() -> None:
    run(
        task=MemmapBuilderTask(),
        deployment=TaskDeploymentSpec(
            secondary_module="dynrunner.build_memmap",
            image_name="asm-tokenizer",
        ),
        description="Memory-mapped binary file builder (per-binary-group parallel).",
    )


if __name__ == "__main__":
    main()
