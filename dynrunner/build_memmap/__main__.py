from dynamic_runner import make_subprocess_spawn_factory, run

from .memmap_builder_task import MemmapBuilderTask


def main() -> None:
    run(
        task=MemmapBuilderTask(),
        spawn_secondary_factory=make_subprocess_spawn_factory("dynrunner.build_memmap"),
        description="Memory-mapped binary file builder (per-binary-group parallel).",
    )


if __name__ == "__main__":
    main()
