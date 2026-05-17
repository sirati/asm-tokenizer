"""Dynrunner entry point for the full tokenize → unify → memmap pipeline.

Replaces the historical three-dispatch ``--task all`` chain with a
single dispatch that hands the framework a composite
:class:`~dynrunner.full_pipeline.FullPipelineTask`. The framework
then drives all three phases on one persistent secondary mesh — one
sbatch wave under SLURM, one Docker-image upload, one mesh
formation round.
"""

from dynamic_runner import TaskDeploymentSpec, run

from .full_pipeline_task import FullPipelineTask


def main() -> None:
    run(
        task=FullPipelineTask(),
        deployment=TaskDeploymentSpec(
            secondary_module="dynrunner.full_pipeline",
            image_name="asm-tokenizer",
        ),
        description=(
            "Three-phase tokenize → unify-vocab → build-memmap pipeline "
            "driven as a single framework dispatch (persistent mesh; "
            "one sbatch wave under SLURM)."
        ),
    )


if __name__ == "__main__":
    main()
