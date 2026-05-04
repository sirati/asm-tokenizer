"""Dynrunner entry point for vocabulary unification.

Vocab unification has no parallelism to exploit (each binary's vocab
registers onto a shared `VocabularyManager`), but it's now a runner
task so the full 3-phase pipeline (tokenize → unify-vocab →
build-memmap) can run autonomously on SLURM. The dispatch produces
exactly one TaskInfo, dispatched to one worker on one secondary;
phase 1 and phase 3 retain their multi-worker parallelism.
"""

from dynamic_runner import TaskDeploymentSpec, run

from .vocab_unifier_task import VocabUnifierTask


def main() -> None:
    run(
        task=VocabUnifierTask(),
        deployment=TaskDeploymentSpec(
            secondary_module="dynrunner.unify_vocab",
            image_name="asm-tokenizer",
        ),
        description=(
            "Vocabulary unification: build a single shared vocab + "
            "per-binary mapping files from the per-binary CSVs the "
            "tokenize phase produced."
        ),
    )


if __name__ == "__main__":
    main()
