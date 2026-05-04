from dynamic_runner import TaskDeploymentSpec, run

from .tokenizer_task import TokenizerTask


def main():
    run(
        task=TokenizerTask(),
        deployment=TaskDeploymentSpec(
            secondary_module="dynrunner.tokenize",
            image_name="asm-tokenizer",
        ),
        description="Dynamic batch processing for binary tokenization with memory-aware parallel execution",
    )


if __name__ == "__main__":
    main()
