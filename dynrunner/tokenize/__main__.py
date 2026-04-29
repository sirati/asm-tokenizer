from dynamic_runner import make_subprocess_spawn_factory, run

from .tokenizer_task import TokenizerTask


def main():
    run(
        task=TokenizerTask(),
        spawn_secondary_factory=make_subprocess_spawn_factory("dynrunner.tokenize"),
        description="Dynamic batch processing for binary tokenization with memory-aware parallel execution",
    )


if __name__ == "__main__":
    main()
