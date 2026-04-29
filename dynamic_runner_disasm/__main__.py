"""Acceptance-test entry: prove the runner is task-agnostic by using
the same `dynamic_runner.run` facade and `make_subprocess_spawn_factory`
helper that the tokenizer package uses, with a totally different task.
"""

from dynamic_runner import make_subprocess_spawn_factory, run

from .disasm_task import DisasmTask


def main() -> None:
    run(
        task=DisasmTask(),
        spawn_secondary_factory=make_subprocess_spawn_factory("dynamic_runner_disasm"),
        description="Acceptance-test sibling task package: hypothetical disassembler",
    )


if __name__ == "__main__":
    main()
