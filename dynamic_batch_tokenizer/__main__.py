import subprocess
import sys

from dynamic_batch.cli import run

from .tokenizer_task import TokenizerTask


def _make_spawn_secondary(args):
    """Create a spawn_secondary callback for the tokenizer task."""

    def spawn_secondary(primary_url: str, secondary_id: str, quic_port: int):
        cmd = [sys.executable, "-m", "dynamic_batch_tokenizer"]
        cmd += ["--secondary", primary_url]
        cmd += ["--secondary-id", secondary_id]
        cmd += ["--secondary-quic-port", str(quic_port)]
        if args.raw_logs:
            cmd.append("--raw-logs")
        return subprocess.Popen(cmd)

    return spawn_secondary


def main():
    run(
        task=TokenizerTask(),
        spawn_secondary_factory=_make_spawn_secondary,
        description="Dynamic batch processing for binary tokenization with memory-aware parallel execution",
    )


if __name__ == "__main__":
    main()
