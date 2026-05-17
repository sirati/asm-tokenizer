"""Run asm-tokenizer's tokenize task with each secondary in its own podman container.

Bypasses `dynamic_runner.run` (which is subprocess-only) and calls
`run_primary` directly with a podman spawn callback. The primary
listens on 127.0.0.1; containers join via --network=host so the
loopback URL resolves.

Cgroup story: the parent shell is meant to run inside a
`systemd-run --user --scope` with the global limits set. We pass
`--cgroup-parent` so each container's cgroup nests under that scope,
which makes the kernel share the budget across the primary process
and every container.

Outputs end up under /tmp/asm-podman-multi/out-secondary-<i>/ — one
host dir bind-mounted per container as /app/out-tmp.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Make the asm-tokenizer's `dynrunner.tokenize.tokenizer_task` importable
# without going through `dynamic_runner.run`'s argparse.
REPO_ROOT = Path("/home/sirati/devel/python/asm-tokenizer")
sys.path.insert(0, str(REPO_ROOT))

import dynamic_runner as _rs  # noqa: E402
from dynrunner.tokenize.tokenizer_task import TokenizerTask  # noqa: E402


IMAGE = "localhost/asm-tokenizer:latest"
INPUT_DIR = REPO_ROOT / "src" / "zlib"


def discover_outer_cgroup() -> str | None:
    """Read /proc/self/cgroup to find the systemd scope we are inside.
    Returns a path like `/user.slice/user-1000.slice/user@1000.service/app.slice/run-rXXXXXX.scope`,
    or None if we are not inside a transient scope."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            # cgroup v2: "0::<path>"
            if line.startswith("0::"):
                return line[3:]
    except OSError:
        return None
    return None


def make_podman_spawn(
    *,
    input_host: Path,
    output_root: Path,
    raw_logs: bool,
    cgroup_parent: str | None,
    container_cpus: float,
    container_memory: str,
    extra_runner_flags: list[str],
):
    """Return a `spawn_secondary(primary_url, secondary_id, quic_port)` callback
    that podman-runs the asm-tokenizer image as a secondary."""

    def spawn(primary_url: str, secondary_id: str, quic_port: int) -> subprocess.Popen:
        per_secondary_out = output_root / f"out-{secondary_id}"
        per_secondary_out.mkdir(parents=True, exist_ok=True)
        per_secondary_tmp = output_root / f"tmp-{secondary_id}"
        per_secondary_tmp.mkdir(parents=True, exist_ok=True)

        cmd = [
            "podman", "run", "--rm",
            "--name", f"asm-{secondary_id}",
            "--network=host",
            # Workers see the host's loopback for QUIC. The shared
            # cgroup-parent makes the container budget nest inside the
            # outer systemd scope (so kernel-level enforcement is shared).
            # `--cgroup-manager=cgroupfs` is required for podman to
            # accept a `.scope` parent — its default systemd manager
            # only accepts `.slice` parents.
            "--cgroup-manager=cgroupfs",
            f"--cpus={container_cpus}",
            # No `--memory` here: under cgroupfs+systemd-scope-parent the
            # outer scope hasn't enabled the memory controller in
            # `cgroup.subtree_control`, so crun fails to write
            # `memory.max` for the container's nested cgroup. The outer
            # scope's MemoryMax is what enforces total RAM anyway, so
            # the per-container slice is informational at best.
        ]
        if cgroup_parent:
            cmd.extend(["--cgroup-parent", cgroup_parent])
        # Bind-mount the input dir at TWO paths inside the container:
        #   - the same absolute host path the primary used to discover
        #     binaries (so `relative_path` over the wire, which is
        #     actually an absolute host-side path in the current
        #     dispatch model, resolves inside the container)
        #   - /app/src-network (the conventional mount the secondary
        #     CLI defaults expect)
        # Without the first mount, workers see "Not a valid binary
        # file: <host-path>" because the host path doesn't exist in
        # the container.
        # Bind-mount the input dir at TWO paths inside the container:
        #   - the same absolute host path the primary used to discover
        #     binaries (so the workers' `relative_path`, which arrives
        #     as the absolute host-side path in the current dispatch
        #     model, resolves inside the container)
        #   - /app/src-network (the conventional mount the secondary
        #     CLI defaults expect)
        # `-v /dev/null:/.dockerenv:ro` creates the `/.dockerenv` marker
        # so `_dispatch_secondary`'s in_docker check passes and outputs
        # land in /app/out-tmp (the bind-mount) instead of an ephemeral
        # tmpdir inside the container.
        cmd.extend([
            "-v", f"{input_host}:{input_host}:ro",
            "-v", f"{input_host}:/app/src-network:ro",
            "-v", f"{per_secondary_out}:/app/out-tmp",
            "-v", f"{per_secondary_tmp}:/app/src-tmp",
            "-v", "/dev/null:/.dockerenv:ro",
            IMAGE,
            "dynrunner", "--task", "tokenize",
            "--secondary", primary_url,
            "--secondary-id", secondary_id,
            "--secondary-quic-port", str(quic_port),
            "--src-network", str(input_host),
            "--src-tmp", str(input_host),
        ])
        if raw_logs:
            cmd.append("--raw-logs")
        cmd.extend(extra_runner_flags)
        logging.info("[%s] spawning: %s", secondary_id, " ".join(cmd))
        return subprocess.Popen(cmd)

    return spawn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-secondary podman test for asm-tokenizer.",
    )
    parser.add_argument("--num-secondaries", type=int, default=2)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-root", type=Path,
                        default=Path("/tmp/asm-podman-multi"))
    parser.add_argument("--name-regex", default="minigzipsh")
    parser.add_argument("--platform", nargs="+",
                        default=["x86", "x64", "arm32", "arm64", "mips32"])
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument("--raw-logs", action="store_true")
    parser.add_argument("--keep-output", action="store_true")
    # Per-container budget. Defaults pin each secondary to exactly one
    # worker so the workload spreads across containers (otherwise the
    # SLURM-primary's local manager drains the small set before peer
    # secondaries see any cross-secondary assignments).
    parser.add_argument("--container-cpus", type=float, default=1.0)
    parser.add_argument("--container-memory", default="4G")
    parser.add_argument("--secondary-cores", type=int, default=1,
                        help="Per-secondary --cores forwarded to dynrunner; "
                             "1 means the manager spawns a single worker.")
    parser.add_argument("--secondary-max-memory", default="4G",
                        help="Per-secondary --max-memory forwarded to dynrunner.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output_root.exists() and not args.keep_output:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    cgroup = discover_outer_cgroup()
    logging.info("outer cgroup: %s", cgroup)

    # Build the binaries list at the orchestrator (= "starting instance"
    # in the new task_protocol vocabulary). `TokenizerTask.discover_items`
    # is the canonical entry point: it composes selection-args parsing,
    # walk_dataset, name/platform/compiler/opt filtering, Ghidra-artifact
    # skipping (Bug #2), and pkg-group + size-desc sorting in one pass.
    # The orchestrator just supplies the args Namespace shape TokenizerTask
    # expects.
    selection_args = argparse.Namespace(
        source=str(args.input_dir),
        output=str(args.output_root),
        platform=args.platform,
        compiler=args.compiler,
        compiler_versions=None,
        opt=None,
        opt_regex="[oO]?([0123s]|fast|z)",
        version_regex=None,
        name_regex=args.name_regex,
        exclude_subfolder=None,
        list_files=False,
        file_format="platform-compiler-version-optimisationlevel_binaryname",
        debugs=False,
        source_already_staged=None,
        gateway=None,
        skip_existing=False,
    )
    task = TokenizerTask()
    binaries = list(task.discover_items(Path(args.input_dir), selection_args))
    logging.info("discovered %d input binaries", len(binaries))
    if not binaries:
        logging.error("no binaries matched; check --input-dir and filters")
        return 1

    # Forward a per-secondary resource budget so the worker manager
    # plans against the same numbers podman is enforcing at the cgroup.
    # With --secondary-cores 1 the manager spawns exactly one worker.
    extra_runner_flags: list[str] = [
        "--cores", str(args.secondary_cores),
        "--max-memory", args.secondary_max_memory,
    ]

    spawn_secondary = make_podman_spawn(
        input_host=args.input_dir,
        output_root=args.output_root,
        raw_logs=args.raw_logs,
        cgroup_parent=cgroup,
        container_cpus=args.container_cpus,
        container_memory=args.container_memory,
        extra_runner_flags=extra_runner_flags,
    )

    primary_cfg = _rs.PrimaryConfig(num_secondaries=args.num_secondaries)
    coord = _rs.RustPrimaryCoordinator(
        args.num_secondaries, task, spawn_secondary,
    )
    try:
        coord.run(binaries)
    except Exception as exc:
        logging.exception("primary coordinator failed: %s", exc)
        return 2

    logging.info("=" * 60)
    logging.info("completed=%d failed=%d", coord.completed, coord.failed)
    for sub in sorted(args.output_root.iterdir()):
        if sub.is_dir() and sub.name.startswith("out-"):
            csvs = list(sub.rglob("*_output.csv"))
            logging.info("  %s: %d *_output.csv", sub.name, len(csvs))
    return 0 if coord.failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
