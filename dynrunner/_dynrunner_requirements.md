# Required dynamic_runner changes for asm-tokenizer's dynrunner integration

Found while migrating asm-tokenizer to the new task_protocol API on
`dynamic_runner` main (`b1b97431d0679ed5c8ba7df9c801471e697d61b5`).
These items must land upstream before asm-tokenizer's pipeline runs
end-to-end against the new API.

## 1. `dynamic_runner.run()` does not call `discover_items`

**Where:** `python/dynamic_runner/run.py:82-128` (`_collect_binaries`).

**Current behaviour:** the framework calls `find_matching_binaries`
itself and then passes the result to the legacy method
`task.organize_and_sort_items(binaries)`. The new-API method
`task.discover_items(source_dir, args)` is never invoked from the
`run()` entry point; it only runs in the test harness which calls
`run_local` / `run_distributed` directly.

**Impact on asm-tokenizer:** every `dynrunner/<task>/` package on the new
API must keep an `organize_and_sort_items` compat method (returning the
already-`discover_items`-ordered items unchanged) or `run()` raises
`AttributeError`. New-API consumers paying for the migration get no
benefit from `discover_items` until this is wired.

**Suggested fix:** in `_collect_binaries`, branch on
`hasattr(task, "discover_items")`: if present, call
`list(task.discover_items(config.source_dir, args))` and skip the
framework-side `find_matching_binaries` call. Existing `--list-files`
/ `skip_existing` filters can run on the result the same way. Keep the
old path as a fallback for one release, gated on the absence of
`discover_items`.

## 2. `TaskInfo.payload` is not delivered to workers

**Where:**
`crates/dynrunner-protocol-manager-worker/src/codec.rs:24` —
`Command::ProcessTask` carries only `relative_path`. The
`task_payloads` field on `LocalManager` (and the wire
`TaskAssignment.payload_json` between primary/secondary) are
**output-side only** (the worker emits `done:<bytes>` which the
framework collects).

**Impact on asm-tokenizer:** any task that needs structured per-item
input (e.g. memmap building's paired CSV/mapping list per binary
group) cannot get it through the runner. The architectural rule
"workers never plan or discover" is unenforceable as long as the only
input a worker receives is a path string. Two workarounds, neither
ideal:

  - Encode everything in the path itself (only works if the path
    parses to the full required identifier, like the tokenizer's
    filename → `BinaryIdentifier` round-trip).
  - Have `discover_items` write a per-item manifest file to a known
    location (e.g. `output_dir/.dynrunner-<task>/<key>.json`) and set
    `TaskInfo.path` to the manifest. The worker reads its manifest
    instead of its raw input file. asm-tokenizer's
    `dynrunner/build_memmap/` uses this pattern.

**Suggested fix:** extend `Command::ProcessTask` with an optional
opaque payload (base64-JSON of `TaskInfo.payload` would be the obvious
choice, line-delimited just like the rest of the codec). The Rust
side already carries `TaskInfo<I>` end-to-end inside
`PendingPool` / `LocalManager`, so the data is available at the
dispatch site — only the wire format needs the slot. Python workers
gain a `command.payload: dict` field they can read instead of
re-deriving state.

## 3. Resource discovery is not cgroup-aware

**Where:** `python/dynamic_runner/system_resources.py` (and the
secondary's RAM probe at
`crates/dynrunner-manager-distributed/src/secondary/setup.rs`).

**Current behaviour:** `psutil.virtual_memory().total` and
`psutil.cpu_count()` read system-wide values from `/proc/meminfo` and
the kernel cpuset, not from the cgroup the process is in. When
asm-tokenizer runs inside a `systemd-run --scope -p MemoryMax=24G -p
CPUQuota=1600%` (or any container/cgroup-limited environment), each
secondary reports `ram_gb=91.87` and the manager's scheduler budgets
against that, while the kernel still enforces the cgroup at 24 GiB.
Result: the runner over-commits and, on a non-toy workload, the
cgroup OOM-killer fires.

**Suggested fix:** read `/sys/fs/cgroup/memory.max` (cgroup v2) or the
v1 equivalent first, fall back to `/proc/meminfo` only when the
cgroup isn't memory-constrained. Same idea for cpus: prefer
`/sys/fs/cgroup/cpu.max` (the `quota period` pair maps to fractional
cores) then fall back to `psutil.cpu_count()`. The Rust
`secondary/setup.rs` has the same issue and needs the same fix.

### 3a. Secondary path silently drops `--cores` and `--max-memory`

**Where:** `python/dynamic_runner/run.py:221-227` (`_dispatch_secondary`).

**Current behaviour:** the secondary's `argparse` accepts `--cores` /
`--max-memory` (the same flags the primary takes), but
`_dispatch_secondary` builds `SecondaryConfig` from
`psutil.virtual_memory().total` and
`psutil.cpu_count(logical=False) or 4` instead of `args.cores` /
`args.max_memory`. The user-supplied override is parsed and discarded.

**Impact on asm-tokenizer:** the multi-secondary podman test at
`test/multi_secondary/podman_orchestrator.py` cannot get a single
worker per container even with `--cores 1` forwarded into the
container — every secondary spawns 16 workers (one per host physical
core) and the SLURM-promoted secondary drains the entire workload
before peer secondaries see any cross-secondary assignment. Effect:
multi-secondary dispatch can't be observed on small workloads.

**Suggested fix:** in `_dispatch_secondary` honor `args.cores` /
`args.max_memory` when present (same `parse_cores` / `parse_memory`
helpers the primary uses), fall back to the cgroup-aware probe, fall
back to psutil last. This is also the only API the test harness has
to constrain a containerized secondary to fewer workers than the
host's physical-core count — `--cpus` and `--cpuset-cpus` on the
container don't reach `psutil.cpu_count(logical=False)`, which reads
`/proc/cpuinfo` and ignores the cgroup.

## 4. Chained TaskDefinitions in one primary session (workflow API)

**Motivation:** asm-tokenizer's pipeline is three TaskDefinitions
run in strict sequence — `TokenizerTask` (per-binary tokenization) →
`VocabUnifierTask` (single-instance aggregator) →
`MemmapBuilderTask` (per-binary-group memmap build). They are
*separate concerns* (different worker modules, different per-item
shapes, different resource curves) and the user has explicitly asked
that they not be merged into one mega-TaskDefinition — "tasks are
units of work, not phase orchestration".

**Current state:** each invocation of `dynamic_runner.run(task=…)`
runs exactly one TaskDefinition, then exits. The CLI `--task all`
in asm-tokenizer's dispatcher (`dynrunner/__main__.py`) approximates
end-to-end execution by calling the framework three times back-to-
back from the same local Python process. Side effects:

  - The image is uploaded **three times** to the gateway (each
    `dynamic_runner.run(...)` re-runs the layered transfer).
  - The SLURM secondary fleet is allocated, ramped, drained, and
    deallocated three times.
  - The local primary must stay alive across all three invocations.
    A planned local disconnect (e.g. laptop sleep) loses the workflow
    even though one of the secondaries was eligible to be promoted to
    primary on failover.
  - The single-instance `unify_vocab` task spends a full SLURM job
    submission + image-pull cycle just to run one Python invocation
    on one node.

**Asked feature:** allow a primary session to carry a *sequence* of
TaskDefinitions, where:

  1. The sequence is declared up-front at run start (so failover
     observers know the full plan).
  2. The image upload happens **once** for the whole sequence —
     identical image, identical layered cache, no re-transfer.
  3. Worker fleets may differ between TaskDefinitions in the sequence
     (different `worker_module`s, different memory profiles), so the
     framework needs the ability to drain phase-N workers and ramp
     phase-(N+1) workers without tearing down the secondary fleet
     itself. The secondary holds the container; the worker pool
     inside it is recyclable.
  4. **Failover-aware:** the active TaskDefinition index, plus its
     in-flight progress, is part of the primary's replicated state.
     If a SLURM secondary is promoted to primary mid-pipeline (e.g.
     after the local primary's planned disconnect), the new primary
     resumes at the current TaskDefinition with the queues and
     completion sets intact.
  5. Item discovery for downstream TaskDefinitions runs **at the
     start of that TaskDefinition** (not at the start of the
     sequence). Reason: phase-2/phase-3 inputs are produced by
     phase-1; we need their on-disk state to plan against. asm-
     tokenizer's planner can predict the *expected* paths from the
     phase-1 input list (see resilience note below) but it should
     skip versions whose inputs aren't actually present, and the
     only timing where that's correct is at phase-N start.

**API sketch (illustrative, not prescriptive):**

```python
from dynamic_runner import TaskWorkflow, TaskDeploymentSpec, run_workflow

run_workflow(
    workflow=TaskWorkflow(
        name="asm-tokenizer-pipeline",
        steps=(
            TokenizerTask(),
            VocabUnifierTask(),
            MemmapBuilderTask(),
        ),
    ),
    deployment=TaskDeploymentSpec(
        secondary_module="dynrunner.tokenize",  # bootstrap module; the
                                                 # workflow chooses the
                                                 # actual worker per step.
        image_name="asm-tokenizer",
    ),
    description="...",
)
```

A sentinel "no-op step" or per-step `enabled` flag would let
operators run a sub-sequence (e.g. "memmap only, inputs are already
on disk") without rebuilding a new entry point.

**Resilience contract for downstream steps (asm-tokenizer side):**
the asm-tokenizer worker side already implements per-version skip
for phase 2/3 (see `tokenizer/vocab_unifier/unifier.py` and
`dynrunner/build_memmap/worker.py`): missing inputs are logged and
skipped; only fail unrecoverably if **every** input is missing.
That contract relies on the framework letting phase-N start even
when phase-(N-1) had partial failures — i.e. phase dependencies must
be "barrier on completion" not "barrier on success". Today's single-
TaskDefinition `PhaseSpec.depends_on` already meets this if "drain"
counts both successes and failures.

**Out of scope for this request:** changing `TaskDefinition` itself.
The unit-of-work abstraction stays exactly as it is; the new layer
sits *above* it and orchestrates a sequence of them.

## 5. Document the `/app/out-tmp` vs `/app/out-network` mount split

**Where:** the SLURM wrapper bind-mounts BOTH `/app/out-tmp`
(ephemeral, per-job, rm-rf'd on container exit) and
`/app/out-network` (durable, gateway-side) into every secondary
container — `crates/dynrunner-slurm/src/wrapper_script.rs:174,177`
plus the Python equivalent in `packaging/job_manager.py:227-246`.
But the framework auto-points the worker's `--output` at
`/app/out-network` directly
(`crates/dynrunner-pyo3/src/config/primary_secondary.rs:180-182`,
const `WRAPPER_OUT_NETWORK`).

That's a deliberate design choice — the framework provides the
mount infrastructure but treats output staging as an application
concern. The runbook bug-history note N1 alludes to a previous
design that did stage in out-tmp and then "rm-rf'd" prematurely;
the current design avoids that hazard by writing straight to the
durable mount.

**The undocumented bit:** task authors writing new TaskDefinitions
need to know that:
  - Killing the worker mid-write (SLURM time-out, OOM, container
    SIGKILL) leaves partial files at the canonical filename in
    `/app/out-network`, where the next run's `--skip-existing`
    filter will see them.
  - The application is responsible for atomic-publishing its
    outputs. Either by writing to a `.partial` sibling and
    renaming inside `/app/out-network` (intra-mount atomic) or by
    using `/app/out-tmp` as scratch and copying to
    `/app/out-network` on DoneResponse (cross-mount, requires
    copy+fsync+rename for durability).
  - The `/app/out-tmp` mount is provided for exactly this purpose
    but its use is opt-in — framework neither writes there nor
    reads from there.

**Asked:** add a "Writing a TaskDefinition that produces durable
outputs" section to the framework README / `task_protocol.py`
docstring covering the above. The mount layout is already in the
secondary config docstring, but the *behavioral contract* (worker
writes to `/app/out-network`, application owns crash-safety, here
are the patterns) isn't documented anywhere — the only way to find
it is to grep the wrapper script and config code, which is what
caused us to miss this on the asm-tokenizer side and ship a
non-atomic write path that needed two passes to fix
(`tokenizer/run_tokenizer.py:159-256` `.partial`-rename, then a
follow-up to use `/app/out-tmp` for genuine cross-mount staging).

## 6. SLURM wrapper `podman run` lacks `--pull=never`, falls through to docker.io

**Where:** `crates/dynrunner-slurm/src/wrapper_script.rs:174-180`
(and the Python equivalent in
`packaging/job_manager.py:224-234,240-250`) — both compose a
`podman run --rm --network host -v ... {image_name}:{image_tag} ...`
without `--pull=never` or any other flag pinning resolution to the
local store.

**Symptom (run-7 phase 1, 2026-05-04):** secondary-2 on
`felsit.cip.ifi.lmu.de` failed exit 125 *after* the bash script had
logged "Image loaded successfully":

```
podman run asm-tokenizer:latest …
  → Trying to pull docker.io/library/asm-tokenizer:latest
  → requested access to the resource is denied
```

The cold-load OCI archive in `$LOCAL_IMAGE` got loaded into the
per-job podman store, but the subsequent `podman run` resolved the
unqualified tag `asm-tokenizer:latest` to `docker.io/library/...`
(podman's last-resort registry). The dispatcher then hung 8+ minutes
waiting for the missing 4th secondary's `SecondaryWelcome` and exited
with `primary coordinator failed: timeout waiting for secondaries:
3/4 sent SecondaryWelcome`.

Did not recur on retry, so it's a transient — likely a podman tag-
inference race when the OCI archive's manifest doesn't pin a fully-
qualified registry/repo prefix. But "transient that wastes 10
minutes when it fires" is still a reliability bug.

**Asked fix:** `--pull=never` on the `podman run` lines (both
reverse_connection and standard branches in `wrapper_script.rs`).
Failure mode then becomes immediate "image not in local store" rather
than a 30-second remote-pull-then-deny that masquerades as a hung
secondary. Optionally also tag the loaded image as `localhost/...`
explicitly — `podman load` on a manifest with a bare `name:tag`
sometimes registers under `localhost/name:tag` and the bare run
reference doesn't find it. Both fixes together would make this
class of flake unreachable.

