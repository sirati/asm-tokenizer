# SLURM dispatch runbook

Recipe for running `dynrunner.tokenize` (and `dynrunner.build_memmap`) end-to-end on the LMU CIP SLURM cluster, plus what to do when something breaks.

The intended audience is a fresh subagent or a future-you returning to this after a few weeks. **Follow the commands verbatim.** Do not re-explore the codebase to "figure out" the dispatch flags; this document is the source of truth, and if it disagrees with the framework that's a bug to file (see "When the runbook is wrong" at the end).

## Prerequisites

- Working `nix develop` shell from `/home/sirati/devel/python/asm-tokenizer`. All commands run inside `nix develop --command bash -c "..."` unless explicitly stated otherwise.
- `~/.ssh/config` has the `lmu` host alias (or just use `kruppb@remote.cip.ifi.lmu.de` directly). 1Password SSH agent must be unlocked — if the gateway returns `signing failed for ED25519 "LMU CIP SSH Key" from agent: communication with agent failed`, the agent is locked and only the user can unlock it.
- `flake.lock` pinned to a `dynamic-runner` revision that contains the SLURM-path bug fixes (A–G; current minimum is `edde265` from 2026-05-01). Bump with `nix flake update dynamic-runner` and rebuild with `nix build --no-link .#dockerImage`.
- Image is rebuilt locally (`nix path-info .#dockerImage` → store path of the tar.gz). The runner uploads it via layered-blob transfer; only changed layers re-upload, so this is fast on iteration.

## The dispatch command

This is the canonical small-batch dispatch (filtered to `minigzipsh` only via `--debugs`, which currently matches 2 binaries on the gateway corpus):

```bash
nix develop --command bash -c '
  python -m dynrunner.tokenize \
    --multi-computer slurm \
    --packaging podman \
    --gateway ssh://kruppb@remote.cip.ifi.lmu.de \
    --slurm-root-folder /home/k/kruppb/BIG/slurm \
    --source ~/.cache/asm-tokenizer-srcbins-mirror \
    --output ~/.cache/asm-tokenizer-out-mirror \
    --debugs \
    --jobs 1 \
    --slurm-time-limit 30 \
    --skip-existing
' 2>&1 | tee /tmp/dispatch-$(date +%s).log
```

### Flag-by-flag rationale (do not omit any)

| Flag | Why |
|------|-----|
| `--multi-computer slurm` | Selects the SLURM dispatch pipeline. Don't use the deprecated `--slurm` flag. |
| `--packaging podman` | Required for SLURM; the cluster's wrapper uses rootless podman, not docker. |
| `--gateway ssh://kruppb@remote.cip.ifi.lmu.de` | Always this hostname; never substitute the per-session FQDN (`beryll`, `amazonit`, …) the load balancer happens to land you on. |
| `--slurm-root-folder /home/k/kruppb/BIG/slurm` | The gateway-side root for image, srcbins, out, log subfolders. Absolute path from gateway perspective. |
| `--source <local mirror>` | Local directory where `discover_items` walks to find binaries. Must contain a copy/mirror of what's on the gateway under `<slurm-root>/image_bin/srcbins/`; the framework cross-references hashes between local discovery and gateway-side mounts. |
| `--output <local mirror>` | Where the local primary collects the per-binary CSV outputs. The gateway-side `<slurm-root>/out/` is independent and is what the secondaries write to under their `--root` overlay. |
| `--debugs` | Filters to binaries named `minigzipsh`. Use this for first-time validation runs; only 2 binaries match, dispatch finishes in seconds once SLURM gives a node. |
| `--jobs 1` | One secondary node. Don't multi-spawn until the single-node path is green. |
| `--slurm-time-limit 30` | sbatch `--time` in minutes. Short enough that a runaway job auto-terminates. |
| `--skip-existing` | Idempotent re-runs: skip binaries with existing CSV output. Lets you re-dispatch without manually clearing the output dir. |

### What you do NOT need

- You do NOT manually create dirs on the gateway. The dispatcher creates `image_bin/`, `image_bin/srcbins/`, `out/`, `log/`, `log/run_<ts>/`, `log/run_<ts>/connection_info/` itself.
- You do NOT manually upload the image. It's transferred via layered-blob upload from the local nix store the first time it differs.
- You do NOT manually scp binaries to the gateway. The framework stages them via `StageFile` notifications during dispatch.
- You do NOT inspect the image with `python -c "import tarfile..."`. If you find yourself doing this, stop.
- You do NOT need `--skip-image-build` once you have a hash mismatch fixed. Use it ONLY when you've verified the gateway image already matches the local build (rare).

## Source corpus — pre-stage on the gateway, mirror locally

**The framework does NOT upload binaries.** Per the upstream owner: data placement is a user/cluster concern, not the framework's role. The dispatch contract is "the file at `<slurm-root>/image_bin/srcbins/<rel>` already exists on the gateway when dispatch starts; the framework will mount it at `/app/src-network/<rel>` inside the secondary container."

So **before any dispatch** the corpus must exist at the gateway path. Push it once (or upon corpus changes):

```bash
# Push your corpus to the gateway (one-time or when the corpus changes)
rsync -av --delete \
  ./your-corpus/ \
  kruppb@remote.cip.ifi.lmu.de:/home/k/kruppb/BIG/slurm/image_bin/srcbins/
```

The local `--source` directory the dispatcher walks is independent. It needs the same binary names so `discover_items` produces the right `TaskInfo` set; the contents matter only for local hash computation. If you want to keep them in sync mechanically, use a local mirror that's an rsync of the gateway path:

```bash
mkdir -p ~/.cache/asm-tokenizer-srcbins-mirror
rsync -av --delete \
  kruppb@remote.cip.ifi.lmu.de:/home/k/kruppb/BIG/slurm/image_bin/srcbins/ \
  ~/.cache/asm-tokenizer-srcbins-mirror/
```

For a `--debugs` smoke test only the 2 `minigzipsh` binaries need to be present in both places.

## Watching a running dispatch

The dispatcher emits structured log lines prefixed `INFO | HH:MM:SS |P|` (primary) and `|S0|` (secondary 0). Useful greps for a Monitor (do NOT tail the raw log into your context):

```bash
# Filter monitor — emit only state transitions and clear failure modes
tail -f /tmp/dispatch-*.log | \
  grep -E --line-buffered \
    'Phase [0-9]|Job submitted|Secondary connected|Worker [0-9]+|TaskCompleted|TaskFailed|NonRecoverable|Traceback|connection refused|Completed:'
```

Cluster-side liveness check from a wake-up script (every 60s):

```bash
ssh kruppb@remote.cip.ifi.lmu.de \
  "sacct -u kruppb --starttime $TEST_START_ISO --format JobID,State,Elapsed,ExitCode -P" \
  | head -20
```

## Inspecting a running secondary

When a SLURM job is in state `R`, find its node and inspect the container:

```bash
NODE=$(ssh kruppb@remote.cip.ifi.lmu.de "squeue -u kruppb -h -o %N -j <jobid>")
ssh -A -J kruppb@remote.cip.ifi.lmu.de kruppb@${NODE}.cip.ifi.lmu.de \
  'ASM_DIR=$(ls -dt /tmp/asm-* | head -1) && \
   podman --root $ASM_DIR/storage --runroot $ASM_DIR/run ps; \
   podman --root $ASM_DIR/storage --runroot $ASM_DIR/run logs --tail 50 $(podman --root $ASM_DIR/storage --runroot $ASM_DIR/run ps -q | head -1)'
```

`-A` (agent forwarding) and `-J` (ProxyJump through gateway) are both required; compute nodes are not directly reachable.

Gateway-side structured logs (per secondary): `~/BIG/slurm/log/run_<ts>/slurm_<jobid>.{out,err}`.

## Cleanup after a run

```bash
# Whether the run succeeded or failed
ssh kruppb@remote.cip.ifi.lmu.de 'squeue -u kruppb'              # any leftover jobs?
ssh kruppb@remote.cip.ifi.lmu.de 'scancel <stuck-jobid>'         # only if stuck
pkill -f 'ssh.*-J kruppb.*-R'                                    # local stray reverse tunnels
pkill -f 'python.*-m dynrunner.tokenize'                         # local primary if it didn't exit
```

`scancel` on the controller does NOT propagate a kill to the podman container on the compute node — observed bug, scope unclear (might be cluster-side, might be wrapper-side). After scancel, also:

```bash
ssh -A -J kruppb@remote.cip.ifi.lmu.de kruppb@${NODE}.cip.ifi.lmu.de \
  'pgrep -fa "dynrunner.tokenize.*--secondary" | awk "{print \$1}" | xargs -r kill -TERM'
```

## When the runbook is wrong

If the dispatch errors out with a message that contradicts what's documented here, the bug is one of:

1. **Framework regression in `dynamic_runner`** — file:line + legacy diff (`git show cab668ba^:dynamic_batch/<path>` in this repo) → bug report to peer Claude on `dryrunner-tokenizer` channel. Bugs A–G all came from the 2026-04-28 Rust port + packaging refactor; further regressions of the same shape are likely.
2. **Recipe drift here** — flag was renamed, default changed, etc. Update this file alongside the bumped commit.
3. **Cluster-side change** — wrapper script regenerated, BIG paths moved, podman version changed. Re-derive the gateway layout via `ssh kruppb@remote.cip.ifi.lmu.de 'ls ~/BIG/slurm/'` and update.

In all three cases the fix is durable: update the runbook (or upstream) so the next person doesn't repeat the diagnosis.

## Bug history (for context — do not re-debug)

| Bug | Symptom | Fix commit (in `dynamic_runner`) |
|-----|---------|-----------------------------------|
| A | Rust URL parser rejected hostnames in secondary connect address | `070f015` |
| B | tilde paths shlex-quoted preventing bash expansion | `070f015` |
| C | `pipeline.py` skipped `gateway.setup_port_forwarding()` | `b07f5e7` |
| D | SSH tunnel direction `-L` (legacy was `-R`) + filename + format mismatches | `cf0b6ca` |
| E | Rust primary bound `127.0.0.1:0` instead of caller-supplied `primary_quic_port` | `c009339` |
| F | `in_docker` checked `/.dockerenv` (docker-only); podman has `/run/.containerenv`; fix uses `/app/src-network` mountpoint as runtime-agnostic sentinel | `0d91947` |
| G | `_collect_binaries` and `_drive_rust_primary` independently called `find_matching_binaries`, possibly disagreeing → "Queued 0 StageFile notifications" with non-empty corpus | `edde265` |
| H-B | `setup.rs:130 handle_initial_assignment` lacked the `dispatch.rs:50` fail-loud guard for `resolved_path.is_none()`; cache miss silently passed primary's local path to the worker → first-attempt Recoverable, second-attempt NonRecoverable through the operational loop. Fixed by factoring the predicate into `report_unresolvable_task` and calling from both setup.rs and dispatch.rs. | `76500ac` |
| H-A / K | `secondary/setup.rs:wait_for_setup` had no `MessageType::StageFile` match arm — StageFile messages arriving between PeerInfo and InitialAssignment fell to `other =>` and were silently dropped. Fixed by inlining `staged_files` records into the `InitialAssignment` message; the secondary now registers staged files atomically with the assignment batch, eliminating the ordering hazard category-wide rather than just the specific arm. (The deeper fix; Option 1 from the H-A note.) | `1cc3b69` |
| J | `pipeline.py` calls `notify_stage_file(rel, rel)` but never uploads binaries; legacy's `_distribute_files` ZIP-batched + SCP'd. **Resolved by clarifying the contract**: data placement is a user/cluster concern, not the framework's. Pre-stage the corpus to `<slurm-root>/image_bin/srcbins/` via rsync before dispatch (see "Source corpus" section above). | _won't fix (upstream)_ |
| L | Cross-run tunnel cleanup `pkill -f 'ssh.*-L.*localhost'` matched nothing after Bug D's `-L`→`-R` flip. Trivial two-character fix to use `-R`. | `5848803` |
| M | `StageFile.file_hash` field carries `compute_task_hash` (DefaultHasher on path+identifier, 16-char hex) but `staging.rs` verifies via `compute_file_hash` (SHA256 of contents, 64-char hex). Hash schemes never match → every stage fails "hash mismatch". Fix: split the wire field into `file_hash` (task identity) + `content_hash` (SHA256). | `86887b9` |
| N | New `compute_file_content_hash` PyO3 function registered on `_native` but not re-exported in `python/dynamic_runner/__init__.py`. `pipeline.py` does `import dynamic_runner as _rs` and accesses `_rs.compute_file_content_hash` → AttributeError → primary aborts before sbatch submission. Trivial one-line fix in `__init__.py`. | _pending peer_ |

Plus two preserved-from-legacy notes: (N1) — **as of `02f3941` this is the live blocker**: end-to-end dispatch + tokenization succeeds (`completed=N/N`, workers `success=true`), but worker outputs land in `/app/out-tmp` (= host `/tmp/asm-XXXX/out`) which the wrapper's `trap cleanup EXIT` `rm -rf`s, so nothing reaches the durable `/app/out-network` mount. Filed with peer; pending fix. (N2) `--secondary-quic-port` argv flag is parsed but ignored — the secondary unconditionally binds `0.0.0.0:0`; peers learn the real port via `CertExchange`.

All bugs above were introduced by the 2026-04-28 packaging refactor (`1f8d0a1`). Pre-refactor working baseline is `cab668ba^:dynamic_batch/...`.
