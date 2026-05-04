# SLURM dispatch runbook

Recipe for running `dynrunner.tokenize` (and `dynrunner.build_memmap`) end-to-end on the LMU CIP SLURM cluster, plus what to do when something breaks.

The intended audience is a fresh subagent or a future-you returning to this after a few weeks. **Follow the commands verbatim.** Do not re-explore the codebase to "figure out" the dispatch flags; this document is the source of truth, and if it disagrees with the framework that's a bug to file (see "When the runbook is wrong" at the end).

## Prerequisites

- Working `nix develop` shell from `/home/sirati/devel/python/asm-tokenizer`. All commands run inside `nix develop --command bash -c "..."` unless explicitly stated otherwise.
- `~/.ssh/config` has the `lmu` host alias (or just use `kruppb@remote.cip.ifi.lmu.de` directly). 1Password SSH agent must be unlocked — if the gateway returns `signing failed for ED25519 "LMU CIP SSH Key" from agent: communication with agent failed`, the agent is locked and only the user can unlock it.
- `flake.lock` pinned to a `dynamic-runner` revision that contains the SLURM-path bug fixes (A–G + 1-9 from the 2026-05-04 lineage; current minimum is `8da909d` from 2026-05-04). Bump with `nix flake update dynamic-runner` and rebuild with `nix build --no-link .#dockerImage`.
- Image is rebuilt locally (`nix path-info .#dockerImage` → store path of the tar.gz). The runner uploads it via layered-blob transfer; only changed layers re-upload, so this is fast on iteration.

## The dispatch command

Canonical small-batch dispatch against cluster-resident data (filtered to `minigzipsh` across all 6 platforms / all compilers / all opts; ~235 binaries on the LMU `Dataset-1/zlib/` corpus):

```bash
nix develop --command bash -c '
  python -m dynrunner --task tokenize \
    --multi-computer slurm \
    --packaging podman \
    --gateway ssh://kruppb@remote.cip.ifi.lmu.de \
    --slurm-root-folder /home/k/kruppb/BIG/slurm \
    --source-already-staged /home/k/kruppb/BIG/Dataset-1/zlib \
    --source /tmp \
    --output ~/.cache/asm-tokenizer-out-slurm \
    --name-regex minigzipsh \
    --platform x86 x64 arm32 arm64 mips32 mips64 \
    --jobs 1 \
    --slurm-time-limit 30 \
    --raw-logs
' 2>&1 | tee /tmp/dispatch-$(date +%s).log
```

### Flag-by-flag rationale (do not omit any)

| Flag | Why |
|------|-----|
| `--multi-computer slurm` | Selects the SLURM dispatch pipeline. Don't use the deprecated `--slurm` flag. |
| `--packaging podman` | Required for SLURM; the cluster's wrapper uses rootless podman, not docker. |
| `--gateway ssh://kruppb@remote.cip.ifi.lmu.de` | Always this hostname; never substitute the per-session FQDN (`beryll`, `amazonit`, …) the load balancer happens to land you on. |
| `--slurm-root-folder /home/k/kruppb/BIG/slurm` | The gateway-side root for image, out, log subfolders. Absolute path from gateway perspective. |
| `--source-already-staged <gateway-abs>` | The gateway-side directory containing your binaries (e.g. cluster NFS). Discovery walks this remotely via SSH; the wrapper bind-mounts it into each secondary container at `/app/src-network` (read-only); no rsync, no zip-copy, no `StageFile` round-trip via the primary. |
| `--source /tmp` | Vestigial — the framework's local `--source` validation is skipped when `--source-already-staged` is set, but argparse still expects the flag. Pass any existing local dir; `/tmp` is fine. |
| `--output <local cache>` | Where the local primary mirrors completion telemetry. Actual CSV/PKL outputs land on the gateway under `<slurm-root>/out/<file>`. |
| `--name-regex <pattern>` | Filter binaries by basename component (the part after the format-string's last separator). Replaces the older `--debugs` shorthand. |
| `--platform x86 x64 arm32 arm64 mips32 mips64` | Allowlist of architectures matched against the format-string's `platform` field. Omit a platform to drop those binaries. |
| `--jobs N` | Number of SLURM secondaries to spawn. Start at 1 for first validation; bump to 2+ for multi-node fan-out. |
| `--slurm-time-limit 30` | sbatch `--time` in minutes. Short enough that a runaway job auto-terminates. |
| `--raw-logs` | Bypass log file rewrites. Recommended for ad-hoc dispatches; the wrapper script still tees the secondary's stdout/stderr into `<slurm-root>/log/run_<ts>/slurm_<jobid>.{out,err}`. |
| `--skip-existing` | (Optional) Idempotent re-runs: skip binaries with existing CSV output. Useful for re-dispatch without manually clearing the gateway out dir. |

### What you do NOT need

- You do NOT manually create dirs on the gateway. The dispatcher creates `image_bin/`, `out/`, `log/`, `log/run_<ts>/`, `log/run_<ts>/connection_info/` itself.
- You do NOT manually upload the image. It's transferred via layered-blob upload from the local nix store the first time it differs.
- You do NOT manually scp binaries to the gateway when using `--source-already-staged`. The data is already on the cluster; the wrapper bind-mounts it directly. (The legacy `StageFile`-based push is still available without `--source-already-staged`, but only useful if your data lives only on your local machine — rare for the LMU corpus.)
- You do NOT inspect the image with `python -c "import tarfile..."`. If you find yourself doing this, stop.
- You do NOT need `--skip-image-build` once you have a hash mismatch fixed. Use it ONLY when you've verified the gateway image already matches the local build (rare).

## Source corpus — `--source-already-staged` is the canonical path

For data that already lives on cluster NFS (typical for LMU's `~/BIG/Dataset-1/`), point `--source-already-staged` directly at the gateway-side path. The framework will:

1. Discover items by walking that path via SSH (using the existing `--gateway` connection — no Python required on the gateway, just GNU `find` + `ssh`).
2. Skip the primary's `StageFile` pass entirely (no transfer through the local primary).
3. Bind-mount the same path into each secondary container at `/app/src-network` (read-only) when the SLURM wrapper script launches.

Outputs land flat under `<slurm-root>/out/<filename>` mirroring the relative layout under `--source-already-staged`. If you point `--source-already-staged` at `Dataset-1` (one level up) you'll get `<slurm-root>/out/<package>/<file>`; pointing at `Dataset-1/zlib` (one level into a single package) gives flat `<slurm-root>/out/<file>`.

### Legacy: pushing a local-only corpus

If your data lives only on your local machine (rare), the framework's `StageFile` path still works without `--source-already-staged`: pass `--source <local-dir>` and omit `--source-already-staged`; the primary will hash + transfer each binary to the secondaries via `StageFile` notifications during dispatch. Slower (every byte goes over the wire), but works without the data being pre-staged on the cluster. Not recommended for sizes >100 MB total.

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

(N1) **Resolved 2026-05-04** — outputs DO land in the durable `/app/out-network` mount; earlier "rm-rf'd into /app/out-tmp" diagnosis was stale post-`fb1df86`. `8da909d` additionally fixed the directory-mirroring (worker's `--source` now matches the bind-mount root, so `relative_to(source_dir)` succeeds and outputs mirror the source layout under `<slurm-root>/out/`). (N2) `--secondary-quic-port` argv flag is parsed but ignored — the secondary unconditionally binds `0.0.0.0:0`; peers learn the real port via `CertExchange`.

All bugs above were introduced by the 2026-04-28 packaging refactor (`1f8d0a1`). Pre-refactor working baseline is `cab668ba^:dynamic_batch/...`.

## 2026-05-04 dispatch-path bug lineage (post-PR3 `--source-already-staged`)

PR3 (`eb69a80 feat(slurm): --source-already-staged`) shipped underspecified — the feature path had multiple unexercised dispatch sites. Eight follow-up bugs landed within the same day; recipe in this runbook reflects the post-fix state:

| # | Commit | Symptom | Root cause |
|---|--------|---------|------------|
| 1 | `144b9da` | `find: unknown predicate -L` on SSH discovery | GNU `find` requires `-L` *before* the path, not after. |
| 2 | `bf1ce02` | "not pre-staged" + hash-machinery 16-char vs SHA256 64-char mismatch | Pre-staged mode skips `StageFile` so cache is empty; secondary's hash-based resolver couldn't match. New `resolve_pre_staged` path. |
| 3 | `a344b0e` | "not pre-staged at /home/...gateway-abs/..." even after bf1ce02 | Wire's `local_path` was the gateway-absolute path; secondary's `src_network.join(local_path)` dropped LHS (Path::join with absolute RHS). Primary now strips `source_pre_staged_root` before wire emit. |
| 4 | `059f132` | `TypeError: source_pre_staged kwarg unexpected` | `pipeline.py` passed `source_pre_staged=bool(...)` but Rust pyclass had renamed to `source_pre_staged_root: Option<PathBuf>`. |
| 5 | `76d074a` | `cargoHash` mismatch in `nix/wheel.nix` | `38596aa`'s test-fixture commit added `tempfile` to `Cargo.lock` without bumping the recorded hash. |
| 6 | `796feff` | `NameError: slurm_config` in `_drive_rust_primary` | `059f132`'s patch referenced a variable not in scope. |
| 7 | `217093c` | Initial-batch tasks fail NonRecoverable while operational-loop tasks resolve correctly | `primary/assignment.rs:127` didn't use `wire_local_path`; only `task.rs` and `lifecycle.rs` were patched in `a344b0e`. |
| 8 | `8658c5b` | 219/235 Recoverable "Not a valid binary file" with gateway-abs paths | SLURM-promoted-secondary's self-assign path (`secondary/slurm.rs:224-236`) called the hash-verifying resolver instead of branching on `pre_staged_mode`. Refactor extracted `resolve_for_dispatch` helper called from all 4 dispatch sites. |
| 9 | `8da909d` | Outputs land at `<slurm-root>/out/src-network/<file>` instead of mirroring source layout | `_dispatch_secondary` passed `cfg.src_tmp` (not `cfg.src_network`) as worker's `--source`; worker's `relative_to(source_dir)` fell through to the parent-name fallback. |

End-to-end dispatch confirmed green at `8da909d` against 235 minigzipsh on `/home/k/kruppb/BIG/Dataset-1/zlib/` — 1-secondary 235/235 (run-5 reference); 2-secondary cross-node fan-out 235/235 (run-6).

## Open framework issues observed in run-7

- **Heartbeat clock not reset at connection-establishment.** Primary's heartbeat monitor checks `last_seen_s` against a fixed clock that started at primary startup, not at each secondary's connection. Containers that take >threshold to start + handshake (~38s for the LMU SLURM wrapper) are dropped immediately when the operational loop begins, with their in-flight tasks requeued to peers. The dropped secondary still completes its already-assigned batch independently before being isolated. Not blocking for our test (sec-0 picked up the requeue), but production-relevant for larger clusters with staggered scheduling. Filed with peer.
