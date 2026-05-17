# LMU CIP operations notes

Companion to `SLURM_RUNBOOK.md`. The runbook explains **how to dispatch**; this document captures the **firm LMU-specific operational policy, shared-cap coordination state, and the running pin-lineage observation log** that the runbook intentionally keeps out of its flag table.

If you are a fresh subagent or a future-me reading this after some weeks: read `SLURM_RUNBOOK.md` first for the canonical recipe, then read this file for the LMU-specific firm overrides and coordination state.

## Firm LMU CLI overrides

| Flag | LMU value | Why firm |
|---|---|---|
| `--slurm-partition Krater` | `Krater` (40-node partition) | Framework default is `All` (131 nodes, union of all rock-named nodes). `All` works — sbatch accepts it, jobs run — but is the wrong placement for asm-tokenizer and adjacent dispatches. User flagged this 2026-05-15 after a Tier-3 run landed on `All`. The 40-node `Krater` partition is comfortably more than our 15-job cap, so concurrency is not constrained. |
| `--jobs 15` | `15` (kruppb cap) | kruppb has a 15-parallel-task quota across all kruppb SLURM jobs at LMU. This is shared with asm-dataset-nix (compiler_suit_runner). Pre-flight orphan-scan is mandatory or the next run won't get full quota. |
| `--slurm-time-limit 10080` | `10080` minutes (= 1 week) | sbatch `--time` in minutes. Production corpus-scale dispatches **must** use the 1-week ceiling. A SLURM TIMEOUT mid-Phase-1 triggers the framework's orphan-conmon bug (2026-05-17 LMU incident: `proctrack/cgroup` reaps the wrapper + its watchdog, but rootless-podman conmon lives in `user.slice` outside the SLURM cgroup and survives — leaving 14 orphan containers writing to NFS for 30+ min post-TIMEOUT). Until the framework's shutdown-manager rearchitecture lands (forwarded to `dynrunner-owner` on 2026-05-17), TIMEOUT is **not** a safe terminator on LMU; the only safe way out of a runaway dispatch is `scancel` (also affected, but with bounded blast radius — see `scancel` notes below). Short smokes can use 240 (4h); never go below that on real LMU. |

### Setup deadline auto-scales — do NOT pass `--slurm-setup-deadline-secs`

Since `ba889cd`, the framework auto-computes the per-secondary setup deadline as `max(60, num_secondaries * 15)` (`crates/dynrunner-slurm/src/pipeline.rs::compute_setup_deadline_secs`). The 15s/secondary slope was calibrated against LMU Krater empirical observation. For `--jobs 15` you get 225s automatically; for `--jobs 32` you get 480s. The 60s floor covers `--jobs 1..4`.

Override with `--slurm-setup-deadline-secs N` ONLY if you're on a cluster slower than LMU (or running a probe that intentionally needs to wait longer). On LMU Krater the auto-scaled value is the validated default; do not pass the flag.

## Do NOT confuse with slurm-test-env mandates

The following are **slurm-test-env-only** flags. They are NOT LMU policy and should NOT be carried over to LMU dispatches:

| Test-env-only flag | Test-env reason | What LMU does instead |
|---|---|---|
| `--cores 2` | Test-env nodes are 2-CPU rootless-podman boxes; framework auto-detect would spawn way more workers than the cgroup allows. Required for every test-env smoke (memory `feedback_test_worker_cap.md`). | Do NOT pass `--cores 2` on LMU. LMU compute nodes are 14-CPU per `--slurm-cpus-per-task`; let the framework default size workers to the node, or pass a value matched to the node's cpus-per-task. Forcing `--cores 2` on LMU cripples production throughput. |
| `--ssh-identity-file <key>` | Test-env generates per-instance keypairs at provision time; the user dispatches as `kruppb@localhost:<SSH_PORT>` using that key, so the file path is the only auth route. | LMU's canonical auth is the 1Password SSH agent. Do NOT pass `--ssh-identity-file` for LMU dispatches; let the framework use the agent. `ssh-add -L` returning "no identities" is normal 1Password behaviour — auth still works via per-prompt approval. |
| `--slurm-cpus-per-task 2` | Test-env framework-default `--slurm-cpus-per-task 14` fails sbatch on the 2-CPU test-env nodes (memory `slurm_test_env_sbatch_flags.md`). | LMU uses the framework default `14`. Do NOT override `--slurm-cpus-per-task` on LMU. |
| `--slurm-partition debug` | Test-env only has the `debug` partition. | LMU uses `--slurm-partition Krater` (see firm table above). |

## SSH topology — why hand-rolled `-R` will silently fail

LMU's gateway has `GatewayPorts no`. A single-port `-R` bind on the public gateway interface does NOT work. The framework opens SSH ProxyJump through the gateway to each compute node (`ssh -J kruppb@remote.cip.ifi.lmu.de kruppb@<rock-node>`) and binds the `-R` reverse tunnel on the compute node's localhost. This is automatic; you do not configure it.

But it means: if you try to hand-roll a "let me just expose the primary on the gateway" topology it WILL silently fail. Compute nodes are not directly reachable; both `-A` (agent forwarding) and `-J` (ProxyJump) are required.

Gateway hostname is always `kruppb@remote.cip.ifi.lmu.de`. Never substitute the per-session FQDN the load balancer happens to land you on (`beryll`, `amazonit`, …) — `hostname -f` on the remote returns the load-balanced name, which is not the framework's expected hostname.

## Pre-flight ritual (mandatory every dispatch)

```bash
# 1. Orphan scan — 15-job cap means stale jobs starve the next dispatch
ssh kruppb@remote.cip.ifi.lmu.de "squeue -u kruppb --format='%i %P %j %T %M %l %R'"

# 2. If anything's there from a prior session that's not actively yours:
#    EITHER scancel specific job IDs (preferred when sharing the cap)
ssh kruppb@remote.cip.ifi.lmu.de "scancel <jobid>"
#    OR clear everything (only if you're sure nothing legitimate is running)
ssh kruppb@remote.cip.ifi.lmu.de "scancel -u kruppb"

# 3. Verify the gateway can reach your authorized key
ssh -i ~/.ssh/id_ed25519 kruppb@remote.cip.ifi.lmu.de true
```

If `ssh-add -L` says "no identities" that is the 1Password agent's design — auth still works via per-prompt approval. Do not pass `--ssh-identity-file` on LMU — that flag is for slurm-test-env's per-instance keypairs (see "Do NOT confuse" table above).

## Watching a running dispatch (60s LOCAL wake-tick + sacct fallback)

The watcher runs a 60-second wake-tick loop. The tick itself is **local** — a plain `sleep 60`, NOT an SSH-poll loop (per-tick SSH would thrash the 1Password agent):

```bash
while true; do date -Iseconds; sleep 60; done
```

On every wake:

1. **Check local progress evidence**:
   - dispatch log file's line count / size grew since last tick (filtered `tail` is fine; do NOT pull raw lines into context)
   - primary's stdout shows new `|P|` state-transition lines or `Completed: N/M` counter advanced
   - output dir (`--output` arg) entry count grew (Phase 1 emits per-binary CSVs; Phase 2 emits `unified_vocab.csv`; Phase 3 emits `memmap/<binary>/...`)

   Useful filter for the dispatch log (do NOT tail raw into context):

   ```bash
   tail -f /tmp/dispatch-*.log | grep -E --line-buffered \
     'Phase [0-9]|Job submitted|Secondary connected|Worker [0-9]+|TaskCompleted|TaskFailed|NonRecoverable|Traceback|connection refused|Completed:'
   ```
2. **If progress evidence is present**: continue watching.
3. **If no progress evidence**: ONE SSH check (not a poll loop) — `ssh kruppb@remote.cip.ifi.lmu.de "sacct -u kruppb --starttime <test-start-iso> --format=JobID,State,Elapsed,ExitCode -P"`. Compare against last tick's sacct output.
4. **Terminal failure criterion**: sacct shows every secondary's job + its `.batch` step in a terminal state (`FAILED`, `CANCELLED`, `TIMEOUT`, `COMPLETED`) AND no progress evidence was ever observed → the run is dead. Collect gateway log paths, sacct output, last observed local-log lines; proceed to cleanup.
5. **Stuck-but-still-running**: sacct shows `RUNNING` but the job hasn't advanced past mesh-formation for >10 ticks (~10 min) — same as terminal for decision purposes.

## Cleanup after a dispatch (success OR failure)

The cleanup steps below run on the parent side (or on the subagent only with explicit parent ack). Run all four; skipping any leaves stale state that breaks the next dispatch:

```bash
# 1. Orphan-scan the gateway — must see only OUR active jobs or empty
ssh kruppb@remote.cip.ifi.lmu.de "squeue -u kruppb"
# 2. scancel orphans (shared cap; check ownership before nuking)
ssh kruppb@remote.cip.ifi.lmu.de "scancel <jobid>"   # specific
ssh kruppb@remote.cip.ifi.lmu.de "scancel -u kruppb" # all (only if sure)
# 3. Kill local stray SSH reverse-tunnel processes from prior dispatches.
#    Stale tunnels block port allocation and produce confusing
#    "address in use" errors that look like framework bugs.
pkill -f 'ssh.*-J kruppb.*-R'
# 4. Kill local primary process if it did not exit cleanly
pkill -f 'python.*-m dynrunner'
```

Note: `scancel` does NOT propagate to nested podman containers on compute nodes — conmon double-forks to host systemd. Post-`a12f84a` the wrapper spawns a `setsid -f` watchdog that polls `squeue -j $SLURM_JOB_ID` and runs `podman kill` + `rm -f` when the job disappears. **The wrapper-side watchdog is NOT sufficient on LMU Krater** — `ProctrackType=proctrack/cgroup` + `KillWait=30s` reaps the detached `setsid -f` subshell (cgroup walker doesn't care about pgroup escape) before the watchdog's 60s SIGTERM-grace + SIGKILL escalation completes. The 2026-05-17 incident left 14/15 secondaries' conmon processes orphaned in `user.slice`, still writing to NFS 30+ min post-TIMEOUT. Confirmed by greppable `WATCHDOG: ... sending SIGTERM` lines in `slurm_*.out` with NO matching `exited gracefully` / `force-killing` terminal line. Until the framework's shutdown-manager rearchitecture lands (forwarded to `dynrunner-owner` 2026-05-17), if `squeue -u kruppb` shows empty but containers are still alive on compute nodes, follow the "Manual orphan-conmon recovery" recipe below before any next dispatch.

### Manual orphan-conmon recovery (when wrapper watchdog failed)

Symptoms: `squeue -u kruppb` empty, `sacct` shows TIMEOUT/FAILED for the recent jobs, but the dispatch log still receives `task complete` events OR CSVs keep landing on `<slurm-root>/out/` after the jobs ended. Diagnostic: per-krater-node check via ProxyJump (read-only):

```bash
for node in krater{01..15}; do
  ssh -J kruppb@remote.cip.ifi.lmu.de "kruppb@${node}.cip.ifi.lmu.de" \
    "pgrep -fa 'conmon.*asm-' | head -1; ls -1d /tmp/asm-* 2>/dev/null"
done
```

A live `conmon --api-version 1 -c <hash> ... -n asm-<8hex>-secondary-N ...` means the orphan is alive on that node. Recovery (per affected node):

```bash
ssh -J kruppb@remote.cip.ifi.lmu.de "kruppb@<node>.cip.ifi.lmu.de" '
  for orphan_storage in /tmp/asm-*/storage; do
    [ -d "$orphan_storage" ] || continue
    orphan_runroot="${orphan_storage%/storage}/run"
    podman --root "$orphan_storage" --runroot "$orphan_runroot" --cgroup-manager=cgroupfs stop -t 10 -a 2>/dev/null
    podman --root "$orphan_storage" --runroot "$orphan_runroot" --cgroup-manager=cgroupfs rm -af 2>/dev/null
  done
  podman unshare rm -rf -- /tmp/asm-* 2>/dev/null || rm -rf -- /tmp/asm-* 2>/dev/null
'
```

The wrapper's own pre-flight scan also catches these on the *next* dispatch's startup — but only the node that the next dispatch happens to land on, so manual recovery is still required for the rest. Keep TIMEOUT off the table (large `--slurm-time-limit`) and reserve `scancel` for genuine aborts.

## Shared `/home/k/kruppb/BIG/slurm/out/` convention

The gateway-side `<slurm-root>/out/` directory is shared between dynrunner consumers (asm-tokenizer + asm-dataset-nix's compiler_suit_runner). The conventions:

- **asm-tokenizer** writes flat under `<slurm-root>/out/` mirroring the source layout of `--source-already-staged` (per commit `8da909d`). Reads from `<slurm-root>/out/dataset/...` as Phase-1 input when asm-dataset-nix has prepared binaries.
- **asm-dataset-nix (compiler_suit_runner)** writes to `<slurm-root>/out/dataset/<binary-name>/<variant-id>/` with sibling `<variant-id>.json` sidecars per variant. Do NOT change this output-path layout without warning — asm-tokenizer reads from it.

If you are running asm-dataset-nix and want to change the `./dataset/<binary>/<variant-id>/` layout, ping `asm-tokenizer` on the claude-comm channel first; their `--source-already-staged` paths reference this layout.

## 15-job cap coordination with asm-dataset-nix

kruppb's 15-parallel-task quota is shared. We coordinate slot ownership via the claude-comm channel between peers `asm-tokenizer` and `asm-dataset-nix`.

Protocol:
1. **Before a large dispatch**, ping the other peer with the planned `--jobs N`, partition, ETA wallclock.
2. **Run `squeue -u kruppb`** before sending — if non-zero entries that are NOT yours, ask the other peer if they own them before scancelling.
3. **On dispatch completion (or abort)**, signal slots-free so the other peer can pick up.
4. **Never `scancel` jobs you don't own** without explicit ack from the other peer.

asm-dataset-nix entry point: `python -m compiler_suit_runner submit` (thin wrapper around dynamic_runner with the same flag discipline as `python -m dynrunner --task <name>` — same `--gateway`, `--slurm-partition Krater`, `--jobs 15`, `--source-already-staged`, etc.).

## dynamic_runner pin policy

The asm-tokenizer flake input `dynamic-runner` is bare (`github:sirati/dynamic-runner`) — `flake.lock` tracks the actual rev. Upgrade with `nix flake update dynamic-runner` and rebuild the image with `nix build --no-link .#dockerImage`.

### Pin history (Tier-3 LMU green markers)

- **`328a78e`** (2026-05-15) — first Tier-3 GREEN end-to-end on LMU Krater `--jobs 15`. Includes the 11-commit fix lineage: sync-walk-aware discovery (`be3e2e9`), args-forwarding through phase chain (`1670e7a`), scale-aware setup-deadline (`ba889cd`), SSH-tunnel stagger + retry on MaxStartups (`d4ad1b7`), chain-gate (`76fe930`), peer-bus ClusterMutation arm (`ad71e83`), peer-repoll on PromotePrimary (`cd729fe`), originator flush rendezvous (`328a78e`). See memory `tier3_green_at_328a78e.md`.
- **`8ecd382`** (post-rebase) — upstream rebased/re-merged the DAG; `328a78e` is no longer a literal ancestor of main, but `8ecd382` is the equivalent merge (same merge title "Merge handoff/fix-runcomplete-writer-flush-race", same tree hash `834f643a036eec15dc315aa955c12e7fb362d345`). Functionally identical.
- **`2552f7c`** (2026-05-15) — current main tip at last check. Beyond `8ecd382` it adds: PyO3 codec migration (`2a31304`), secondary subprocess lifecycle migration (`365b649`), PodmanExecWorkerFactory migration (`8315a13`), SLURM submit_job + preparation migration (`612cfe3`, `01849ca`), ErrorType::Unfulfillable wire variant (`a581939`). **Not yet validated on LMU end-to-end** (as of 2026-05-15) — asm-dataset-nix is the canary.

### Lineage check rule of thumb

When asked to verify "is pin X on the same lineage as pin Y":

```bash
git -C ~/devel/python/dynamic_runner merge-base --is-ancestor Y X && echo OK || echo "not ancestor — check tree hashes"
git -C ~/devel/python/dynamic_runner show -s --format='%h %s' Y
git -C ~/devel/python/dynamic_runner show -s --format='%h %s' X
# If "not ancestor" — upstream may have rebased. Compare tree hashes to find the equivalent merge:
git -C ~/devel/python/dynamic_runner log --oneline --all | grep -F "<merge title>"
git -C ~/devel/python/dynamic_runner rev-parse Y^{tree}
git -C ~/devel/python/dynamic_runner rev-parse <candidate>^{tree}
```

Same tree hash on the same merge-title = same merge re-applied post-rebase = functionally equivalent.

## What you do NOT do

- You do NOT hand-roll `sbatch` + `ssh kruppb@<rock-node>` + `podman run` for an LMU dispatch. The framework owns all of that. The user has flagged this multiple times across both asm-tokenizer and asm-dataset-nix sessions.
- You do NOT manually create gateway directories. The dispatcher creates `image_bin/`, `out/`, `log/`, `log/run_<ts>/`, `log/run_<ts>/connection_info/` itself.
- You do NOT manually upload the image. It's layered-blob-uploaded from the local nix store; only changed layers re-upload.
- You do NOT manually scp binaries to the gateway when using `--source-already-staged`. The data is already on cluster NFS; the wrapper bind-mounts at `/app/src-network` (RO).
- You do NOT inspect the image with `python -c "import tarfile..."`. If you find yourself reaching for this, stop and re-read the runbook.
- You do NOT substitute the gateway hostname with the load-balanced FQDN `hostname -f` returns on the remote.
- You do NOT `scancel` jobs without first checking who owns the cap (15-job quota is shared).
- You do NOT change the `<slurm-root>/out/dataset/<binary>/<variant-id>/` output layout without coordinating with `asm-tokenizer` first.

## Reference paths

Files in this repo:
- `dynrunner/SLURM_RUNBOOK.md` — canonical dispatch recipe + flag-by-flag rationale + debugging recipes.
- `dynrunner/LMU_OPERATIONS.md` — this document.

Companion repos (sibling clones expected at the same parent dir):
- `~/devel/python/dynamic_runner/slurm-test-env/README.md` — slurm-test-env flake-app docs (45 lines). `nix run .#{up,down,smoke-test,provision-user,reboot-node}` from that directory. `INSTANCE_ID=<tag> SSH_PORT=<port>` scopes each instance.
- `~/devel/python/dynamic_runner/` — the framework source; check `git log --oneline -- python/` for recent Python-surface changes when bumping the flake input.

## When this document is wrong

Same protocol as `SLURM_RUNBOOK.md`: if a dispatch contradicts what's documented here, the bug is either (1) framework regression (escalate to `dynrunner-owner` peer), (2) recipe drift in this file (update alongside the bumped commit), or (3) LMU cluster-side change (re-derive the gateway layout and update).

In all three cases the fix is durable: update this file or upstream so the next person does not repeat the diagnosis.
