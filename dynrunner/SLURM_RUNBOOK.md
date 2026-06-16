# SLURM dispatch runbook

Recipe for running `dynrunner.tokenize` (and `dynrunner.build_memmap`) end-to-end on the LMU CIP SLURM cluster, plus what to do when something breaks.

The intended audience is a fresh subagent or a future-you returning to this after a few weeks. **Follow the commands verbatim.** Do not re-explore the codebase to "figure out" the dispatch flags; this document is the source of truth, and if it disagrees with the framework that's a bug to file (see "When the runbook is wrong" at the end).

## Framework pin + watch protocol

> `flake.lock` pins the `dynamic-runner` flake input; the recipe below reflects the pinned head. Bump with `nix flake update dynamic-runner` — the dispatch rebuilds + uploads the image automatically.
>
> - **Required watch protocol (every dynrunner run):** pass **`--important-stdio-only`** (only wake-worthy events hit stdout; the verbose log goes to the `--full-log-file`/`--full-log-dir` targets; a status summary lands on a ~10-min cadence). Because stdout is sparse, **run the dispatch itself, BARE, as the Monitor's command** — its stdout IS the event stream. Wrapping it in anything is forbidden; see "Run the dispatch bare" under "The dispatch command" + the forbidden anti-patterns in "Watching a running dispatch".
> - **Topology:** under `--source-already-staged` the submitter promotes a setup-secondary to primary and demotes itself to observer — you'll see `primary changed primary=secondary-0`, after which **the submitter exits `rc=0` before cluster-side tokenization finishes** and its stdout goes quiet (post-promotion narration moves to the relocated primary → `--log-dir/secondary-0/primary.log`). **Confirm completion by the gateway output files, not the submitter's exit.**

## Prerequisites

- Working `nix develop` shell from `/home/sirati/devel/python/asm-tokenizer`. The dispatch runs DIRECTLY via `nix develop --command python -m dynrunner …` — **never** wrapped (no `bash -c "…"`, no `cd`, no env-var prefix, no pipe/`tee`, no `timeout`). See "Run the dispatch bare — wrappers are forbidden" under "The dispatch command".
- `~/.ssh/config` has the `lmu` host alias (or just use `kruppb@remote.cip.ifi.lmu.de` directly). 1Password SSH agent must be unlocked — if the gateway returns `signing failed for ED25519 "LMU CIP SSH Key" from agent: communication with agent failed`, the agent is locked and only the user can unlock it.
- Image is rebuilt locally (`nix path-info .#dockerImage` → store path of the tar.gz). The runner uploads it via layered-blob transfer; only changed layers re-upload, so this is fast on iteration.

## The dispatch command

> ### Run the dispatch BARE — wrappers are forbidden
>
> The dispatch is `nix develop --command python -m dynrunner …`, run **DIRECTLY**. When watching it, that exact invocation **IS** the Monitor's command (see "Watching a running dispatch"). NEVER wrap the dispatch in anything:
> - ❌ `bash -c "…"` / `sh -c`, or a `cd &&` that chains/wraps logic. **EXCEPTION:** a single leading `cd <the-correct-clean-worktree> &&` used PURELY to set the dispatch's cwd is permitted — and is REQUIRED when the running agent's shell cwd defaults to a different or dirty worktree, so the image builds from the intended clean tree (e.g. a vanish-resistant worktree while the agent's cwd resets to a dirty primary). It is cwd-only: it does NOT pipe, filter, buffer, env-prefix, redirect, or `timeout` the `--important-stdio-only` stdout event stream this rule protects, so one line still = one event. Nothing else may wrap the dispatch.
> - ❌ an env-var prefix of ANY kind (e.g. `DYNRUNNER_*=…`, `PYTHONSAFEPATH=1`, `PYTHONPATH=… python -m dynrunner …`). The logging env vars are **removed** (use `--full-log-file` / `--full-log-dir`); and the canonical command is `nix develop --command python -m dynrunner …` run from the worktree cwd, which already provides the deps + puts the project on `sys.path` — `PYTHONSAFEPATH`/`PYTHONPATH` are NOT needed and `PYTHONSAFEPATH=1` actively BREAKS `-m dynrunner` resolution by stripping cwd.
> - ❌ `| grep` / `| sed` / `| awk` / `| rg` / `--line-buffered` / **ANY text-filter or pipe** on the dispatch stdout — AND none on a `--full-log-file` used as the event source either. **`--important-stdio-only` IS the filter.** A `grep` on top (e.g. `| grep -E "promot|secondary|primary|Traceback|error|complete|…"`) SILENTLY DROPS wake events whose wording you didn't predict — the run can finish, promote, or fail and the Monitor never surfaces it. The Monitor's command is the BARE dispatch; its `--important-stdio-only` stdout flows UNFILTERED, one line = one event.
> - ❌ `| tee`, any other pipe, or `>`/`2>&1` redirect
> - ❌ `timeout …`
> - ❌ backgrounding (`&`) + a separate `tail -f`/`grep` Monitor on the log, or any `while/until … sleep` babysitter / single-fire waiter loop
>
> Run it bare; verify after with a ONE-SHOT check (gateway output files + `ps`/`squeue`), never a timed/polling loop. (NB: the framework's own `dynrunner-slurm-wrapper` binary is an internal component the dispatcher uploads — unrelated to this; "no wrappers" means no scaffolding around YOUR `python -m dynrunner` invocation.)

Canonical small-batch dispatch against cluster-resident data (filtered to `minigzipsh` across all 6 platforms / all compilers / all opts; ~235 binaries on the LMU `Dataset-1/zlib/` corpus). Bare, with `--important-stdio-only` — this whole line is what you put in the Monitor:

```bash
nix develop --command python -m dynrunner --task tokenize \
  --multi-computer slurm \
  --packaging podman \
  --gateway ssh://kruppb@remote.cip.ifi.lmu.de \
  --slurm-root-folder /home/k/kruppb/BIG/slurm/tokenizer \
  --source-already-staged /home/k/kruppb/BIG/slurm/corpus-v2/zlib \
  --source /tmp \
  --output ~/.cache/asm-tokenizer-out-slurm \
  --name-regex minigzipsh \
  --platform x86 x64 arm32 arm64 mips32 mips64 \
  --jobs 1 \
  --slurm-time-limit 30 \
  --important-stdio-only
```

### Flag-by-flag rationale (do not omit any)

| Flag | Why |
|------|-----|
| `--multi-computer slurm` | Selects the SLURM dispatch pipeline. Don't use the deprecated `--slurm` flag. |
| `--packaging podman` | Required for SLURM; the cluster's wrapper uses rootless podman, not docker. |
| `--gateway ssh://kruppb@remote.cip.ifi.lmu.de` | Always this hostname; never substitute the per-session FQDN (`beryll`, `amazonit`, …) the load balancer happens to land you on. |
| `--slurm-root-folder /home/k/kruppb/BIG/slurm/tokenizer` | The gateway-side root for asm-tokenizer's `image_bin/`, `out/`, `log/` subfolders. **Per-consumer root** — each dynrunner consumer owns its own subdir under `BIG/slurm/` (asm-tokenizer → `…/tokenizer`; asm-dataset-nix → `…/gen-binary-dataset`) so their `out/` trees never collide. The shared corpus source lives at `BIG/slurm/corpus-v2` — that goes to `--source-already-staged`, NOT here. Absolute path from gateway perspective. |
| `--source-already-staged <gateway-abs>` | The gateway-side directory containing your binaries (e.g. cluster NFS). Discovery walks this remotely via SSH; the wrapper bind-mounts it into each secondary container at `/app/src-network` (read-only); no rsync, no zip-copy, no `StageFile` round-trip via the primary. |
| `--source /tmp` | Vestigial — the framework's local `--source` validation is skipped when `--source-already-staged` is set, but argparse still expects the flag. Pass any existing local dir; `/tmp` is fine. |
| `--output <local cache>` | Where the local primary mirrors completion telemetry. Actual CSV/PKL outputs land on the gateway under `<slurm-root>/out/<file>`. |
| `--name-regex <pattern>` | Filter binaries by basename component (the part after the format-string's last separator). Replaces the older `--debugs` shorthand. |
| `--platform x86 x64 arm32 arm64 mips32 mips64` | Allowlist of architectures matched against the format-string's `platform` field. Omit a platform to drop those binaries. |
| `--jobs N` | Number of SLURM secondaries to spawn. Start at 1 for first validation; bump to 2+ for multi-node fan-out. |
| `--slurm-time-limit 30` | sbatch `--time` in minutes. Short enough that a runaway job auto-terminates. |
| `--slurm-partition <name>` | (Optional) sbatch `--partition`. Defaults to `All` (all 131 rock-named nodes). For LMU CIP, useful values: `AMD` (66 nodes, "Gesteine B"), `NvidiaAll` (25, GPU nodes), `Krater` (40), `Abaki` (4). |
| `--raw-logs` | Bypass log file rewrites. Recommended for ad-hoc dispatches; the wrapper script still tees the secondary's stdout/stderr into `<slurm-root>/log/run_<ts>/slurm_<jobid>.{out,err}`. |
| `--skip-existing` | (Optional) Idempotent re-runs: filtered task-side in `discover_items` against `args.resolved_output_root` (gateway-side `<slurm-root>/<output-subfolder>` in pre-staged mode, local `--output` otherwise). Walks the output dir via the same `_native.find_items` SSH backend the source walk uses, drops any binary whose `get_output_filename_pattern` filename is already present. |

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

Outputs land under `<slurm-root>/out/` (for asm-tokenizer, `BIG/slurm/tokenizer/out/`) mirroring the relative layout under `--source-already-staged`. If you point `--source-already-staged` at `corpus-v2` (one level up) you'll get `<slurm-root>/out/<package>/<file>` (e.g. `tokenizer/out/zlib/<file>_output.csv`); pointing at `corpus-v2/zlib` (one level into a single package) gives flat `<slurm-root>/out/<file>`.

**`--skip-existing` matches on this exact resolved path**, so the source granularity AND the slurm-root must be consistent across runs — otherwise a re-run won't see the prior outputs and will silently re-tokenize the whole corpus. Always use the per-consumer root.

### Legacy: pushing a local-only corpus

If your data lives only on your local machine (rare), the framework's `StageFile` path still works without `--source-already-staged`: pass `--source <local-dir>` and omit `--source-already-staged`; the primary will hash + transfer each binary to the secondaries via `StageFile` notifications during dispatch. Slower (every byte goes over the wire), but works without the data being pre-staged on the cluster. Not recommended for sizes >100 MB total.

## Watching a running dispatch

**The Monitor's command IS the dispatch, run DIRECTLY — nothing wraps it.** Put
the exact bare `nix develop --command python -m dynrunner … --important-stdio-only`
invocation from "The dispatch command" above into the Monitor, verbatim — no
env-var prefix, no `bash -c`/`cd`, no `| tee` or any pipe, no `timeout`, no
surrounding scaffolding (the full forbidden list is in "Run the dispatch bare"
above). `--important-stdio-only` makes stdout carry only wake-worthy events
(`all secondaries connected`, `primary changed`, completion, failures) plus a
~10-min summary; each line is one Monitor event, and the process exits on its
own.

When an event needs context, **`Read` the gateway per-role log**
(`--log-dir/<secondary_id>/primary.log` + `secondary.log`) — never stream a log
as events. To confirm the run produced results: under `--source-already-staged`
the submitter is promoted away and **exits `rc=0` at `primary changed`, before
tokenization finishes on the cluster**, and its final summary/exit is not
authoritative — so do a ONE-SHOT check of the gateway output dir
(`*_output.csv` + `*_meta.json` per binary) plus `ps`/`squeue`. **Never** a
waiter loop, polling Monitor, or `timeout`.

### Forbidden — every one of these is wrong; do not do them

- **Backgrounding the dispatch and pointing a *separate* Monitor at its log**
  (`Bash run_in_background …` + `Monitor 'tail -f out.log'`). Two moving parts
  where one suffices — the Monitor must BE the dispatch.
- **Wrapping the Monitor command in a babysitter loop** — no
  `while true; do squeue/sacct/ps …; sleep N; done`, no `until` loops, no
  `tee`/`echo`-marker/dedup scaffolding around it. The dispatch's
  `--important-stdio-only` stdout already IS the event stream.
- **A `squeue`/`sacct` poll-loop Monitor to "watch teardown."** The promoted
  secondary owns teardown; if you must confirm it, one-shot `squeue` after the
  fact — never a looping Monitor.
- **Omitting `--important-stdio-only`** then monitoring the raw verbose stdout
  — it floods the event stream and the Monitor auto-stops.
- **Filtering the dispatch stdout (or the full-log) through `grep`/`sed`/`awk`/`rg`.**
  e.g. `nix develop --command python -m dynrunner … --important-stdio-only | grep -E "promot|secondary|primary|Traceback|error|complete|…"`. This is the single most insidious violation: `--important-stdio-only` is ALREADY the event filter, and a keyword `grep` on top drops any wake line whose exact wording you failed to anticipate (a new promotion phrase, an unfamiliar failure string), so the run silently finishes/fails behind your filter and the Monitor never wakes you. NO text-filter, on the stdout OR on a `--full-log-file` tailed as the event source. The Monitor runs the BARE dispatch; every `--important-stdio-only` line is an event, unfiltered.
- **Any scaffolding around the bare dispatch** — env-var prefixes (including
  `PYTHONSAFEPATH`/`PYTHONPATH`), `bash -c`/`cd`, `timeout`, `| tee`/`| grep`/any
  pipe, OR a separate single-fire waiter loop to detect teardown. Run the
  dispatch bare (`nix develop --command python -m dynrunner …` from the worktree
  cwd); verify after with a one-shot check, never a timed or polling loop.

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

If a `scancel`'d job ever leaves a container behind on a node, kill it there:

```bash
ssh -A -J kruppb@remote.cip.ifi.lmu.de kruppb@${NODE}.cip.ifi.lmu.de \
  'pgrep -fa "dynrunner.tokenize.*--secondary" | awk "{print \$1}" | xargs -r kill -TERM'
```

## When the runbook is wrong

If the dispatch errors out with a message that contradicts what's documented here, the bug is one of:

1. **Framework regression in `dynamic_runner`** — capture the file:line + the dispatch error → bug report to dynrunner-owner via claude-comm.
2. **Recipe drift here** — flag was renamed, default changed, etc. Update this file alongside the bumped commit.
3. **Cluster-side change** — wrapper script regenerated, BIG paths moved, podman version changed. Re-derive the gateway layout via `ssh kruppb@remote.cip.ifi.lmu.de 'ls ~/BIG/slurm/'` and update.

In all three cases the fix is durable: update the runbook (or upstream) so the next person doesn't repeat the diagnosis.

## slurm-test-env vs LMU CIP — differences worth knowing

`slurm-test-env` is a legitimate SLURM testing environment: it runs real `slurmctld` + `slurmd`, with a per-instance shared `/home` (network-share-shaped) and per-worker `/tmp` (individual scratch), so SLURM-protocol semantics, job submission, scheduling, multi-node srun, and dispatch flow all behave like the production cluster. A green dispatch here is meaningful evidence that the framework's SLURM path works.

The differences below are "things that look the same but aren't quite" — useful for interpreting results, not blockers on declaring green:

- **I1 — Partitions.** Test-env has only `debug`; LMU CIP has `All`, `AMD`, `NvidiaAll`, `Krater`, `Abaki`. Frameworks must pass `--slurm-partition debug` for test-env (LMU's default is `All`).
- **I2 — Scale.** Test-env runs 4 workers (ceiling 16); LMU CIP has 131 nodes. Race conditions or arithmetic that only manifests at scale is unlikely to surface locally.
- **I3 — `/home` backing.** Test-env's `/home` is a host bind-mount; LMU CIP's is NFS. Strong-consistency operations behave the same; NFS-specific failure modes (`ESTALE`, soft-mount recovery, lock-daemon issues) are NFS-only.
- **I4 — SSH topology.** LMU CIP requires `-J kruppb@gateway-fqdn` for compute-node access; test-env exposes workers on a podman network with direct routability. Frameworks that consistently use `-J` work both places; anything that hardcodes direct compute-node ssh would fail on real cluster.
- **I5 — Accounting.** Test-env disables `slurmdbd` (`sacct` returns empty); LMU CIP has it. Frameworks that poll `sacct` need a non-`sacct` fallback (`squeue`) for test-env mode.
- **I6 — Munge key.** Test-env bakes a fixed key at image build; LMU CIP rotates. Anything that caches auth state across rotations is LMU-specific.
- **I7 — SSH key path.** Test-env uses per-instance keypairs via `--ssh-identity-file`; LMU CIP uses a 1Password agent (locked-agent failure modes are LMU-only). Both go through the same `--ssh-config` / `--ssh-identity-file` framework primitives.
- **I8 — Image-load latency.** Test-env podman load is rootless + slow: ~50–90 s per worker, sequential (each reads the same ~1.5 GB tarball from the shared `/home` bind-mount), so secondaries connect later than on LMU CIP. The `unconfigured_deadline_secs` default (600 s) covers it — no override needed.
- **I9 — polkit / linger self-enable.** Both environments run active polkit whose stock policy grants unprivileged `loginctl enable-linger`. A test-env cluster brought up on an older node image fails linger — re-`nix run .#up` to rebuild from the flake. To exercise the polkit-DENY branch on a test-env node, see the parity section in slurm-test-env/README.md.

In practice: iterate locally on `slurm-test-env` (with the I8 override), then do a small-`--jobs 1` confirmation run on LMU CIP before scaling up. The local→cluster delta is small enough that this is a sanity check, not a redo.
