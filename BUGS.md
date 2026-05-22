# asm-tokenizer bugs surfaced 2026-05-08 dispatch

Found while running `dynamic_runner_tokenizer` dispatch against the `slurm-test-env` tokenizer instance per plan `i-have-entered-plan-buzzing-giraffe.md`. Scope: bugs in **this repo**. Framework / dispatch-recipe issues are out of scope (tracked via dynrunner-owner peer).

## 1. Missing `nix/extract-layer-assignment.py` — FIXED 2026-05-12

Resolution:
- Framework: dynamic-runner `0d1b6b7` added `PodmanPackaging(layer_extractor_script_path=...)` ctor arg + `DYNRUNNER_LAYER_EXTRACTOR_SCRIPT` env-var fallback, before falling back to the legacy `<root>/nix/extract-layer-assignment.py` path.
- Consumer: asm-tokenizer's `flake.nix` `shellHook` now exports `DYNRUNNER_LAYER_EXTRACTOR_SCRIPT=${nix-docker-layered-image.packages.${system}.extract-layer-assignment}/bin/extract-layer-assignment`. Verified the env var lands in `nix develop` and points at the executable script.
- Acceptance: cold-rebuild dispatch no longer emits `Layer extractor not found at ...; skipping cache refresh`; layer-cache metadata refreshes incrementally.

Original report retained below for traceability.

---

Primary log on dispatch:

```
Layer extractor not found at .../nix/extract-layer-assignment.py; skipping cache refresh. Next build will be cold.
```

- **Impact**: layer-cache metadata cannot refresh; first post-image-rebuild dispatch re-uploads the entire layered tarball (~1.5 GB) instead of incrementally. Harmless when cache hit; painful on cold rebuild.
- **Actual location**: NOT in asm-tokenizer. The script lives in the upstream `nix-docker-layered-image` flake input at `pkgs/extract-layer-assignment/extract-layer-assignment.py`, exposed via `packages.${system}.extract-layer-assignment` + the default overlay.
- **Source of the warning**: `dynamic_runner/python/dynamic_runner/packaging/podman.py:62` hardcodes `LAYER_EXTRACTOR_SCRIPT_REL = Path("nix") / "extract-layer-assignment.py"`, then warns at podman.py:207 when the path doesn't exist. The hardcoded path predates the 2026-04-29 `nix-docker-layered-image` split.
- **Framework-side acknowledgement**: `dynamic_runner/python/dynamic_runner/tests/test_podman_partial_build.py:117` has a comment "the real extract-layer-assignment.py now lives [elsewhere]" — the move is known, the consumer lookup hasn't been updated.
- **Fix direction (framework)**: have the framework discover the extractor via the consumer's nixpkgs overlay or by calling `nix eval` on the known flake input, rather than via a relative path. Reported to dynrunner-owner 2026-05-12.
- **Stopgap (asm-tokenizer)** if user directs: symlink `nix/extract-layer-assignment.py` → upstream's location. Not yet applied.

## 2. Discovery filter doesn't exclude Ghidra project artifacts — FIXED 2026-05-10

`--name-regex minigzipsh` matched paths like:

```
src/zlib/x64-gcc-5-O3_minigzip_ghidra/x64-gcc-5-O3_minigzip_ghidra.lock
```

These are Ghidra project lock files / `.gpr` / `.rep` artefacts left in `src/` by a prior tokenizer run. Not ELF binaries; should never enter the StageFile queue.

- **Impact**: queue inflation + worker errors. Confirmed live 2026-05-10 via the `--multi-computer local` smoke at dynamic_runner 41c3e16: angr blew up on `x64-gcc-5-O3_minigzip_ghidra/x64-gcc-5-O3_minigzip_ghidra.gpr` and `*.lock~` with `cle.errors.CLECompatibilityError: Unable to find a loader backend`.
- **Fix**: `dynrunner/tokenize/tokenizer_task.py` adds `_is_ghidra_workspace_artifact(path)` and applies it in both `_iter_local_pairs` (local walk) and `visit()` (gateway/SSH walk). Filters paths inside any `*_ghidra/` directory plus known sidecar extensions (`.gpr`, `.rep`, `.lock`, `.lock~`, `.bak~`, `.prp`).
- **Acceptance**: discovery on `src/zlib` (which still contains 2 polluting `_ghidra/` dirs from prior runs) now reports `0` Ghidra-artifact items in the discovered set.

## 3. Discovery N× over-count vs `find -name` — FIXED 2026-05-11

`--name-regex minigzipsh` on the local-mode pathway discovered **940 items** while `find src/zlib -path '*minigzipsh*' -type f -not -path '*_ghidra/*'` returns **235**. The full BIG corpus reported 3103 queued vs find's ~236 — over an order of magnitude inflation. `dataset/busybox/*.tar.zst` paths also surfaced as items.

- **Cause**: `TokenizerTask._iter_local_pairs` uses `tokenizer.binary_discovery.walk_dataset` which calls `VariantInfo.from_legacy_filename` to parse filenames. That parse is independent of the framework's name-regex (which lives in `SelectionFilters.binary_format`). Post-parse filtering in `_variant_passes_filters` checked platform/compiler/version/opt but not the binary-name slot. So `--name-regex minigzipsh` only filtered the SLURM `find_items` pathway (which goes through `match_filename`); the local pathway admitted every parseable name.
- **Fix**:
  - `dynrunner/binary_selection/binary_selector.py`: added `name_pattern: re.Pattern | None` to `SelectionFilters`, populated by `compile_selection_filters` from `config.name_regex`.
  - `dynrunner/tokenize/tokenizer_task.py`: `_variant_passes_filters` now also checks `filters.name_pattern.search(variant.pkg)`.
- **Acceptance** (passing now): `--name-regex minigzipsh` on `src/zlib` returns 235 items, matching `find src/zlib -path '*minigzipsh*' -type f -not -path '*_ghidra/*' | wc -l`.

## 3a. Ghidra concurrent-project LockException — FIXED 2026-05-11

Surfaced in the 2bf8410 Tier-1 local-mode smoke (1 of 940 tasks failed):

```
ghidra.framework.store.LockException: Unable to lock project!
/tmp/ghidra-projects/secondary-secondary-1-25101-src/mips64-clang-3.5-O0_minigzip_ghidra/...
```

- **Cause**: `GhidraDisassemblyProvider.__init__` keyed `project_location` on `binary_path.parent.name`. Two workers in the same secondary processing binaries under the same parent dir (e.g. all of `zlib/`) end up at the same `project_location`. On retry or concurrent open, pyghidra/Ghidra takes a directory-level lock that collides → `LockException`, task fails.
- **Fix**: `tokenizer/disasm/ghidra_provider.py` now keys `project_location` on `{parent.name}-{os.getpid()}` so each worker process gets an isolated workspace dir. No more cross-worker contention.
- **Acceptance**: a 940-task local-mode smoke (`--multi-computer local --jobs 2 --cores 2`) should report `failed=0` (not `failed=1`).

## 5. `gateway_url` kwarg lingered in build_memmap + unify_vocab `_native.find_items` calls — FIXED 2026-05-13

The 2026-05-13 native-task-discovery migration updated `TokenizerTask` to call the new 2-arg `_native.find_items(visitor, root)` signature, but the analogous helpers in `dynrunner/build_memmap/memmap_builder_task.py` and `dynrunner/unify_vocab/vocab_unifier_task.py` were missed — both continued to pass `gateway_url=...` as a keyword. dynamic_runner 0.4.0 dropped that kwarg, so calling `_native.find_items(visitor, root, gateway_url=None)` raised `TypeError: find_items() got an unexpected keyword argument 'gateway_url'`.

- **Symptom**: under chained `--task all` on SLURM, Phase 1 (tokenize) ran GREEN. Phase 2 (unify-vocab) and Phase 3 (build-memmap) setup-secondary calls to `task.discover_items` raised the TypeError during the setup-promote step; the secondary container exited with code 0 (false success), local dispatcher reported `P|Completed: 0 Failed: 0 Stranded: 0`, chained to the next phase, which then also failed identically. Net: no Phase 2/Phase 3 outputs, dispatcher reports clean exit. Reproduced 2026-05-13 09:46.
- **Fix**: dropped `gateway_url` parameter from `_walk_with_filters` and `_collect_existing_output_filenames` helpers in both files; dropped the `args.source_already_staged` branching in `discover_items` — now uses the framework-passed `source_dir` directly (mirrors TokenizerTask). `build_worker_command_args`'s `args.source_already_staged` check stays — it gates `--vocab-source` forwarding semantics.
- **Acceptance**: chained `--task all` on slurm-test-env produces both `unified_vocab.csv` (Phase 2) and `memmap/<binary>_index.bin` (Phase 3) for the `--name-regex minigzipsh --platform x86 x64 arm32 arm64 mips32 mips64` recipe. Pending re-run after the current dispatch finishes its SLURM-reap loop.

Note: the SAME framework-side concern from #N (cosmetic dispatcher accounting bug) compounds with this — a consumer-side TypeError during setup-promote registers as "Container exited 0 + dispatcher reports 0/0/0," indistinguishable from "no tasks discovered." Reported framework-side ask (non-zero exit on discover_items raise + abnormal-end distinct from empty-corpus in accounting) to dynrunner-owner 2026-05-13.

## 4. Duplicated `find_matching_binaries` / selection helpers — RESOLVED 2026-05-13 (refactor-superseded)

Framework's native-task-discovery refactor (`6f583fc` → `2f4aba0`) removes the dispatcher-side `_native.find_items` SSH-walker entirely. Discovery now ALWAYS runs against a local filesystem (either submitter-side for local sources, or on a promoted "setup-secondary" in the cluster bind-mounted to the gateway-staged corpus). The asymmetry between framework-side `find_items` and consumer-side `walk_dataset` disappears at the architectural level.

Consumer-side cleanup landed: deleted `TokenizerTask._iter_gateway_pairs`, `TokenizerTask.visit`, the `--source-already-staged` branch in `_iter_filtered_pairs` (then `_iter_filtered_pairs` itself, now redundant), `match_filename` import. `_collect_existing_output_filenames` updated to new `_native.find_items(task_definition, root)` two-arg signature. Verified: discover_items still returns 235 items on `src/zlib` for `--name-regex minigzipsh`, 0 non-matching, 0 Ghidra artifacts.

Original report retained below for traceability.

---

### Original report (2026-05-09)

Two parallel implementations of the same asm-binary corpus-walk +
filename-parse logic exist in this repo:

- `shared/binary_selector.py` (returns `list[BinaryInfo]`,
  asm-tokenizer's older internal type) — imported by
  `tokenizer/{memmap_validation,vocab_unifier,memmap_builder}/__main__.py`.
- `dynrunner/binary_selection/binary_selector.py` (returns
  `list[TaskInfo]`, the framework-canonical type) — imported by
  `dynrunner/{tokenize,unify_vocab,build_memmap}/` task modules and
  now by `test/multi_secondary/podman_orchestrator.py` (post the
  2026-05-09 framework refactor that removed
  `dynamic_runner._shared.find_matching_binaries`).

Both packages also re-vendor `process_selection_arguments`,
`SelectionConfig`, `add_*_selection_arguments`, `print_selection_summary`
with subtly different signatures (e.g. `shared.SelectionConfig` has
`platforms: list[str]`; `dynrunner.binary_selection.SelectionConfig`
allows `list[str] | None`).

- **Impact**: maintenance hazard — any fix to discovery (e.g. Bug #2
  Ghidra filtering, Bug #3 over-count) has to be applied twice and
  the two will inevitably drift. CLAUDE.md flags this exact
  antipattern: *"the same import appearing in N≥2 files at the same
  call site level"*.
- **Origin**: `dynrunner/binary_selection/` was vendored out of
  dynamic_runner at commit `6c65bb7` (per its own docstring) when
  the framework decided not to own asm-binary corpus opinion;
  `shared/` predates the vendoring and was never consolidated.
- **Fix direction**: pick one canonical module
  (`dynrunner/binary_selection/` is the more recent, framework-typed
  one) and have `shared/` either (a) re-export from it, or (b) be
  deleted with worker imports updated to
  `from dynrunner.binary_selection import …`. Type alignment
  (`BinaryInfo` → `TaskInfo` or vice versa) needs a careful audit
  of all worker call sites — the two types may differ on more than
  the name.
- **Acceptance**: a single `find_matching_binaries` definition in
  the repo; `grep -rn 'def find_matching_binaries'` returns one hit.

---

## Out of scope (framework / env, not this repo)

For traceability only — these were surfaced in the same dispatch and reported through `dynrunner-owner` and the test-env owner peer:

- **`dynamic_runner` `job_manager.py:upload_source_binaries:92-103`** — Bug-B Python twin: relative `binary.path` resolves against CWD instead of `src_root`, so every binary is skipped (`0/3103 uploaded`). Pre-staged patch on `handoff/upload-relative-paths`. (Not asm-tokenizer code.)
- **`dynamic_runner` `ssh_gateway.py:275-277`** — dead-comment cleanup. (Not asm-tokenizer code.)
- **`dynamic_runner` `ssh_gateway.py:96`** — error message could hint at `--ssh-config`. (Not asm-tokenizer code.)
- **`dynamic_runner` `gateway.disconnect()`** — cleanup race; master socket gone before `-O exit`. Cosmetic. (Not asm-tokenizer code.)
- **`dynamic_runner` `slurm_config.py:16`** — `cpus_per_task=14` LMU-shaped default (debatable). (Not asm-tokenizer code.)
- **Recipe drift**: stale `known_hosts` for `[localhost]:2223` on test-env re-up. SLURM_RUNBOOK or `--ssh-config` recipe entry would prevent re-tripping.
- **`dynamic_runner` wrapper script lacks a SIGTERM-trap that calls `podman kill <container_name>`** — confirms and *extends* runbook line 122. Not just "scancel doesn't propagate to nested podman": conmon's double-fork detaches the nested container so its PPID becomes host systemd (PID 11137 in our run), and neither the SLURM job's pidtree nor the dispatcher's pidtree owns it any more. Net effect: `scancel` + dispatcher SIGTERM both succeed, but `eloquent_solomon` (id `9952df14...`, storage `/tmp/asm-46197652/`) and its worker python (PID 28204) survived. Fixed at dynamic_runner `a12f84a` (now in main `733559c`): detached watchdog polls squeue + `podman kill`/`rm` on job-gone.
- **`dynamic_runner` wrapper relay-loop CPU-spin after EXIT trap (`job_manager.py:305-340`)** — `while true; do read -r CMD < "$cmd_socket"; ...; done &` has no exit condition. Under normal operation `read` blocks on the FIFO (kernel-event-driven). When the EXIT trap removes `$RNDTMP` without killing `$CMD_RELAY_PID`, the redirect fails ENOENT every iteration → ~16 K busy-spin/sec → ~1.4 GB/h of identical "cmd.sock: No such file or directory" lines to SLURM stderr. Reproduced independently on tokenizer instance (13 GB) and ds-test instance (186 GB). Fixed at dynamic_runner `90ba235` (now in main): cleanup trap kills `$CMD_RELAY_PID` before `rm -rf`; loop guarded with `while [ -p "$cmd_socket" ]` for defense-in-depth.
- **`dynamic_runner` image-load failure was diagnostically silent** — wrapper aborted on non-zero podman-load exit but the .out file showed no error marker; symptom was a 10-min "0/1 sent SecondaryWelcome" timeout on the primary with no upstream signal of *why*. Fixed at dynamic_runner `733559c`: load wrapped in `if ! <load>; then echo ERROR; exit 1; fi`. Root cause of *why* podman load failed (storage / layer corruption / version mismatch) remains consumer-side investigation when next triggered.
- **`dynamic_runner` timeline race — primary listener binds AFTER 3103-file upload, secondary's 9.5-min connect-retry expires first** (run-3 finding, 2026-05-08). Sequence: SLURM job submitted at upload start → secondary boots ~7 min later → secondary retries 589× over 9.5 min → secondary gives up at 17:19:43 with `failed to connect to primary` → primary finally binds at 17:24:09 (5 min after secondary already exited code=0) → primary stuck in `waiting for 1 secondaries` forever. Fix candidate: bind primary listener BEFORE `sbatch` submission (listener-port is cheap to hold during upload). Pending dynrunner-owner triage.
- **`dynamic_runner` wrapper cleanup logs "Cleaning up temporary directory: $RNDTMP" but doesn't actually `rm -rf`** (run-3 finding). slurm_8.out final lines confirm the echo fired; slurm-worker2's `/tmp/asm-b2c31a9b` still exists at 3.2 GB after secondary clean-exit. Each non-cleaned-properly run leaks ~3.2 GB on `${WORKER_TMP_BASE}`. Pending dynrunner-owner investigation (rm-rf may be missing, or failing silently on 0700 subdirs).
- **slurm-test-env — slurm-worker1 marked DOWN+NOT_RESPONDING and not auto-recovered** after a `podman restart`. slurmd inside reports `active (running)` but slurmctld won't re-include it without manual `scontrol update NodeName=slurm-worker1 State=RESUME`. Test-env-side; not asm-tokenizer code, not dynrunner code. Tracked.
- **`dynamic_runner` dispatcher-accounting display empty on demoted-primary observer post-promotion** (2026-05-13 finding at 54d665d). Phase-1 tokenize green cluster-side: setup-secondary discovery refactor hydrates the pool with 235 tasks, primary continues independently, all 235 CSVs land on gateway under `/home/kruppb/slurm/out/`, RunComplete signal broadcast. BUT dispatcher's local-primary observer exits at the moment of promotion (~06:47:51) and reports `total=0 succeeded=0` because TaskAdded/TaskComplete broadcasts from the now-authoritative setup-secondary never reach it. Dispatcher exit code is 0 (clean) so chained `--task all` proceeds correctly. **User-visible alarm-fatigue trap**: `DYNRUNNER_EXIT=0 + P|Completed: 0 + 235 actual CSVs`. dynrunner-owner confirmed root cause is the legacy `PrimaryTransport` / `SecondaryTransport` not falling back to `PeerTransport` after demotion; unification refactor is on roadmap (~20-30 call sites). Interim: trust output-file counts, not stdout accounting.

---

## Probe campaign results (2026-05-08)

Per plan `i-have-entered-plan-buzzing-giraffe.md` verification #2: each fast-tier probe ran at least once. Pass / fail / "test-env doesn't simulate" recorded.

### Phase 0 — Sanity (against `tokenizer` instance)

| Probe | Description | Result |
|-------|-------------|--------|
| D1 | `sinfo` healthy | ✓ pass — partition `debug*` up, 4 idle workers |
| D6 | `sacct -u kruppb` empty (test-env disables accounting) | ✓ pass — "Slurm accounting storage is disabled" |
| C1 | per-worker `/tmp` isolation | ✓ pass — file written on worker1 not visible on worker2 |
| C2 | shared `/home` (write worker1 → read worker3) | ✓ pass |
| C3 | host-side read of `/home/kruppb` | ✓ pass — uid 109999 (mapped subuid), top-level mode 0755 readable from host uid 1000 |
| F2 | image cache hit on second dispatch | ✓ pass — 50% layer-cache hit on run-2 (only wheel layer re-uploaded after flake.lock bump) |

### Phase 2 / 3 / 4 — fast probes

| Probe | Description | Result |
|-------|-------------|--------|
| B3 | `/dev/shm` size | ✓ pass — 46 GB on all 4 workers (much larger than rootless-podman default 64 MB; inheriting host tmpfs since `--privileged --systemd=always`) |
| D2 | partition-rejection: `sbatch --partition=All` | ✓ pass — "Invalid partition name specified" |
| D3 | short time-limit kill (`--time=1`) | ⚠ partial — submitted but parent flow interrupted before captured trap output |
| D4 | `scancel` propagation to nested podman | ✗ **fails as documented** + extension — confirms runbook line 122; nested-container conmon double-fork survives both scancel and dispatcher SIGTERM. Surface bug catalogued; fix landed at dynrunner `a12f84a`/main `733559c` (watchdog) |
| D5 | multi-node srun | ✓ pass — `srun -N4 hostname` returns all 4 |
| E1 | worker→gateway DNS + `slurm-gateway:6817` reach | ✓ pass — resolves 10.89.0.2, slurmctld reachable |
| E2 | worker→worker DNS resolution | ✓ pass — workers 2/3/4 all resolve from worker1 |
| E3 | serial MaxAuthTries (10 sequential ssh) | ✓ pass — 0/10 fails |
| F1 | sha256 marker present in image_bin | ✓ pass — `771ea4a530afa6c1ddb16cd2b81920f185c9471e7fd8dfd654290b4fcf3fc3ee` |
| G1 | re-run `provision-user.sh` (idempotent) | ✓ pass — "Found existing /home/kruppb with cluster_uid=10000; reusing." +0 new keys |
| G2 | authorized_keys line count stable | ✓ pass — 1 line before, 1 line after re-provision |

### Real-cluster gaps (I1-I7)

Documented in `dynrunner/SLURM_RUNBOOK.md` under "## slurm-test-env coverage gaps (vs LMU CIP)". Listed for traceability — these are gaps, not failures: the test-env cannot validate them by construction. Confirmation must happen on actual LMU CIP.

### Probes not run (deep-tier or out-of-scope)

- A1-A9 — worker2 sd-bus mystery; **dropped** per user direction (self-inflicted from prior session's process kills, not a real bug)
- B1 (pids-limit ceiling), B2 (memory cap), B4 (drop `--privileged`), B5 (concurrent `up.sh`), C4 (atime divergence), E4 (MaxStartups), E5 (IPv6), E6 (port collision), F3 (RO source-file re-upload), F4 (image staleness), G3 (UID conflict), H1-H4 (cleanup / `down.sh` hygiene)
- These are deep-tier or destructive; not blocking for the discovery campaign. Worth scheduling as follow-up if a specific bug class needs verification.

## Probe campaign results (2026-05-18 — Pillar C follow-up on consts-smoke)

Plan `immutable-whistling-twilight.md` Pillar C fast-tier probes. Cluster: `consts-smoke` (port 2226), 4 idle workers, partition `debug*`, RealMemory=3500, CPUs=2 per node.

| Probe | Description | Result |
|-------|-------------|--------|
| C-B1 | pids-limit ceiling | ✓ characterized — cgroup `pids.max` = **32768**, shell `ulimit -u` = 376190. Effective limit is 32768 (well above any plausible asm-tokenizer worker fan-out; --cores 2 × --jobs N never reaches this). No fix needed. |
| C-B2 | `--mem` math vs RealMemory=3500 | ✓ characterized — `--mem=3700` rejected pre-flight: `Memory specification can not be satisfied / Requested node configuration is not available`. `--mem=3500` accepted; cgroup honors request (3.4 GB bytearray inside `--mem=3500` succeeds). Hard ceiling enforced by slurmctld, not by cgroup OOM. Framework dispatches must request ≤3500 on slurm-test-env (LMU Krater has different RealMemory; same probe shape applies). |
| B7 | wrapper-internal cleanup leaves no residue post-c8536bc | ✓ confirmed on validation dispatch; residue inventory (6 dirs/worker, sizes 28K-4.7G, mtimes 08:14-10:43 UTC) is **pre-c8536bc iteration history**, not post-fix leakage. c8536bc-validated dispatch (forced TIMEOUT, 24 binaries) cleaned its own prefix. Legacy residue is one-time cost of cleanup-arc development; cleanup requires explicit user authorization (mass-rm on shared SLURM workers exceeds session-autonomy scope). |
| D2 | corpus shape baseline (Ghidra artifact filter) | ✓ baseline recorded — sidecar-corpus `minigzipsh` shape: **264 files raw**, **235 binaries post-filter** (29 ghidra workspace artifacts `.lock`/`.gpr`/`.rep`/`_ghidra/*` excluded). Green criterion for a future dispatch: queued count = 235. |
| Tier-0-imports | framework consumer surface | ✓ green at c8536bc — framework imports (`TaskDeploymentSpec`, `Task`, `TaskInfo`, `PodmanPackaging`, etc.) load cleanly; `nix flake check` passes; all 12 SLURM_RUNBOOK flags present in `--help`. **Plan-update note**: plan `immutable-whistling-twilight.md` Tier 0 imports `find_matching_binaries` from `dynamic_runner._shared`; this symbol was vendored out to asm-tokenizer's `shared/binary_selector.py` in 6c65bb7. Adjust plan if re-used. |
