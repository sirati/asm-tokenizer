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

## 3. `nix/wheel.nix` `cargoDeps.hash` is stale post-v0.1.1

**Where:** `nix/wheel.nix:27`.

**Current behaviour:** the hash was pinned at v0.1.1 build time. The
phases/types/affinity work added new Rust dependencies (`serde_json`,
`thiserror 2.0.18`, etc.); the recorded hash no longer matches the
vendored Cargo deps. `nix develop` / `nix build .#dockerImage` from
asm-tokenizer fails with `hash mismatch in fixed-output derivation`.

**Suggested fix:** rebuild against the current `Cargo.lock` and pin
the new hash. `lib.fakeHash` placeholder + one failing build to read
back the correct value is the canonical recipe. asm-tokenizer's
flake already does this for its own pin — the same flow applies
upstream.
