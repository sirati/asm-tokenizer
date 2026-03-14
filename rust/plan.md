# Rust Rewrite Plan for `dynamic_batch`

---

## Phase 8: Nix Build for dynamic_batch_rs Python Package

### Context

The Rust `db_python_provider` crate produces a Python extension module (`dynamic_batch_rs`). Currently it's built via `maturin develop --release` into the `.venv`, but that doesn't integrate with the nix-based dev shell. We need a proper nix build so the module is available as a Python package in the nix environment.

The user wants:
- New nix code mostly in `rust/` to keep it self-contained
- Use `rust-overlay` (oxalica) for the Rust toolchain
- Provide an overlay so the root `flake.nix` can add it to `deploymentPythonPackages`

### Files to Create/Modify

#### 1. `rust/dynamic_batch/package.nix` (NEW)

A function callable from the root flake that builds the Python package:

```nix
{
  lib,
  python3Packages,
  rustPlatform,
  pkg-config,
  openssl,
}:

python3Packages.buildPythonPackage rec {
  pname = "dynamic-batch-rs";
  version = "0.1.0";
  pyproject = false;  # maturin, not pyproject

  src = lib.cleanSource ./.;   # rust/dynamic_batch/ workspace root

  cargoDeps = rustPlatform.fetchCargoVendor {
    inherit src;
    hash = "";  # will fill after first build attempt
  };

  buildAndTestSubdir = "crates/db_python_provider";

  nativeBuildInputs = with rustPlatform; [
    cargoSetupHook
    maturinBuildHook
  ];

  buildInputs = [ openssl ];
  nativeCheckInputs = [ pkg-config ];

  # No Python tests in this crate
  doCheck = false;
}
```

Key points:
- `src` points at the workspace root (`rust/dynamic_batch/`)
- `buildAndTestSubdir` points to the specific crate within the workspace
- `fetchCargoVendor` vendors all workspace dependencies from `Cargo.lock`
- `maturinBuildHook` builds the wheel
- Hash needs to be computed on first build (nix will tell us the correct one)

#### 2. `flake.nix` (MODIFY)

Changes:
- Add `rust-overlay` input
- Apply `rust-overlay.overlays.default` to `pkgs`
- Build a `rustPlatform` from the overlay's stable toolchain (or just use the default one — the overlay replaces `pkgs.rustc`/`pkgs.cargo` so `pkgs.rustPlatform` automatically uses it)
- Import `rust/dynamic_batch/package.nix` and call it with the right args
- Add the resulting package to `deploymentPythonPackages`
- Remove standalone `rustPackages` list items (`rustc`, `cargo`, etc.) since the overlay provides them, but keep `maturin`, `rust-analyzer`, `clippy`, `rustfmt` in `devPackages` or get them from the overlay toolchain with extensions

Concrete changes to `flake.nix`:

```nix
inputs = {
  nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  rust-overlay = {
    url = "github:oxalica/rust-overlay";
    inputs.nixpkgs.follows = "nixpkgs";
  };
  gitignore = { ... };  # unchanged
};

outputs = { self, nixpkgs, rust-overlay, gitignore }:
  let
    # ... existing code ...
  in
  {
    # Add overlay for other flakes to consume
    overlays.default = final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          dynamic-batch-rs = python-final.callPackage ./rust/dynamic_batch/package.nix {
            rustPlatform = final.rustPlatform;  # uses rust-overlay's toolchain
          };
        })
      ];
    };

    devShells = forAllSystems (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
            self.overlays.default
          ];
        };
      in {
        default = pkgs.mkShell {
          packages = [
            (pkgs.python314.withPackages (py:
              (deploymentPythonPackages py) ++
              (devPythonPackages py) ++
              [ py.dynamic-batch-rs ]  # <-- the Rust module
            ))
            # ... rest of packages
          ];
        };
      }
    );
  };
```

Simpler approach — build the package directly, no overlay machinery needed:

```nix
let
  pkgs = import nixpkgs {
    inherit system;
    overlays = [ rust-overlay.overlays.default ];
  };

  rustToolchain = pkgs.rust-bin.stable.latest.default.override {
    extensions = [ "rust-src" "rust-analyzer" "clippy" ];
  };

  rustPlatform = pkgs.makeRustPlatform {
    cargo = rustToolchain;
    rustc = rustToolchain;
  };

  dynamic-batch-rs = pkgs.python314Packages.callPackage ./rust/dynamic_batch/package.nix {
    inherit rustPlatform;
  };
in
{
  devShells.default = pkgs.mkShell {
    packages = [
      (pkgs.python314.withPackages (py:
        (deploymentPythonPackages py) ++ (devPythonPackages py) ++ [ dynamic-batch-rs ]
      ))
      rustToolchain  # provides cargo, rustc, rust-analyzer, clippy
      pkgs.maturin   # for manual maturin develop during iteration
      pkgs.rustfmt
      # ... other dev packages
    ];
  };
}
```

This is the approach we'll use.

### Rust toolchain from overlay

`rust-overlay` does NOT replace `pkgs.rustPlatform` automatically — it only adds `pkgs.rust-bin`. We must explicitly create a `rustPlatform` via `pkgs.makeRustPlatform`:

```nix
rustToolchain = pkgs.rust-bin.stable.latest.default.override {
  extensions = [ "rust-src" "rust-analyzer" "clippy" ];
};

rustPlatform = pkgs.makeRustPlatform {
  cargo = rustToolchain;
  rustc = rustToolchain;
};
```

This `rustPlatform` is then passed to `package.nix` via `callPackage` override. The `package.nix` receives it as an argument.

For the devShell, `rustToolchain` goes into `packages` to provide `cargo`, `rustc`, `rust-analyzer`, `clippy` in one package. This replaces the old `rustPackages` list.

### Hash computation

On first build, `fetchCargoVendor` will fail with a hash mismatch. The error message will include the correct hash. We paste it into `package.nix`.

### Verification

1. `nix develop` — shell opens with `dynamic_batch_rs` importable via `python -c "import dynamic_batch_rs"`
2. `nix build .#packages.x86_64-linux.dockerImage` — docker image includes the Rust module
3. `cargo test --workspace` still works inside `nix develop` (dev tools available)

---

## Phase 7: Make BinaryIdentifier Generic (DONE)

### Context

`BinaryIdentifier` currently hardcodes 5 fields (`binary_name`, `platform`, `compiler`, `version`, `opt_level`) specific to the tokenizer task. The TODO in `rust/todo.md` says "the specific code that runs e.g. binary identifier is user dependent and the code here must be generic over it." Different task definitions should be able to use different identifier structures.

### Approach: Generic type parameter `I` threaded through all types

Replace the concrete `BinaryIdentifier` struct with a generic type parameter `I` on `BinaryInfo<I>`, `FailedTask<I>`, and all types that contain them. The concrete tokenizer-specific struct becomes the *default* / *reference implementation* provided by `db_python_provider`.

**Trait bound alias** (defined in `db_comm_api_base`):
```rust
pub trait Identifier: Clone + Debug + Hash + Eq + Serialize + for<'de> Deserialize<'de> + Send + 'static {}
impl<T> Identifier for T where T: Clone + Debug + Hash + Eq + Serialize + for<'de> Deserialize<'de> + Send + 'static {}
```
(`Send + 'static` needed for `tokio::spawn` in distributed manager.)

### Files Changed (by crate, in dependency order)

#### 1. `db_comm_api_base/src/types.rs`
- Remove concrete `BinaryIdentifier` struct (move to `db_python_provider`)
- Add `Identifier` blanket trait
- `BinaryInfo<I: Identifier>` with `identifier: I`
- `FailedTask<I: Identifier>` with `binary: BinaryInfo<I>`
- Remove accessor methods (`binary_name()`, etc.) — they assumed concrete fields
- Keep `BinaryInfo::path` and `BinaryInfo::size` (universal)

#### 2. `db_comm_api_base/src/lib.rs`
- Re-export `Identifier` trait

#### 3. `db_scheduler_api/src/lib.rs`
- `WorkerBudgetInfo<I: Identifier>` with `current_task: Option<BinaryInfo<I>>`
- `Scheduler` trait methods become generic: `fn assign_initial<I: Identifier>(&self, ..., pending: &[BinaryInfo<I>], ...)`
  - **OR** make `Scheduler` itself generic: `trait Scheduler<I: Identifier>` — simpler since all impls will be for a fixed `I`
- `AssignmentDecision` stays non-generic (uses `usize` indices)
- `OomDecision` stays non-generic (uses `WorkerId`)

#### 4. `db_scheduler_impl/src/lib.rs`
- `impl<I: Identifier> Scheduler<I> for MemoryStealingScheduler` — scheduler never inspects identifier fields, only `binary.size`
- Update test helpers to use a simple test identifier type

#### 5. `db_local_manager/src/worker.rs`
- `WorkerHandle<M, I: Identifier>` with `current_binary: Option<BinaryInfo<I>>`
- `WorkerEvent<I>`: variants carry `BinaryInfo<I>` or `Option<BinaryInfo<I>>`

#### 6. `db_local_manager/src/manager.rs`
- `LocalManager<M, S, E, I>` where `I: Identifier`
- `ProcessingStats` stays non-generic (just counters)
- `process_binaries(&mut self, binaries: Vec<BinaryInfo<I>>, ...)` 
- Internal vecs: `Vec<BinaryInfo<I>>`, `Vec<FailedTask<I>>`

#### 7. `db_primary_secondary_comm/src/messages.rs`
- `DistributedBinaryInfo<I: Identifier>` replaces hardcoded fields with `identifier: I`
  - Keeps `path: String` and `size: u64` (universal)
- `ZipBinaryEntry<I>`, `TaskInfo<I>` become generic
- `DistributedMessage<I: Identifier>` — the enum becomes generic
  - Only ~3 variants use `I` (`InitialAssignment`, `TaskAssignment`, `FullTaskList`), but all must carry the parameter since Rust enums are monomorphic
  - Serde `#[serde(tag = "msg_type")]` works fine with generic enums since it uses the variant name, not the type parameter
- Codec functions become generic: `encode<I: Identifier>(msg: &DistributedMessage<I>)`, `decode<I: Identifier>(data: &[u8])`

#### 8. `db_distributed_manager/src/primary.rs`
- `PrimaryCoordinator<T, S, E, I>` gains `I: Identifier` parameter
- `binary_to_distributed()` becomes trivial: just moves `identifier` field
- `compute_task_hash()` works via `Hash` on the identifier (no field-by-field hashing)
- `RemoteWorkerState<I>` stores `Option<BinaryInfo<I>>`

#### 9. `db_distributed_manager/src/secondary.rs`
- `SecondaryCoordinator<PT, M, S, E, I>` gains `I: Identifier` parameter
- `distributed_to_binary()` becomes trivial: reconstruct `BinaryInfo` from `DistributedBinaryInfo`

#### 10. `db_transport_quic/src/transport.rs`
- `QuicTransport<I>` becomes generic because it serializes/deserializes `DistributedMessage<I>`
- **OR** keep transport non-generic by working with raw bytes, pushing ser/de to the caller
- Preferred: make it generic since it already calls codec encode/decode internally

#### 11. `db_python_provider/src/lib.rs`
- Define the concrete `TokenizerIdentifier` struct here (the current 5-field struct):
  ```rust
  #[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
  pub struct TokenizerIdentifier {
      pub binary_name: String,
      pub platform: String,
      pub compiler: String,
      pub version: String,
      pub opt_level: String,
  }
  ```
- `PyBinaryIdentifier` wraps `TokenizerIdentifier` for Python
- All PyO3 classes instantiate with `I = TokenizerIdentifier`:
  - `PyLocalManager` wraps `LocalManager<..., TokenizerIdentifier>`
  - `PyDistributedManager` wraps coordinators with `I = TokenizerIdentifier`
- `extract_binaries()` returns `Vec<BinaryInfo<TokenizerIdentifier>>`

#### 12. Tests
- Crates that don't need tokenizer-specific fields use a minimal test identifier:
  ```rust
  #[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
  struct TestId(String);
  ```
- Integration tests in `db_local_manager` and `db_distributed_manager` use `TestId`
- `db_python_provider` tests (if any) use `TokenizerIdentifier`

### What does NOT change
- `db_manager_runner_comm` — no identifier awareness (line protocol)
- `db_runner_impl` — no identifier awareness (receives paths as strings)
- `db_transport_channel` — works with `Command`/`Response`, not identifiers
- `db_transport_socket` — works with `Command`/`Response`, not identifiers
- `ErrorType`, `TaskResult`, `Command`, `Response` — identifier-independent

### Key simplification: conversion functions
Before (5 field-by-field copies):
```rust
fn binary_to_distributed(binary: &BinaryInfo) -> DistributedBinaryInfo {
    DistributedBinaryInfo {
        path: binary.path.to_string_lossy().into_owned(),
        size: binary.size,
        binary_name: binary.identifier.binary_name.clone(),
        // ... 4 more fields
    }
}
```
After (generic, zero knowledge of identifier structure):
```rust
fn binary_to_distributed<I: Identifier>(binary: &BinaryInfo<I>) -> DistributedBinaryInfo<I> {
    DistributedBinaryInfo {
        path: binary.path.to_string_lossy().into_owned(),
        size: binary.size,
        identifier: binary.identifier.clone(),
    }
}
```

### Verification
1. `cargo build --workspace` compiles
2. `cargo test --workspace` — all existing tests pass (updated for generic types)
3. `maturin develop --release` + Python import test
4. Re-run the Python integration test (local + distributed with 12 test binaries)
5. Verify wire format compatibility: serialize a `DistributedMessage<TokenizerIdentifier>` and confirm JSON structure has the identifier fields flattened (use `#[serde(flatten)]` on the identifier field)

### Wire format consideration
To maintain backward compatibility with Python coordinators, `DistributedBinaryInfo<I>` should serialize with the identifier fields flattened into the parent JSON object (not nested under an "identifier" key). Use `#[serde(flatten)]`:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedBinaryInfo<I: Identifier> {
    pub path: String,
    pub size: u64,
    #[serde(flatten)]
    pub identifier: I,
}
```
This ensures `{"path": "...", "size": 123, "binary_name": "...", "platform": "...", ...}` — same wire format as before.

---

## Context

The Python `dynamic_batch` module is a distributed batch processing framework for binary tokenization. It suffers from lack of strict typing and boundary enforcement. This plan rewrites the core (manager-side scheduling, memory management, worker lifecycle, distributed coordination) in Rust as a 12-crate Cargo workspace under `rust/dynamic_batch/`, with PyO3 bindings. Python worker subprocesses remain Python — the Rust code replaces the **manager** side.

## Architecture Overview

### Crate Structure

```
rust/dynamic_batch/
  Cargo.toml (workspace)
  crates/
    db_comm_api_base/          # Foundation types + traits
    db_manager_runner_comm/    # Manager↔Runner protocol + ZST state machine + codec
    db_primary_secondary_comm/ # 20+ distributed message types + codec + state machine
    db_scheduler_api/          # Scheduler trait abstraction
    db_scheduler_impl/         # Memory-stealing scheduler (ports decision_impl.py)
    db_transport_socket/       # Unix socketpair + named socket (tokio)
    db_transport_channel/      # In-process mpsc transport (testing)
    db_transport_quic/         # QUIC (quinn) + WSS fallback
    db_runner_impl/            # Transport-agnostic runner main loop
    db_local_manager/          # Local manager (ports LocalWorkerManager)
    db_distributed_manager/    # Primary/secondary coordinators
    db_python_provider/        # PyO3 bindings (ONLY crate that knows Python)
```

### Dependency DAG

```
db_python_provider
├── db_local_manager
│   ├── db_scheduler_impl → db_scheduler_api → db_comm_api_base
│   ├── db_runner_impl → db_manager_runner_comm → db_comm_api_base
│   ├── db_transport_socket → db_manager_runner_comm
│   └── db_transport_channel → db_manager_runner_comm
├── db_distributed_manager
│   ├── db_primary_secondary_comm → db_comm_api_base
│   ├── db_transport_quic → db_primary_secondary_comm
│   ├── db_scheduler_impl
│   └── db_transport_channel (testing)
└── db_comm_api_base
```

No crate except `db_python_provider` depends on PyO3.

## Key Design Decisions

### 1. Manual Typestate Pattern for ZST State Machines
Use `PhantomData<S>` + unit struct states. Zero dependencies, full async support, compile-time enforcement. No proc macro crates needed.

State machines:
- **RunnerProtocol\<S\>**: `Unconnected → WaitingForReady → Idle ↔ Processing → Stopped`
- **SecondaryConnection\<S\>**: `AwaitingWelcome → Handshaking → CertExchanging → PeerDiscovery → InitialAssigning → FileTransferring → Operational → ShuttingDown`
- **ProcessingPipeline\<S\>**: `Initializing → InitialAssignment → MainPhase → RetryPhase → OomPhase → UnassignedPhase → Complete`

Since workers are stored in `Vec` and each may be in a different state, wrap in a runtime enum:
```rust
pub enum RunnerProtocolState {
    Unconnected(RunnerProtocol<Unconnected>),
    WaitingForReady(RunnerProtocol<WaitingForReady>),
    Idle(RunnerProtocol<Idle>),
    Processing(RunnerProtocol<Processing>),
    Stopped(RunnerProtocol<Stopped>),
}
```

### 2. Tokio Current-Thread Runtime
Single-threaded async everywhere. No Send/Sync requirements. Types freely use `Rc`, `RefCell`. Matches Python GIL model for PyO3 bridging.

### 3. Composition Replaces Python Mixin Hierarchy
Python: `DecisionWorkerManMixin + ExecutionWorkerManBaseImpl = LocalWorkerManager`
Rust: `LocalManager` owns a `Box<dyn Scheduler>`. Scheduler trait provides decision logic; manager struct provides execution logic.

### 4. Event-Driven via Channels (not polling)
Replace `sleep(0.1)` polling with `tokio::select!` over:
- `mpsc::UnboundedReceiver<WorkerEvent>` (per-worker background reader tasks)
- `tokio::time::interval` (periodic OOM checks)
- Distributed message channels (if in distributed mode)

### 5. Backward-Compatible Wire Format
The manager↔worker line-delimited text protocol (`"done\n"`, `"error:oom:msg\n"`, etc.) is preserved exactly so existing Python workers work unchanged. The distributed protocol uses length-prefixed JSON matching the existing Python format.

## Critical Files (Python sources to port)

| Rust Crate | Python Source |
|---|---|
| `db_comm_api_base` | `comm/proto/messages.py`, `models.py`, `shared/binary_info.py`, `comm/interface/base_interface.py` |
| `db_manager_runner_comm` | `comm/proto/messages.py` (parse/serialize) |
| `db_scheduler_impl` | `worker_manager/decision_impl.py` (assignment algorithms), `worker_manager/execution_impl.py` (OOM logic), `worker_manager/base.py` (budget formula) |
| `db_local_manager` | `worker_manager/base.py` (5-phase pipeline), `worker_manager/local.py`, `worker/local_worker.py`, `worker_lifecycle.py` |
| `db_distributed_manager` | `worker_manager/authoritative_base.py`, `worker_manager/submissive_base.py`, `multi_computer/protocol.py`, `multi_computer/primary/coordinator.py`, `multi_computer/secondary/coordinator.py` |
| `db_transport_socket` | `comm/interface/unix_socket.py`, `comm/interface/named_socket.py` |
| `db_transport_quic` | `multi_computer/quic_transport.py`, `multi_computer/message_router.py` |

## Key Type Mappings

```rust
// db_comm_api_base
pub struct BinaryIdentifier { binary_name, platform, compiler, version, opt_level: String }
pub struct BinaryInfo { path: PathBuf, size: u64, identifier: BinaryIdentifier }
pub enum ErrorType { OutOfMemory, NonRecoverable, Recoverable }
pub struct TaskResult { success: bool, error_type: Option<ErrorType>, error_message: Option<String>, warnings: u32, filtered: u32 }
pub struct FailedTask { binary: BinaryInfo, error_type: ErrorType, error_message: String, retry_count: u32 }
pub enum Command { Stop, ProcessBinary { relative_path: String } }
pub enum Response { Ready, Done { warnings, filtered }, Error { error_type, message }, PickledError { exception_type, message, traceback }, PhaseUpdate { phase_name }, Keepalive }

// Traits
pub trait CommandSender { async fn send_command(&mut self, cmd: Command) -> Result<(), String>; }
pub trait CommandReceiver { async fn recv_command(&mut self) -> Option<Command>; }
pub trait ResponseSender { async fn send_response(&mut self, resp: Response) -> Result<(), String>; }
pub trait ResponseReceiver { async fn recv_responses(&mut self) -> Vec<Response>; }
pub trait ManagerEndpoint: CommandSender + ResponseReceiver {}
pub trait RunnerEndpoint: CommandReceiver + ResponseSender {}

// db_scheduler_api
pub trait MemoryEstimator { fn estimate_memory(&self, binary_size: u64) -> u64; }
pub trait Scheduler {
    fn assign_initial(...) -> AssignmentDecision;
    fn assign_normal(...) -> AssignmentDecision;
    fn check_oom(...) -> OomDecision;
    fn initial_budget(worker_index: u32, max_memory: u64) -> u64;
}
```

## Nix Updates

Add to `flake.nix` devShell:
```nix
pkgs.rustc pkgs.cargo pkgs.rust-analyzer pkgs.clippy pkgs.rustfmt pkgs.maturin
```
Or use `rust-overlay` for a pinned toolchain. Commands run inside `nix shell` since env won't auto-update.

## Workspace Dependencies (Cargo.toml)

```toml
[workspace.dependencies]
tokio = { version = "1", features = ["rt", "net", "io-util", "sync", "time", "process", "macros"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
thiserror = "2"
pyo3 = { version = "0.23", features = ["extension-module"] }
quinn = "0.11"
rustls = "0.23"
rcgen = "0.13"
tokio-tungstenite = { version = "0.26", features = ["native-tls"] }
nix = { version = "0.29", features = ["socket", "process"] }
```

## Implementation Order

### Phase 1: Foundation (DONE)
1. Update `flake.nix` with Rust toolchain
2. `db_comm_api_base` — types, enums, traits
3. `db_manager_runner_comm` — protocol codec + ZST state machine
4. `db_scheduler_api` — scheduler trait + phase/budget types
5. `db_scheduler_impl` — memory-stealing scheduler (port budget formula, initial/normal assignment, OOM logic)

### Phase 2: Transport (DONE)
6. `db_transport_channel` — mpsc channel pairs for testing
7. `db_transport_socket` — Unix socketpair + named socket via tokio

### Phase 3: Core Manager (DONE)
8. `db_runner_impl` — transport-agnostic runner loop + TaskExecutor trait
9. `db_local_manager` — full local manager with 5-phase pipeline, subprocess workers, event-driven main loop

### Phase 4: Python Bindings (DONE)
10. `db_python_provider` — PyO3 module: `PyLocalManager`, `PyBinaryInfo`, `PyTaskResult`, Python `TaskDefinition` bridge

### Phase 5: Distributed (DONE)
11. `db_primary_secondary_comm` — 20+ message types, JSON codec, state machine
12. `db_distributed_manager` — primary/secondary coordinators
13. `db_transport_quic` — QUIC (quinn) + WSS fallback

## Testing Strategy

- **Unit tests per crate**: codec roundtrips, budget calculations matching Python exactly, state machine compile-time verification
- **Integration tests**: local multi-worker with real Python worker subprocesses, authoritative/submissive in-process via channel transport (mirrors `--test-master-slave`), network simulation via channel transport (mirrors `--test-master-slave-netsim`)
- **No SLURM**: all tests run locally
- **Compile-time tests**: invalid state transitions must fail to compile (commented-out negative tests)

## Verification

1. `cd rust/dynamic_batch && cargo build --workspace` — all crates compile
2. `cargo test --workspace` — all unit + integration tests pass
3. `cd crates/db_python_provider && maturin develop --release` — Python module builds
4. `python -c "import dynamic_batch_rs"` — import works
5. Run `db_scheduler_impl` unit tests that verify budget calculations match Python's `_calculate_initial_budget` and `_assign_binary_to_worker_normal` exactly
6. Run local integration test with real Python workers to verify end-to-end processing

## Phase 6: Incremental Python Migration

Once the Rust package is working, incrementally replace Python `dynamic_batch` code to use the Rust-provided Python package:

1. Add a `--use-rust-backend` CLI flag to `__main__.py`
2. Behind that flag, instantiate the Rust `PyLocalManager` instead of `LocalWorkerManager`
3. Run both backends on the same inputs and compare results (completed/errored/oom counts must match)
4. Each subsequent component (scheduler, distributed coordinator, etc.) gets its own flag
5. Fix Python bugs discovered during comparison — the Python code has known major bugs
6. Once parity is confirmed for each component, the Rust backend becomes default
7. Eventually remove the Python implementations entirely

This is a continuous, incremental process — never stop working until the full migration is complete.
