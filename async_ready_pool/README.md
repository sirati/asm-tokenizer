# tokenizer-async-ready-pool

This crate extracts the asynchronous preparation boundary from the tokenizer
loader. The PyO3-free `PreloadCore<T>` owns bounded ready buffers, worker
claims, cancellation, and failure wakeups. The PyO3 adapter owns only callback
invocation, GIL release around waits/joins, and Python object transport.

Rust callers implement the data-loading function that receives an opaque
`RegistrationId` and returns one prepared item. Python callers can use
`AsyncReadyPool` as a transition adapter for existing no-argument producers.
The pool keeps each registration at its configured ready depth and never lets a
consumer wait forever after a producer fails or the pool closes.

CUDA, tensor upload, tokenizer-specific sampling, and checkpoint/RNG policy
remain outside this crate.

## Data loaders

`loader` is an interface for data loaders written once in Rust: a `Loader`
declares typed parameters (`ParamSpec`: int, float, bool, choice) and how many
inputs a registration takes, and opens a `Source` per registration; a source
has a fixed layout of named typed arrays (`ArraySpec`) and fills one item at a
time into a `BatchMut`. Inputs are files, or directories walked through
`DataDir`, always opened beneath configured data roots with openat2
`RESOLVE_BENEATH`. `fill` must write every byte of every array.

A loader runs

- in-process from Rust: `loader::LocalPool`;
- as a loader binary serving the bridge protocol (`bridge`: versioned,
  bounds-checked messages over SOCK_SEQPACKET, items in sealed read-only
  memfd segments). The serving side, a sandboxed broker for a container, is
  sirati-nixos' `loader-bridge`, which links this crate.

From Python the classes are the same for every kind of producer:

```python
from async_ready_pool import AsyncReadyPool, PoolConfig, RustLoader

pool = AsyncReadyPool(
    [PoolConfig("train", RustLoader("dclm", {"target_len": 2048, "batch": 8, "null_token_id": 0},
                                    ("dclm/dclm-baseline-1.0", "dclm/token_dict.zstd-dict", "tok/tokenizer.json")), 4)],
    workers=2,
)
batch = pool.start().get("train")     # {"input_ids": ndarray, ...}, read-only views
```

`RustPreloader(None, n_threads, loader=..., backend=...)` is the same at the
registration level (`register(keep_ready, params, files)`, `get`, `list_files`,
`info`). The backend is `backend=`, else `$ASYNC_READY_POOL_BACKEND`, else
`"local"`:

- `"bridge"`: the loader-bridge socket (`socket=`, `$LOADER_BRIDGE_SOCKET`,
  `/run/loader-bridge/sock`); the server owns the loaders and data roots.
- `"local"`: the loader binary as a child of this process (`program=` /
  `programs={name: path}`, `roots=` data roots), over a private socket pair.

Items are dicts of read-only numpy arrays over shared memory; the memory goes
back to the loader when the last array of an item is gone. A session the
other side ended raises `SessionEnded`.
