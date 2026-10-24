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
