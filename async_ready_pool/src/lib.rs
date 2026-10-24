//! Reusable asynchronous preparation pool for tokenizer and training data.
//!
//! The core state machine is PyO3-free and generic over the prepared item.
//! The Python module is only an adapter that owns callback/GIL and lifecycle
//! handling; CUDA and data-format concerns remain in their caller modules.
//!
//! - [`PreloadPool`]: the pool over any producer.
//! - [`loader`]: the data-loader interface. A loader is written once in Rust
//!   and runs in-process ([`loader::LocalPool`]), or behind the loader
//!   bridge ([`bridge`]) in another process, such as a container's host.
//! - Python (feature `python`): `RustPreloader` runs a Python callback, a
//!   local loader binary, or a loader over the bridge socket; `AsyncReadyPool`
//!   is the keyed facade over it.

mod core;
#[cfg(target_os = "linux")]
pub mod fs;
pub mod bridge;
pub mod loader;
#[cfg(feature = "python")]
mod python;

pub use core::{ClaimStop, GetError, PreloadCore, PreloadPool, RegistrationId};
