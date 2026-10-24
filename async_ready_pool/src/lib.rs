//! Reusable asynchronous preparation pool for tokenizer and training data.
//!
//! The core state machine is PyO3-free and generic over the prepared item.
//! The Python module is only an adapter that owns callback/GIL and lifecycle
//! handling; CUDA and data-format concerns remain in their caller modules.

mod core;
mod python;

pub use core::{ClaimStop, GetError, PreloadCore, PreloadPool, RegistrationId};
