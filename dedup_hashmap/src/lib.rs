//! `u64 -> u32` hashmap exposed to Python.
//!
//! Drives the primary side of asm-tokenizer's content-addressed dedup
//! in `tokenizer.memmap_builder`. The Python wrapper around this class
//! holds the bulk of the deduplication state (millions of entries on a
//! real corpus); the collision side (cold path, tens of entries) stays
//! in a plain Python `dict` so we avoid `Vec`-value plumbing across the
//! FFI boundary. See `polished-greeting-moler.md`.

use hashbrown::HashMap;
use pyo3::prelude::*;

#[pyclass(module = "dedup_hashmap")]
struct HashMapU64U32 {
    inner: HashMap<u64, u32>,
}

#[pymethods]
impl HashMapU64U32 {
    #[new]
    fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    fn get(&self, key: u64) -> Option<u32> {
        self.inner.get(&key).copied()
    }

    fn set(&mut self, key: u64, value: u32) {
        self.inner.insert(key, value);
    }

    fn __contains__(&self, key: u64) -> bool {
        self.inner.contains_key(&key)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn dedup_hashmap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HashMapU64U32>()?;
    Ok(())
}
