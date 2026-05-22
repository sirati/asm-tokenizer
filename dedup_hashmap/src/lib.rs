//! `u64 -> u32` and `u32 -> u32` hashmaps exposed to Python.
//!
//! [`HashMapU64U32`] drives the primary side of asm-tokenizer's
//! content-addressed dedup in `tokenizer.memmap_builder`. The Python
//! wrapper around it holds the bulk of the deduplication state
//! (millions of entries on a real corpus); the collision side (cold
//! path, tens of entries) stays in a plain Python `dict` so we avoid
//! `Vec`-value plumbing across the FFI boundary. See
//! `polished-greeting-moler.md`.
//!
//! [`HashMapU32U32`] backs `SectionWriter._known_sections` in
//! `tokenizer.aligned_data.matched_sections_bin` — a `function_name_ptr
//! -> section_offset` lookup that grows to one entry per function in
//! the corpus. Both key and value are u32, so the smaller variant
//! shaves the per-entry footprint in half versus the u64-keyed map.

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

#[pyclass(module = "dedup_hashmap")]
struct HashMapU32U32 {
    inner: HashMap<u32, u32>,
}

#[pymethods]
impl HashMapU32U32 {
    #[new]
    fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    fn get(&self, key: u32) -> Option<u32> {
        self.inner.get(&key).copied()
    }

    fn set(&mut self, key: u32, value: u32) {
        self.inner.insert(key, value);
    }

    fn __contains__(&self, key: u32) -> bool {
        self.inner.contains_key(&key)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[pymodule]
fn dedup_hashmap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HashMapU64U32>()?;
    m.add_class::<HashMapU32U32>()?;
    Ok(())
}
