//! `define_hashmap!` — emits one `#[pyclass]` + `#[pymethods]` block per
//! invocation.
//!
//! See `lib.rs` for the invocation site that walks the Cartesian product
//! of supported key/value types and registers each generated class in
//! the `#[pymodule]` block.
//!
//! Each generated class wraps `hashbrown::HashMap<K, V>` and exposes
//! a uniform surface:
//!
//! - `__new__(capacity: int = 0)` — pre-sized when `capacity > 0`,
//!   otherwise `HashMap::new()`. Zero-arg construction is preserved so
//!   existing call sites (`HashMapU64U32()`, `HashMapU32U32()`) keep
//!   working unchanged.
//! - `get(key) / lookup(key) -> Optional[V]` — scalar lookup. Both
//!   names map to the same implementation; `lookup` matches the
//!   batch-API terminology, `get` preserves the pre-existing surface.
//! - `set(key, value) / insert(key, value)` — scalar insert; same
//!   aliasing rationale.
//! - `lookup_ndarray(keys: NDArray[K]) -> NDArray[V]` — vectorized
//!   lookup. Missing keys come back as the per-dtype sentinel (see
//!   `lib.rs` for the table).
//! - `insert_ndarray(keys: NDArray[K], values: NDArray[V])` — batch
//!   insert; iteration is left-to-right so duplicate keys within one
//!   call follow last-wins semantics (consistent with scalar `insert`).
//! - `clean()` — clears entries while retaining the bucket allocation
//!   (`HashMap::clear`). Required by the stage-4 dedup walk that reuses
//!   one map per Category across thousands of rows.
//! - `__len__()`, `__contains__(key)`.

/// Generate a single `HashMap<K><V>` PyO3 class.
///
/// Parameters:
/// - `$cls_ident` — the generated class identifier (`HashMapU32U16` etc.).
/// - `$key_ty`   — the Rust key type (also the numpy element dtype).
/// - `$val_ty`   — the Rust value type.
/// - `$sentinel` — the value returned by `lookup_ndarray` for misses.
macro_rules! define_hashmap {
    ($cls_ident:ident, $key_ty:ty, $val_ty:ty, $sentinel:expr) => {
        #[pyclass(module = "dedup_hashmap._native")]
        pub struct $cls_ident {
            inner: ::hashbrown::HashMap<$key_ty, $val_ty>,
        }

        #[pymethods]
        impl $cls_ident {
            #[new]
            #[pyo3(signature = (capacity = 0))]
            fn new(capacity: usize) -> Self {
                let inner = if capacity > 0 {
                    ::hashbrown::HashMap::with_capacity(capacity)
                } else {
                    ::hashbrown::HashMap::new()
                };
                Self { inner }
            }

            fn get(&self, key: $key_ty) -> Option<$val_ty> {
                self.inner.get(&key).copied()
            }

            fn lookup(&self, key: $key_ty) -> Option<$val_ty> {
                self.inner.get(&key).copied()
            }

            fn set(&mut self, key: $key_ty, value: $val_ty) {
                self.inner.insert(key, value);
            }

            fn insert(&mut self, key: $key_ty, value: $val_ty) {
                self.inner.insert(key, value);
            }

            fn lookup_ndarray<'py>(
                &self,
                py: ::pyo3::Python<'py>,
                keys: ::numpy::PyReadonlyArray1<'py, $key_ty>,
            ) -> ::pyo3::PyResult<::pyo3::Bound<'py, ::numpy::PyArray1<$val_ty>>> {
                let keys_slice = keys.as_slice()?;
                let n = keys_slice.len();
                let mut out: Vec<$val_ty> = vec![$sentinel; n];
                py.detach(|| {
                    for (i, k) in keys_slice.iter().enumerate() {
                        if let Some(v) = self.inner.get(k) {
                            out[i] = *v;
                        }
                    }
                });
                Ok(::numpy::PyArray1::from_vec(py, out))
            }

            fn insert_ndarray<'py>(
                &mut self,
                py: ::pyo3::Python<'py>,
                keys: ::numpy::PyReadonlyArray1<'py, $key_ty>,
                values: ::numpy::PyReadonlyArray1<'py, $val_ty>,
            ) -> ::pyo3::PyResult<()> {
                let keys_slice = keys.as_slice()?;
                let values_slice = values.as_slice()?;
                if keys_slice.len() != values_slice.len() {
                    return Err(::pyo3::exceptions::PyValueError::new_err(format!(
                        "keys and values must have equal length (got {} vs {})",
                        keys_slice.len(),
                        values_slice.len()
                    )));
                }
                py.detach(|| {
                    for (k, v) in keys_slice.iter().zip(values_slice.iter()) {
                        self.inner.insert(*k, *v);
                    }
                });
                Ok(())
            }

            fn clean(&mut self) {
                self.inner.clear();
            }

            fn __contains__(&self, key: $key_ty) -> bool {
                self.inner.contains_key(&key)
            }

            fn __len__(&self) -> usize {
                self.inner.len()
            }
        }
    };
}

pub(crate) use define_hashmap;
