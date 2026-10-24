//! PyO3 adapter for the generic preparation pool.
//!
//! The adapter owns the Python callback and GIL boundary. It does not know the
//! shape or meaning of prepared values; callers register opaque IDs and pull
//! opaque Python objects.

use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::core::{GetError, PreloadPool, RegistrationId};

type Batch = Py<PyAny>;
type PythonProducer = Box<dyn Fn(RegistrationId) -> Result<Batch, String> + Send + Sync>;

#[pyclass]
pub struct RustPreloader {
    pool: PreloadPool<Batch, PythonProducer>,
}

#[pymethods]
impl RustPreloader {
    #[new]
    fn new(produce: Py<PyAny>, n_threads: usize) -> PyResult<Self> {
        let callback = Arc::new(produce);
        let producer: PythonProducer = Box::new(move |registration| {
            Python::attach(|py| {
                callback
                    .bind(py)
                    .call1((registration,))
                    .map(|value| value.unbind())
                    .map_err(|error| format!("producer raised: {error}"))
            })
        });
        let pool = PreloadPool::new(producer, n_threads).map_err(PyValueError::new_err)?;
        Ok(Self { pool })
    }

    /// Register one opaque producer slot and return its registration ID.
    fn register(&mut self, keep_ready: usize) -> PyResult<RegistrationId> {
        self.pool
            .register(keep_ready)
            .map_err(PyValueError::new_err)
    }

    fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.pool.start())
            .map_err(PyRuntimeError::new_err)
    }

    fn get(&self, py: Python<'_>, registration: RegistrationId) -> PyResult<Py<PyAny>> {
        match py.detach(|| self.pool.get(registration)) {
            Ok(batch) => Ok(batch),
            Err(GetError::UnknownBucket(id)) => {
                Err(PyValueError::new_err(format!("unknown registration {id}")))
            }
            Err(GetError::Failed(message)) => Err(PyRuntimeError::new_err(format!(
                "producer failed for registration {registration}: {message}"
            ))),
            Err(GetError::Closing) => Err(PyRuntimeError::new_err(
                "preloader is closed or not started",
            )),
        }
    }

    fn ready_counts(&self) -> Vec<(u64, usize)> {
        self.pool.ready_counts()
    }

    fn close(&mut self, py: Python<'_>) {
        py.detach(|| self.pool.close());
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustPreloader>()?;
    Ok(())
}
