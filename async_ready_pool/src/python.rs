//! PyO3 adapter for the generic preparation pool.
//!
//! The adapter owns the Python callback and GIL boundary. It does not know the
//! shape or meaning of prepared values; callers register opaque IDs and pull
//! opaque Python objects.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

use crate::core::{GetError, PreloadPool, RegistrationId};

type Batch = Py<PyAny>;
type PythonProducer = Box<dyn Fn(RegistrationId) -> Result<Batch, String> + Send + Sync>;

#[pyclass]
pub struct RustPreloader {
    pool: Mutex<PreloadPool<Batch, PythonProducer>>,
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
        Ok(Self {
            pool: Mutex::new(pool),
        })
    }

    /// Register one opaque producer slot and return its registration ID.
    #[pyo3(signature = (keep_ready, count=None))]
    fn register(
        &self,
        py: Python<'_>,
        keep_ready: usize,
        count: Option<usize>,
    ) -> PyResult<RegistrationId> {
        py.detach(|| self.pool.lock().unwrap().register_limit(keep_ready, count))
            .map_err(PyValueError::new_err)
    }

    fn start(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.pool.lock().unwrap().start())
            .map_err(PyRuntimeError::new_err)
    }

    fn get(&self, py: Python<'_>, registration: RegistrationId) -> PyResult<Py<PyAny>> {
        let consumer = py
            .detach(|| self.pool.lock().unwrap().consumer())
            .map_err(|_| PyRuntimeError::new_err("preloader is closed or not started"))?;
        match py.detach(|| consumer.get(registration)) {
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
            Err(GetError::Exhausted) => Err(PyStopIteration::new_err("preparation plan exhausted")),
        }
    }

    fn ready_counts(&self, py: Python<'_>) -> Vec<(u64, usize)> {
        py.detach(|| self.pool.lock().unwrap().ready_counts())
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| self.pool.lock().unwrap().close());
    }
}

impl Drop for RustPreloader {
    fn drop(&mut self) {
        // A Python object normally drops while holding the GIL. Workers can
        // need it to finish, so joining while attached would deadlock.
        let _ = Python::try_attach(|py| py.detach(|| self.pool.get_mut().unwrap().close()));
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustPreloader>()?;
    Ok(())
}
