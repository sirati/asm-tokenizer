//! PyO3 adapter for the generic preparation pool.
//!
//! The adapter owns the Python callback and GIL boundary. It does not know the
//! shape or meaning of prepared values; callers register opaque IDs and pull
//! opaque Python objects.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

use crate::core::{GetError, PreloadCore, PreloadPool, RegistrationId};

type Batch = Py<PyAny>;
type PythonProducer = Box<dyn Fn(RegistrationId) -> Result<Batch, String> + Send + Sync>;
const CPU_CAPSULE: &std::ffi::CStr = c"async_ready_pool.cpu.v1";

/// Native callers retain this capsule while synchronously borrowing task data.
#[pyfunction]
#[pyo3(signature = (workers=8))]
fn cpu_executor(py: Python<'_>, workers: usize) -> PyResult<Py<PyAny>> {
    let api = crate::cpu::Api::new(workers).map_err(PyValueError::new_err)?;
    let pointer = Box::into_raw(api);
    let capsule = unsafe {
        pyo3::ffi::PyCapsule_New(
            pointer.cast(),
            CPU_CAPSULE.as_ptr(),
            Some(drop_cpu_executor),
        )
    };
    if capsule.is_null() {
        unsafe {
            drop(Box::from_raw(pointer));
        }
        return Err(PyErr::fetch(py));
    }
    Ok(unsafe { Bound::from_owned_ptr(py, capsule) }.unbind())
}

unsafe extern "C" fn drop_cpu_executor(capsule: *mut pyo3::ffi::PyObject) {
    let pointer = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, CPU_CAPSULE.as_ptr()) };
    if !pointer.is_null() {
        unsafe {
            drop(Box::from_raw(pointer.cast::<crate::cpu::Api>()));
        }
    }
}

#[pyclass]
pub struct RustPreloader {
    pool: Mutex<PreloadPool<Batch, PythonProducer>>,
}

#[pymethods]
impl RustPreloader {
    #[new]
    #[pyo3(signature = (produce, n_threads, persistent=false))]
    fn new(produce: Py<PyAny>, n_threads: usize, persistent: bool) -> PyResult<Self> {
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
        let pool = PreloadPool::with_persistence(producer, n_threads, persistent)
            .map_err(PyValueError::new_err)?;
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
        let consumer = self.consumer(py)?;
        py.detach(|| consumer.get(registration))
            .map_err(|error| pool_error(error, registration))
    }

    fn cancel(&self, py: Python<'_>, registration: RegistrationId) -> PyResult<()> {
        let consumer = self.consumer(py)?;
        py.detach(|| consumer.cancel(registration))
            .map_err(|error| pool_error(error, registration))
    }

    #[pyo3(signature = (registration, keep_ready, count=None))]
    fn rearm(
        &self,
        py: Python<'_>,
        registration: RegistrationId,
        keep_ready: usize,
        count: Option<usize>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.pool
                .lock()
                .unwrap()
                .rearm_limit(registration, keep_ready, count)
        })
        .map_err(PyValueError::new_err)
    }

    fn ready_counts(&self, py: Python<'_>) -> Vec<(u64, usize)> {
        py.detach(|| self.pool.lock().unwrap().ready_counts())
    }

    fn close(&self, py: Python<'_>) {
        py.detach(|| self.pool.lock().unwrap().close());
    }
}

impl RustPreloader {
    fn consumer(&self, py: Python<'_>) -> PyResult<Arc<PreloadCore<Batch>>> {
        py.detach(|| self.pool.lock().unwrap().consumer())
            .map_err(|_| PyRuntimeError::new_err("preloader is closed or not started"))
    }
}

fn pool_error(error: GetError, registration: RegistrationId) -> PyErr {
    match error {
        GetError::UnknownBucket(id) => PyValueError::new_err(format!("unknown registration {id}")),
        GetError::Failed(message) => PyRuntimeError::new_err(format!(
            "producer failed for registration {registration}: {message}"
        )),
        GetError::Closing => PyRuntimeError::new_err("preloader is closed or not started"),
        GetError::Exhausted => PyStopIteration::new_err("preparation plan exhausted"),
        GetError::Cancelled => PyStopIteration::new_err("preparation registration cancelled"),
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
    m.add_function(wrap_pyfunction!(cpu_executor, m)?)?;
    Ok(())
}
