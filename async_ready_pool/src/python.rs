//! PyO3 adapter for the generic preparation pool.
//!
//! `RustPreloader` has two kinds of producer:
//! - a Python callable (`RustPreloader(produce, n_threads)`): the adapter owns
//!   the callback and the GIL boundary and returns opaque Python objects;
//! - a Rust loader (`RustPreloader(None, n_threads, loader=...)`): typed
//!   parameters and inputs per registration, items as dicts of read-only
//!   numpy arrays over shared memory. The loader runs either as a local child
//!   process over the data roots given here (`backend="local"`), or behind the
//!   loader-bridge socket (`backend="bridge"`, the container's way to the
//!   host's data). The backend comes from the argument, else from
//!   `ASYNC_READY_POOL_BACKEND`, else "local". The surface is the same.

use std::os::raw::c_int;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{create_exception, ffi};

use crate::bridge::client::{ArrayView, Client, DEFAULT_SOCKET, Error, LoaderInfo};
use crate::bridge::proto::{EntryKind, FileRef, ParamKind, ParamSpec, RefuseCode, Value};
use crate::core::{GetError, PreloadPool, RegistrationId};

type Batch = Py<PyAny>;
type PythonProducer = Box<dyn Fn(RegistrationId, u64) -> Result<Batch, String> + Send + Sync>;

create_exception!(
    _native,
    SessionEnded,
    PyRuntimeError,
    "The loader's session ended: the server ended it, or the connection broke."
);

/// Shared memory a local loader may use when the caller sets none.
const DEFAULT_LOCAL_SHARED: u64 = 4 << 30;

/// One array's memory, exported read-only through the buffer protocol.
#[pyclass(frozen, module = "async_ready_pool._native")]
struct SharedArray {
    view: ArrayView,
}

#[pymethods]
impl SharedArray {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let this = slf.get();
        // readonly = 1: a request for a writable buffer fails with BufferError.
        let r = unsafe {
            ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                this.view.as_ptr() as *mut _,
                this.view.len_bytes() as ffi::Py_ssize_t,
                1,
                flags,
            )
        };
        if r != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}

    fn __len__(&self) -> usize {
        self.view.len_bytes()
    }
}

fn to_py(e: Error, registration: Option<u64>) -> PyErr {
    match e {
        Error::UnknownLoader(m) => PyValueError::new_err(m),
        Error::Refused(RefuseCode::NotFound, m) => PyFileNotFoundError::new_err(m),
        Error::Refused(_, m) => PyValueError::new_err(m),
        Error::ProducerFailed(m) => PyRuntimeError::new_err(match registration {
            Some(r) => format!("producer failed for registration {r}: {m}"),
            None => m,
        }),
        Error::Closing => PyRuntimeError::new_err("preloader is closed or not started"),
        Error::Disconnected(m) => SessionEnded::new_err(format!("loader session ended: {m}")),
        Error::Io(m) => SessionEnded::new_err(m),
    }
}

fn split_file(info: &LoaderInfo, s: &str) -> PyResult<FileRef> {
    let (root, path) = s.split_once('/').unwrap_or((s, ""));
    if !info.roots.iter().any(|r| r == root) {
        return Err(PyValueError::new_err(format!(
            "{s:?}: the first component must be a data root ({})",
            info.roots.join(", ")
        )));
    }
    Ok(FileRef {
        root: root.to_owned(),
        path: path.to_owned(),
    })
}

fn to_value(specs: &[ParamSpec], name: &str, v: &Bound<'_, PyAny>) -> PyResult<Value> {
    let Some(spec) = specs.iter().find(|p| p.name == name) else {
        let known: Vec<_> = specs.iter().map(|p| p.name.as_str()).collect();
        return Err(PyValueError::new_err(format!(
            "unknown parameter {name:?} (known: {})",
            known.join(", ")
        )));
    };
    let value = match &spec.kind {
        ParamKind::Bool => Value::Bool(v.extract()?),
        ParamKind::Int { .. } if !v.is_instance_of::<pyo3::types::PyBool>() => {
            Value::Int(v.extract()?)
        }
        ParamKind::Float { .. } => Value::Float(v.extract()?),
        ParamKind::Choice(_) => Value::Choice(v.extract()?),
        ParamKind::Ints { .. } => {
            let mut out = Vec::new();
            for x in v.try_iter()? {
                let x = x?;
                if x.is_instance_of::<pyo3::types::PyBool>() {
                    return Err(PyValueError::new_err(format!(
                        "parameter {name:?} takes ints"
                    )));
                }
                out.push(x.extract::<i64>()?);
            }
            Value::Ints(out)
        }
        ParamKind::Int { .. } => {
            return Err(PyValueError::new_err(format!(
                "parameter {name:?} takes an int"
            )));
        }
    };
    if !spec.accepts(&value) {
        return Err(PyValueError::new_err(format!(
            "parameter {name:?}: {value:?} is not allowed ({:?})",
            spec.kind
        )));
    }
    Ok(value)
}

/// The values of a parameter dict, checked against `specs`, with every
/// parameter without a default given.
fn to_values(
    specs: &[ParamSpec],
    params: Option<Bound<'_, PyDict>>,
) -> PyResult<Vec<(String, Value)>> {
    let mut values = Vec::new();
    if let Some(params) = params {
        for (k, v) in params.iter() {
            let name: String = k.extract()?;
            values.push((name.clone(), to_value(specs, &name, &v)?));
        }
    }
    for spec in specs {
        if spec.default.is_none() && !values.iter().any(|(n, _)| n == &spec.name) {
            return Err(PyValueError::new_err(format!(
                "parameter {:?} is required",
                spec.name
            )));
        }
    }
    Ok(values)
}

fn describe(py: Python<'_>, specs: &[ParamSpec]) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    for p in specs {
        d.set_item(&p.name, format!("{:?}", p.kind))?;
    }
    Ok(d.unbind())
}

struct LoaderState {
    client: Client,
    workers: u32,
    /// Per registration: on request, and the parameters of its requests.
    registrations: Vec<(bool, Vec<ParamSpec>)>,
    started: bool,
    closed: bool,
}

enum Inner {
    Callback(PreloadPool<Batch, PythonProducer>),
    Loader(LoaderState),
}

#[pyclass]
pub struct RustPreloader {
    inner: Inner,
}

impl RustPreloader {
    fn loader(&self) -> PyResult<&LoaderState> {
        match &self.inner {
            Inner::Loader(l) => Ok(l),
            Inner::Callback(_) => Err(PyValueError::new_err(
                "this preloader runs a Python producer, not a loader",
            )),
        }
    }
}

#[pymethods]
impl RustPreloader {
    /// `RustPreloader(produce, n_threads)` runs the callable `produce(id)`;
    /// `RustPreloader(None, n_threads, loader="name", ...)` runs a Rust loader
    /// (see the module docs for `backend`, `socket`, `program`, `roots`).
    #[new]
    #[pyo3(signature = (produce = None, n_threads = 1, *, loader = None, backend = None, socket = None, program = None, roots = None, max_shared_bytes = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        produce: Option<Py<PyAny>>,
        n_threads: usize,
        loader: Option<String>,
        backend: Option<String>,
        socket: Option<PathBuf>,
        program: Option<PathBuf>,
        roots: Option<Vec<(String, PathBuf)>>,
        max_shared_bytes: Option<u64>,
    ) -> PyResult<Self> {
        match (produce, loader) {
            (Some(produce), None) => {
                if backend.is_some() || socket.is_some() || program.is_some() || roots.is_some() {
                    return Err(PyValueError::new_err(
                        "backend, socket, program and roots are for loaders",
                    ));
                }
                let callback = Arc::new(produce);
                let producer: PythonProducer = Box::new(move |registration, _ticket| {
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
                    inner: Inner::Callback(pool),
                })
            }
            (None, Some(loader)) => {
                if n_threads == 0 {
                    return Err(PyValueError::new_err("workers must be >= 1"));
                }
                let workers = u32::try_from(n_threads)
                    .map_err(|_| PyValueError::new_err("too many workers"))?;
                let backend = backend
                    .or_else(|| std::env::var("ASYNC_READY_POOL_BACKEND").ok())
                    .unwrap_or_else(|| "local".into());
                let client = match backend.as_str() {
                    "bridge" => {
                        if program.is_some() || roots.is_some() {
                            return Err(PyValueError::new_err(
                                "the bridge backend's loaders and data roots are the server's; \
                                 program and roots are for backend=\"local\"",
                            ));
                        }
                        let socket = socket
                            .or_else(|| std::env::var_os("LOADER_BRIDGE_SOCKET").map(PathBuf::from))
                            .unwrap_or_else(|| PathBuf::from(DEFAULT_SOCKET));
                        py.detach(|| Client::connect(&socket, &loader))
                            .map_err(|e| to_py(e, None))?
                    }
                    "local" => {
                        let Some(program) = program else {
                            return Err(PyValueError::new_err(
                                "backend \"local\" needs program= (the loader binary)",
                            ));
                        };
                        let roots = roots.unwrap_or_default();
                        let shared = max_shared_bytes.unwrap_or(DEFAULT_LOCAL_SHARED);
                        py.detach(|| {
                            Client::spawn_local(&program, &loader, &roots, shared, workers)
                        })
                        .map_err(|e| to_py(e, None))?
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "unknown backend {other:?} (local, bridge)"
                        )));
                    }
                };
                if workers > client.info().max_workers {
                    return Err(PyValueError::new_err(format!(
                        "n_threads {workers} is over the loader's max of {}",
                        client.info().max_workers
                    )));
                }
                Ok(Self {
                    inner: Inner::Loader(LoaderState {
                        client,
                        workers,
                        registrations: Vec::new(),
                        started: false,
                        closed: false,
                    }),
                })
            }
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "give either a producer or a loader, not both",
            )),
            (None, None) => Err(PyValueError::new_err("give a producer or a loader")),
        }
    }

    /// Registers one producer slot and returns its registration ID. A loader
    /// takes its parameters and inputs ("<root>" or "<root>/<path>") here,
    /// and `resume`, the bytes of an item's `resume_state` to go on after.
    /// `on_request`: items are made only as `request` asks for them, in that
    /// order; otherwise `keep_ready` are kept made ahead.
    #[pyo3(signature = (keep_ready, params = None, files = None, *, resume = None, on_request = false))]
    fn register(
        &mut self,
        py: Python<'_>,
        keep_ready: usize,
        params: Option<Bound<'_, PyDict>>,
        files: Option<Vec<String>>,
        resume: Option<Vec<u8>>,
        on_request: bool,
    ) -> PyResult<RegistrationId> {
        let state = match &mut self.inner {
            Inner::Callback(pool) => {
                if params.is_some() || files.is_some() || resume.is_some() {
                    return Err(PyValueError::new_err(
                        "params, files and resume are for loaders",
                    ));
                }
                return match on_request {
                    true => pool.register_on_request(keep_ready),
                    false => pool.register(keep_ready),
                }
                .map_err(PyValueError::new_err);
            }
            Inner::Loader(state) => state,
        };
        if state.closed {
            return Err(PyValueError::new_err("pool has been closed"));
        }
        if state.started {
            return Err(PyValueError::new_err("register must happen before start"));
        }
        let keep_ready = u32::try_from(keep_ready)
            .ok()
            .filter(|k| *k > 0)
            .ok_or_else(|| PyValueError::new_err("keep_ready must be greater than zero"))?;
        let info = state.client.info().clone();
        let values = to_values(&info.params, params)?;
        if resume
            .as_ref()
            .is_some_and(|r| r.len() > crate::bridge::proto::MAX_RESUME)
        {
            return Err(PyValueError::new_err("resume state over its size limit"));
        }
        let files = files
            .unwrap_or_default()
            .iter()
            .map(|f| split_file(&info, f))
            .collect::<PyResult<Vec<_>>>()?;
        if files.len() < info.min_files as usize || files.len() > info.max_files as usize {
            return Err(PyValueError::new_err(format!(
                "{} files given, the loader takes {} to {}",
                files.len(),
                info.min_files,
                info.max_files
            )));
        }
        let client = &state.client;
        let (id, request_params) = py
            .detach(|| client.register(keep_ready, values, files, resume, on_request))
            .map_err(|e| to_py(e, None))?;
        if id != state.registrations.len() as u64 {
            return Err(SessionEnded::new_err(
                "the host numbered a registration out of order",
            ));
        }
        state.registrations.push((on_request, request_params));
        Ok(id)
    }

    /// Asks a registration on request for its next item, with the request's
    /// parameters (checked here against the registration's). Items come
    /// back to `get` in the order they were requested.
    #[pyo3(signature = (registration, params = None))]
    fn request(
        &self,
        py: Python<'_>,
        registration: RegistrationId,
        params: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let state = match &self.inner {
            Inner::Callback(pool) => {
                if params.is_some() {
                    return Err(PyValueError::new_err("request parameters are for loaders"));
                }
                return pool.request(registration, 1).map_err(PyValueError::new_err);
            }
            Inner::Loader(state) => state,
        };
        if !state.started || state.closed {
            return Err(PyRuntimeError::new_err(
                "preloader is closed or not started",
            ));
        }
        let Some((on_request, specs)) = state.registrations.get(registration as usize) else {
            return Err(PyValueError::new_err(format!(
                "unknown registration {registration}"
            )));
        };
        if !on_request {
            return Err(PyValueError::new_err(format!(
                "registration {registration} is kept ready, not on request"
            )));
        }
        let values = to_values(specs, params)?;
        let client = &state.client;
        py.detach(|| client.request(registration, values))
            .map_err(|e| to_py(e, Some(registration)))
    }

    /// The parameters each request of a registration takes, by name.
    fn request_params(&self, py: Python<'_>, registration: RegistrationId) -> PyResult<Py<PyDict>> {
        let state = self.loader()?;
        match state.registrations.get(registration as usize) {
            Some((_, specs)) => describe(py, specs),
            None => Err(PyValueError::new_err(format!(
                "unknown registration {registration}"
            ))),
        }
    }

    fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        let state = match &mut self.inner {
            Inner::Callback(pool) => {
                return py.detach(|| pool.start()).map_err(PyRuntimeError::new_err);
            }
            Inner::Loader(state) => state,
        };
        if state.closed {
            return Err(PyRuntimeError::new_err("pool has been closed"));
        }
        if state.started {
            return Err(PyRuntimeError::new_err("pool has already been started"));
        }
        if state.registrations.is_empty() {
            return Err(PyRuntimeError::new_err(
                "register at least one producer before start",
            ));
        }
        let (client, workers) = (&state.client, state.workers);
        py.detach(|| client.start(workers))
            .map_err(|e| to_py(e, None))?;
        state.started = true;
        Ok(())
    }

    /// The next item of a registration: the producer's object, or for a
    /// loader a dict {name: read-only numpy array} viewing shared memory;
    /// the memory goes back once every array of the item is gone.
    fn get(&self, py: Python<'_>, registration: RegistrationId) -> PyResult<Py<PyAny>> {
        let state = match &self.inner {
            Inner::Callback(pool) => {
                return match py.detach(|| pool.get(registration)) {
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
                };
            }
            Inner::Loader(state) => state,
        };
        if !state.started || state.closed {
            return Err(PyRuntimeError::new_err(
                "preloader is closed or not started",
            ));
        }
        if registration >= state.registrations.len() as u64 {
            return Err(PyValueError::new_err(format!(
                "unknown registration {registration}"
            )));
        }
        let client = &state.client;
        let batch = py
            .detach(|| client.get(registration))
            .map_err(|e| to_py(e, Some(registration)))?;
        let numpy = py.import("numpy")?;
        let out = PyDict::new(py);
        for view in batch.arrays() {
            let dtype = view.dtype.numpy();
            let shape: Vec<u64> = view.shape.clone();
            let name = view.name.clone();
            let buf = Bound::new(py, SharedArray { view })?;
            let flat = numpy.call_method1(
                "frombuffer",
                (buf, numpy.getattr("dtype")?.call1((dtype,))?),
            )?;
            out.set_item(name, flat.call_method1("reshape", (shape,))?)?;
        }
        Ok(out.into_any().unbind())
    }

    fn ready_counts(&self, py: Python<'_>) -> PyResult<Vec<(u64, usize)>> {
        match &self.inner {
            Inner::Callback(pool) => Ok(pool.ready_counts()),
            Inner::Loader(state) => {
                if state.closed {
                    return Ok(Vec::new());
                }
                let client = &state.client;
                let counts = py
                    .detach(|| client.ready_counts())
                    .map_err(|e| to_py(e, None))?;
                Ok(counts.into_iter().map(|(id, n)| (id, n as usize)).collect())
            }
        }
    }

    fn close(&mut self, py: Python<'_>) {
        match &mut self.inner {
            Inner::Callback(pool) => py.detach(|| pool.close()),
            Inner::Loader(state) => {
                if state.closed {
                    return;
                }
                state.closed = true;
                let client = &state.client;
                let _ = py.detach(|| client.close());
            }
        }
    }

    /// A loader's inputs: entries of a directory under a data root ("<root>"
    /// or "<root>/<path>"; "" lists the roots) as (name, kind, size).
    #[pyo3(signature = (path = ""))]
    fn list_files<'py>(&self, py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyList>> {
        let state = self.loader()?;
        let info = state.client.info();
        let out = PyList::empty(py);
        if path.is_empty() {
            for r in &info.roots {
                out.append((r.as_str(), "dir", 0u64))?;
            }
            return Ok(out);
        }
        let f = split_file(info, path.trim_end_matches('/'))?;
        let client = &state.client;
        let mut offset = 0u32;
        loop {
            let (entries, more) = py
                .detach(|| client.list_dir(&f.root, &f.path, offset))
                .map_err(|e| to_py(e, None))?;
            offset += entries.len() as u32;
            for e in entries {
                let kind = match e.kind {
                    EntryKind::File => "file",
                    EntryKind::Dir => "dir",
                    EntryKind::Symlink => "symlink",
                    EntryKind::Other => "other",
                };
                out.append((e.name, kind, e.size))?;
            }
            if !more {
                return Ok(out);
            }
        }
    }

    /// A loader's name, data roots, parameters and limits.
    #[getter]
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let i = self.loader()?.client.info();
        let d = PyDict::new(py);
        d.set_item("loader", &i.loader)?;
        d.set_item("roots", i.roots.clone())?;
        d.set_item("params", describe(py, &i.params)?)?;
        d.set_item("files", (i.min_files, i.max_files))?;
        d.set_item("max_shared_bytes", i.max_shared_bytes)?;
        d.set_item("max_workers", i.max_workers)?;
        Ok(d)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustPreloader>()?;
    m.add("SessionEnded", m.py().get_type::<SessionEnded>())?;
    m.add("DEFAULT_SOCKET", DEFAULT_SOCKET)?;
    Ok(())
}
