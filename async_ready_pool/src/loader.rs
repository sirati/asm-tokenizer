//! The data-loader interface: a loader written once in Rust, run by this
//! crate's pool in-process ([`LocalPool`]), from Python ([`RustPreloader`]
//! with `loader=`), or in another process over the loader bridge
//! ([`crate::bridge`]; the `loader-bridge` crate serves it to a container).
//!
//! A [`Loader`] declares typed parameters and how many inputs a registration
//! takes, and opens a [`Source`] per registration. Inputs are files, or
//! directories the loader walks itself through [`DataDir`], always opened
//! beneath the data roots the runner was configured with; the loader never
//! sees a path the consumer chose that was not resolved that way.
//!
//! A source makes its items one after another. [`Source::next`] does what
//! must happen in order (draws from the source's random state, cursors) and
//! returns an [`Item`]; runners call it for a registration's items one at a
//! time, in the order the consumer gets them. The item then says its layout
//! (named arrays with element types and shapes, which may differ from item
//! to item) and fills itself into a [`BatchMut`], on any worker,
//! concurrently with other items. `fill` must write every byte of every
//! array: runners reuse the memory of released items without clearing it.
//!
//! A registration is either kept ready (the runner keeps `keep_ready` items
//! made ahead, each with the source's default request) or on request: each
//! item is asked for with its own parameters ([`Source::request_params`]),
//! and the items come back in the order they were asked for. A source that
//! can say where it is ([`Source::state`]) can be resumed: every item then
//! carries [`RESUME_STATE`], the state after it, and a registration given
//! that state ([`Loader::open`]'s `resume`) goes on with the item after it.
//!
//! [`RustPreloader`]: crate#python

use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::sync::{Arc, Condvar, Mutex};

pub use crate::bridge::proto::{
    ArrayDesc, DType, DirEntry, EntryKind, ParamKind, ParamSpec, Value, valid_name,
};
#[cfg(target_os = "linux")]
pub use crate::fs::{DataDir, DataKind, PathError, Root};

use crate::bridge::proto;

/// The array every item of a resumable source carries: the source's state
/// after the item, as bytes.
pub const RESUME_STATE: &str = "resume_state";

/// Requests a registration on request may have waiting.
pub const MAX_QUEUED_REQUESTS: usize = 4096;

/// Validated parameters of one registration: each declared one, with the
/// value given or its default.
#[derive(Debug, Clone, Default)]
pub struct Params {
    values: BTreeMap<String, Value>,
}

impl Params {
    /// Checks `values` against `specs`: every name declared and given at
    /// most once, every value accepted, every parameter without a default
    /// given. The defaults fill in the rest.
    pub fn resolve(specs: &[ParamSpec], values: Vec<(String, Value)>) -> Result<Params, String> {
        let mut out = BTreeMap::new();
        for (name, value) in values {
            let Some(spec) = specs.iter().find(|p| p.name == name) else {
                return Err(format!("unknown parameter {name:?}"));
            };
            if !spec.accepts(&value) {
                return Err(format!("parameter {name:?}: {value:?} is not allowed"));
            }
            if out.insert(name.clone(), value).is_some() {
                return Err(format!("parameter {name:?} given twice"));
            }
        }
        for spec in specs {
            if !out.contains_key(&spec.name) {
                match &spec.default {
                    Some(d) => {
                        out.insert(spec.name.clone(), d.clone());
                    }
                    None => return Err(format!("parameter {:?} is required", spec.name)),
                }
            }
        }
        Ok(Params { values: out })
    }

    pub fn get(&self, name: &str) -> Option<&Value> {
        self.values.get(name)
    }
    pub fn int(&self, name: &str) -> i64 {
        match self.values.get(name) {
            Some(Value::Int(v)) => *v,
            _ => panic!("parameter {name} is not a declared int"),
        }
    }
    pub fn float(&self, name: &str) -> f64 {
        match self.values.get(name) {
            Some(Value::Float(v)) => *v,
            _ => panic!("parameter {name} is not a declared float"),
        }
    }
    pub fn bool(&self, name: &str) -> bool {
        match self.values.get(name) {
            Some(Value::Bool(v)) => *v,
            _ => panic!("parameter {name} is not a declared bool"),
        }
    }
    pub fn ints(&self, name: &str) -> &[i64] {
        match self.values.get(name) {
            Some(Value::Ints(v)) => v,
            _ => panic!("parameter {name} is not a declared int list"),
        }
    }
    pub fn choice(&self, name: &str) -> &str {
        match self.values.get(name) {
            Some(Value::Choice(v)) => v,
            _ => panic!("parameter {name} is not a declared choice"),
        }
    }
}

/// An input a consumer selected, opened read-only beneath its data root.
/// For a loader that [takes directories](Loader::takes_dirs), a directory
/// comes as `dir`, and `file` is that directory (reads fail).
pub struct InputFile {
    pub root: String,
    pub path: String,
    pub file: File,
    #[cfg(target_os = "linux")]
    pub dir: Option<DataDir>,
}

/// One array of every item of a source: name, element type, shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArraySpec {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl ArraySpec {
    pub fn new(name: &str, dtype: DType, shape: &[u64]) -> ArraySpec {
        ArraySpec {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
        }
    }
}

/// A data loader. A runner makes one per session or pool.
pub trait Loader: Send + Sync + 'static {
    /// The parameters a registration may set.
    fn params(&self) -> Vec<ParamSpec>;
    /// How many inputs a registration takes (min, max).
    fn files(&self) -> (u16, u16);
    /// Whether an input may name a directory (`<root>` alone is the root).
    fn takes_dirs(&self) -> bool {
        false
    }
    /// Prepares a registration. `resume` is a [`RESUME_STATE`] an item of an
    /// earlier registration with the same parameters and inputs carried: the
    /// source then makes the items that would have followed that one. A
    /// state the loader does not recognise is an error. An error is reported
    /// to the consumer, which may try again.
    fn open(
        &self,
        params: &Params,
        files: Vec<InputFile>,
        resume: Option<&[u8]>,
    ) -> Result<Box<dyn Source>, String>;
}

/// Makes the items of one registration, one after another.
pub trait Source: Send + 'static {
    /// The parameters of one request. A registration kept ready asks for
    /// every item with their defaults; one on request with the values of
    /// each request.
    fn request_params(&self) -> Vec<ParamSpec> {
        Vec::new()
    }
    /// The arrays of the largest item this source makes: what a runner
    /// budgets its memory for. An item larger than this fails.
    fn max_layout(&self) -> Vec<ArraySpec>;
    /// The next item, for `request`: everything that has to happen in order.
    /// Called for one item at a time, in the order the consumer gets them.
    /// An error fails the registration: the consumer's next `get` for it
    /// raises this message.
    fn next(&mut self, request: &Params) -> Result<Box<dyn Item>, String>;
    /// The state after the last `next`, for [`Loader::open`]'s `resume`;
    /// None when the source cannot be resumed. At most
    /// [`proto::MAX_RESUME`] bytes.
    fn state(&self) -> Option<Vec<u8>> {
        None
    }
    /// The longest `state` can be, for the memory budget.
    fn max_state_len(&self) -> usize {
        proto::MAX_RESUME
    }
}

/// One item a source decided on, to be filled on any worker.
pub trait Item: Send + 'static {
    /// Its arrays.
    fn layout(&self) -> Vec<ArraySpec>;
    /// Writes every byte of every array.
    fn fill(self: Box<Self>, batch: &mut BatchMut<'_>) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// Layout and the item being filled.

/// Items and their arrays start at this alignment.
pub const ALIGN: u64 = 64;

pub fn align_up(v: u64, a: u64) -> Option<u64> {
    v.checked_add(a - 1).map(|x| x & !(a - 1))
}

/// Where each array of an item lies: one after another, each ALIGN-aligned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub arrays: Vec<ArrayDesc>,
    pub bytes: u64,
}

impl Layout {
    /// None when the sizes overflow.
    pub fn new(specs: &[(String, DType, Vec<u64>)]) -> Option<Layout> {
        let mut off = 0u64;
        let mut arrays = Vec::with_capacity(specs.len());
        for (name, dtype, shape) in specs {
            let a = ArrayDesc {
                name: name.clone(),
                dtype: *dtype,
                shape: shape.clone(),
                offset: off,
            };
            let end = off.checked_add(a.nbytes()?)?;
            arrays.push(a);
            off = align_up(end, ALIGN)?;
        }
        Some(Layout {
            arrays,
            bytes: off.max(ALIGN),
        })
    }

    /// The layout of a source, checked as every runner checks it: at least
    /// one array, at most `MAX_ARRAYS`, valid names, at most `MAX_DIMS`
    /// dimensions, sizes that do not overflow.
    pub fn of(specs: &[ArraySpec]) -> Result<Layout, String> {
        if specs.is_empty()
            || specs.len() > proto::MAX_ARRAYS
            || specs
                .iter()
                .any(|a| !valid_name(&a.name) || a.shape.len() > proto::MAX_DIMS)
        {
            return Err("the loader's layout is not representable".into());
        }
        let tuples: Vec<_> = specs
            .iter()
            .map(|a| (a.name.clone(), a.dtype, a.shape.clone()))
            .collect();
        Layout::new(&tuples).ok_or_else(|| "the loader's item size overflows".into())
    }
}

/// Element types a loader can write through a typed slice: plain integers
/// and floats, for which any bit pattern is valid.
///
/// # Safety
/// `matches` may only accept dtypes of the same size whose every bit
/// pattern is a valid `Self`.
pub unsafe trait Element: Copy {
    fn matches(d: DType) -> bool;
}

macro_rules! element {
    ($t:ty, $($d:ident)|+) => {
        unsafe impl Element for $t {
            fn matches(d: DType) -> bool {
                matches!(d, $(DType::$d)|+)
            }
        }
    };
}
element!(u8, U8 | Bool);
element!(i8, I8);
element!(u16, U16 | F16);
element!(i16, I16);
element!(u32, U32);
element!(i32, I32);
element!(u64, U64);
element!(i64, I64);
element!(f32, F32);
element!(f64, F64);

/// One item being filled: its arrays over a region of memory the filler
/// holds exclusively.
pub struct BatchMut<'a> {
    buf: &'a mut [u8],
    arrays: &'a [ArrayDesc],
}

impl<'a> BatchMut<'a> {
    /// The item over `buf` (ALIGN-aligned), `arrays` lying inside it. None
    /// when they do not.
    pub fn new(buf: &'a mut [u8], arrays: &'a [ArrayDesc]) -> Option<BatchMut<'a>> {
        if buf.as_ptr() as usize % ALIGN as usize != 0 {
            return None;
        }
        let fits = arrays.iter().all(|a| {
            a.offset % ALIGN == 0
                && a.nbytes()
                    .and_then(|n| a.offset.checked_add(n))
                    .is_some_and(|end| end <= buf.len() as u64)
        });
        fits.then_some(BatchMut { buf, arrays })
    }

    pub fn arrays(&self) -> &[ArrayDesc] {
        self.arrays
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.arrays.iter().position(|a| a.name == name)
    }

    /// The bytes of array `i`.
    pub fn bytes_mut(&mut self, i: usize) -> &mut [u8] {
        let a = &self.arrays[i];
        // In bounds: checked in new().
        let start = a.offset as usize;
        let n = a.nbytes().unwrap() as usize;
        &mut self.buf[start..start + n]
    }

    /// Array `i` as a typed slice; None when `E` is not its element type.
    pub fn array_mut<E: Element>(&mut self, i: usize) -> Option<&mut [E]> {
        let a = &self.arrays[i];
        if !E::matches(a.dtype) || std::mem::size_of::<E>() != a.dtype.size() {
            return None;
        }
        let bytes = self.bytes_mut(i);
        let n = bytes.len() / std::mem::size_of::<E>();
        // Aligned: the buffer and every array are ALIGN-aligned; any bit
        // pattern is a valid E (Element).
        Some(unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut E, n) })
    }
}

// ---------------------------------------------------------------------------
// Items in process memory.

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Block([u8; ALIGN as usize]);

/// A filled item in this process's memory.
pub struct OwnedBatch {
    pub arrays: Vec<ArrayDesc>,
    blocks: Vec<Block>,
}

impl OwnedBatch {
    fn new(layout: &Layout) -> OwnedBatch {
        let n = (layout.bytes / ALIGN) as usize;
        OwnedBatch {
            arrays: layout.arrays.clone(),
            // Filled with 0xA5 so an array a source leaves partly unwritten
            // shows in tests as it would in shared memory.
            blocks: vec![Block([0xA5; ALIGN as usize]); n],
        }
    }

    fn buf_mut(&mut self) -> &mut [u8] {
        let len = self.blocks.len() * ALIGN as usize;
        unsafe { std::slice::from_raw_parts_mut(self.blocks.as_mut_ptr() as *mut u8, len) }
    }

    fn buf(&self) -> &[u8] {
        let len = self.blocks.len() * ALIGN as usize;
        unsafe { std::slice::from_raw_parts(self.blocks.as_ptr() as *const u8, len) }
    }

    /// The bytes of array `i`.
    pub fn bytes(&self, i: usize) -> &[u8] {
        let a = &self.arrays[i];
        let start = a.offset as usize;
        &self.buf()[start..start + a.nbytes().unwrap() as usize]
    }

    /// Every array with its spec, copied out.
    pub fn to_arrays(&self) -> Vec<(ArraySpec, Vec<u8>)> {
        (0..self.arrays.len())
            .map(|i| {
                let a = &self.arrays[i];
                (
                    ArraySpec::new(&a.name, a.dtype, &a.shape),
                    self.bytes(i).to_vec(),
                )
            })
            .collect()
    }

    /// An array by name: its description and bytes.
    pub fn array(&self, name: &str) -> Option<(&ArrayDesc, &[u8])> {
        let i = self.arrays.iter().position(|a| a.name == name)?;
        Some((&self.arrays[i], self.bytes(i)))
    }
}

/// Fills one item in process memory.
pub fn fill_owned(item: Box<dyn Item>, layout: &Layout) -> Result<OwnedBatch, String> {
    let mut owned = OwnedBatch::new(layout);
    let arrays = owned.arrays.clone();
    let mut batch = BatchMut::new(owned.buf_mut(), &arrays).ok_or("layout outside the item")?;
    item.fill(&mut batch)?;
    Ok(owned)
}

/// The next item of `source` for a request with `values` (the rest at
/// their defaults), outside any pool: for a loader's tests and tools.
pub fn next_detached(
    source: &mut dyn Source,
    values: Vec<(String, Value)>,
) -> Result<OwnedBatch, String> {
    let request = Params::resolve(&source.request_params(), values)?;
    let item = source.next(&request)?;
    let layout = Layout::of(&item.layout())?;
    fill_owned(item, &layout)
}

/// The next item of `source` with the default request.
pub fn fill_detached(source: &mut dyn Source) -> Result<OwnedBatch, String> {
    next_detached(source, Vec::new())
}

// ---------------------------------------------------------------------------
// A registration as runners drive it.

struct Turn {
    source: Box<dyn Source>,
    /// The ticket whose item is next.
    next: u64,
    /// Set once an item failed: the later ones fail too.
    failed: bool,
    requests: VecDeque<Params>,
}

/// One registration's source with the order its items are made in: item
/// `ticket` is made after item `ticket - 1`, with the `ticket`-th request
/// for a registration on request.
pub struct Stream {
    turn: Mutex<Turn>,
    turned: Condvar,
    request_specs: Vec<ParamSpec>,
    /// The request of every item of a registration kept ready; None on request.
    defaults: Option<Params>,
    max: Layout,
    resumable: bool,
    max_state: usize,
}

/// An item made by its stream's source, with the arrays it fills.
pub struct Prepared {
    item: Box<dyn Item>,
    state: Option<Vec<u8>>,
    layout: Layout,
}

impl Prepared {
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Fills the item and, for a resumable source, its [`RESUME_STATE`].
    pub fn fill(self, batch: &mut BatchMut<'_>) -> Result<(), String> {
        if let Some(state) = &self.state {
            let i = batch.index_of(RESUME_STATE).expect("in the layout");
            batch.bytes_mut(i).copy_from_slice(state);
        }
        self.item.fill(batch)
    }
}

impl Stream {
    /// `on_request`: items only as requests ask for them; otherwise every
    /// item with the default request, which needs every request parameter
    /// to have a default.
    pub fn new(source: Box<dyn Source>, on_request: bool) -> Result<Stream, String> {
        let request_specs = source.request_params();
        if request_specs.len() > proto::MAX_PARAMS {
            return Err("the loader declares too many request parameters".into());
        }
        let defaults = match on_request {
            true => None,
            false => Some(Params::resolve(&request_specs, Vec::new()).map_err(|e| {
                format!("this loader's items need requests (register on request): {e}")
            })?),
        };
        let resumable = source.state().is_some();
        let max_state = source.max_state_len().min(proto::MAX_RESUME);
        let mut max = source.max_layout();
        if max.iter().any(|a| a.name == RESUME_STATE) {
            return Err(format!(
                "{RESUME_STATE} is the runner's array, not a loader's"
            ));
        }
        if resumable {
            max.push(ArraySpec::new(RESUME_STATE, DType::U8, &[max_state as u64]));
        }
        let max = Layout::of(&max)?;
        Ok(Stream {
            turn: Mutex::new(Turn {
                source,
                next: 0,
                failed: false,
                requests: VecDeque::new(),
            }),
            turned: Condvar::new(),
            request_specs,
            defaults,
            max,
            resumable,
            max_state,
        })
    }

    pub fn request_params(&self) -> &[ParamSpec] {
        &self.request_specs
    }

    pub fn on_request(&self) -> bool {
        self.defaults.is_none()
    }

    /// What an item may take at most, the resume state included.
    pub fn max_layout(&self) -> &Layout {
        &self.max
    }

    /// Queues a request; the caller then asks its pool for one more item.
    pub fn push_request(&self, values: Vec<(String, Value)>) -> Result<(), String> {
        if self.defaults.is_some() {
            return Err("request for a registration that is kept ready".into());
        }
        let params = Params::resolve(&self.request_specs, values)?;
        let mut t = self.turn.lock().unwrap();
        if t.requests.len() >= MAX_QUEUED_REQUESTS {
            return Err(format!("more than {MAX_QUEUED_REQUESTS} requests waiting"));
        }
        t.requests.push_back(params);
        Ok(())
    }

    /// Item `ticket`, once the items before it are made.
    pub fn next(&self, ticket: u64) -> Result<Prepared, String> {
        let mut t = self.turn.lock().unwrap();
        while t.next != ticket && !t.failed {
            t = self.turned.wait(t).unwrap();
        }
        if t.failed {
            return Err("an earlier item of this registration failed".into());
        }
        // A loader panic must wake later tickets, too. Catch it while the
        // turn lock is held so neither poison nor a parked ticket survives.
        let made = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.make(&mut t)))
            .unwrap_or_else(|_| Err("source panicked while planning an item".into()));
        match made {
            Ok(_) => t.next += 1,
            Err(_) => t.failed = true,
        }
        self.turned.notify_all();
        made
    }

    fn make(&self, t: &mut Turn) -> Result<Prepared, String> {
        let request = match &self.defaults {
            Some(d) => d.clone(),
            None => t
                .requests
                .pop_front()
                .ok_or("an item claimed without a request")?,
        };
        let item = t.source.next(&request)?;
        let mut specs = item.layout();
        if specs.iter().any(|a| a.name == RESUME_STATE) {
            return Err(format!(
                "{RESUME_STATE} is the runner's array, not a loader's"
            ));
        }
        let state = match self.resumable {
            false => None,
            true => {
                let s = t
                    .source
                    .state()
                    .ok_or("a resumable source returned no state")?;
                if s.len() > self.max_state {
                    return Err(format!(
                        "the source's state is {} bytes, more than {}",
                        s.len(),
                        self.max_state
                    ));
                }
                specs.push(ArraySpec::new(RESUME_STATE, DType::U8, &[s.len() as u64]));
                Some(s)
            }
        };
        let layout = Layout::of(&specs)?;
        if layout.bytes > self.max.bytes {
            return Err(format!(
                "an item of {} bytes, the source's max_layout is {}",
                layout.bytes, self.max.bytes
            ));
        }
        Ok(Prepared {
            item,
            state,
            layout,
        })
    }
}

#[cfg(target_os = "linux")]
pub use local::LocalPool;

#[cfg(target_os = "linux")]
mod local {
    use super::*;
    use crate::core::{GetError, PreloadPool, RegistrationId};
    use std::path::Path;

    type Producer = Box<dyn Fn(RegistrationId, u64) -> Result<OwnedBatch, String> + Send + Sync>;

    /// A loader run in this process: registrations over data roots, filled
    /// by the pool's workers into process memory.
    pub struct LocalPool<L: Loader> {
        loader: L,
        roots: Vec<Root>,
        specs: Vec<ParamSpec>,
        streams: Vec<(usize, Arc<Stream>)>,
        pool: Option<PreloadPool<OwnedBatch, Producer>>,
    }

    impl<L: Loader> LocalPool<L> {
        /// `roots` are (name, directory): the only places inputs resolve.
        pub fn new(loader: L, roots: &[(String, &Path)]) -> Result<LocalPool<L>, String> {
            let mut opened = Vec::new();
            for (name, path) in roots {
                if !valid_name(name) {
                    return Err(format!("{name:?} is not a root name"));
                }
                opened.push(
                    Root::open(name, path)
                        .map_err(|e| format!("data root {name} ({}): {e}", path.display()))?,
                );
            }
            let specs = loader.params();
            Ok(LocalPool {
                loader,
                roots: opened,
                specs,
                streams: Vec::new(),
                pool: None,
            })
        }

        pub fn params(&self) -> &[ParamSpec] {
            &self.specs
        }

        fn root(&self, name: &str) -> Result<&Root, String> {
            self.roots
                .iter()
                .find(|r| r.name == name)
                .ok_or_else(|| format!("unknown data root {name:?}"))
        }

        /// One page of a directory beneath a root, as the bridge lists it.
        pub fn list(
            &self,
            root: &str,
            path: &str,
            offset: u32,
        ) -> Result<(Vec<DirEntry>, bool), String> {
            self.root(root)?
                .list(path, offset)
                .map_err(|e| e.to_string())
        }

        /// Registers a source kept ready: `files` are (root, path beneath it).
        pub fn register(
            &mut self,
            keep_ready: usize,
            values: Vec<(String, Value)>,
            files: &[(&str, &str)],
        ) -> Result<RegistrationId, String> {
            self.register_with(keep_ready, values, files, None, false)
        }

        /// Registers a source, from `resume` when given, on request or kept
        /// ready.
        pub fn register_with(
            &mut self,
            keep_ready: usize,
            values: Vec<(String, Value)>,
            files: &[(&str, &str)],
            resume: Option<&[u8]>,
            on_request: bool,
        ) -> Result<RegistrationId, String> {
            if self.pool.is_some() {
                return Err("register must happen before start".into());
            }
            if keep_ready == 0 {
                return Err("keep_ready must be greater than zero".into());
            }
            let params = Params::resolve(&self.specs, values)?;
            let (min, max) = self.loader.files();
            if files.len() < min as usize || files.len() > max as usize {
                return Err(format!(
                    "{} files given, the loader takes {min} to {max}",
                    files.len()
                ));
            }
            let takes_dirs = self.loader.takes_dirs();
            let mut opened = Vec::with_capacity(files.len());
            for (root, path) in files {
                let r = self.root(root)?;
                let input = match r.open_file(path) {
                    Ok(file) => InputFile {
                        root: root.to_string(),
                        path: path.to_string(),
                        file,
                        dir: None,
                    },
                    Err(PathError::NotFound(_)) if takes_dirs => {
                        let dir = r.open_dir(path).map_err(|e| e.to_string())?;
                        InputFile {
                            root: root.to_string(),
                            path: path.to_string(),
                            file: dir.as_file().map_err(|e| e.to_string())?,
                            dir: Some(dir),
                        }
                    }
                    Err(e) => return Err(e.to_string()),
                };
                opened.push(input);
            }
            let source = self.loader.open(&params, opened, resume)?;
            let stream = Stream::new(source, on_request)?;
            self.streams.push((keep_ready, Arc::new(stream)));
            Ok(self.streams.len() as RegistrationId - 1)
        }

        /// The parameters each request of a registration takes.
        pub fn request_params(&self, registration: RegistrationId) -> Option<&[ParamSpec]> {
            self.streams
                .get(registration as usize)
                .map(|(_, s)| s.request_params())
        }

        pub fn start(&mut self, workers: usize) -> Result<(), String> {
            if self.pool.is_some() {
                return Err("pool has already been started".into());
            }
            if self.streams.is_empty() {
                return Err("register at least one producer before start".into());
            }
            let jobs: Vec<Arc<Stream>> = self.streams.iter().map(|(_, s)| Arc::clone(s)).collect();
            let producer: Producer = Box::new(move |id, ticket| {
                let prepared = jobs[id as usize].next(ticket)?;
                let layout = prepared.layout().clone();
                let mut owned = OwnedBatch::new(&layout);
                let arrays = owned.arrays.clone();
                let mut batch =
                    BatchMut::new(owned.buf_mut(), &arrays).ok_or("layout outside the item")?;
                prepared.fill(&mut batch)?;
                Ok(owned)
            });
            let mut pool = PreloadPool::new(producer, workers)?;
            for (keep_ready, stream) in &self.streams {
                match stream.on_request() {
                    true => pool.register_on_request(*keep_ready)?,
                    false => pool.register(*keep_ready)?,
                };
            }
            pool.start()?;
            self.pool = Some(pool);
            Ok(())
        }

        /// Asks a registration on request for its next item.
        pub fn request(
            &self,
            registration: RegistrationId,
            values: Vec<(String, Value)>,
        ) -> Result<(), String> {
            let (Some(pool), Some((_, stream))) =
                (&self.pool, self.streams.get(registration as usize))
            else {
                return Err("not started, or no such registration".into());
            };
            stream.push_request(values)?;
            pool.request(registration, 1).map_err(str::to_owned)
        }

        pub fn get(&self, registration: RegistrationId) -> Result<OwnedBatch, GetError> {
            match &self.pool {
                Some(p) => p.get(registration),
                None => Err(GetError::Closing),
            }
        }

        pub fn ready_counts(&self) -> Vec<(RegistrationId, usize)> {
            match &self.pool {
                Some(p) => p.ready_counts(),
                None => (0..self.streams.len() as u64).map(|i| (i, 0)).collect(),
            }
        }

        pub fn close(&mut self) {
            if let Some(p) = self.pool.as_mut() {
                p.close();
            }
        }
    }
}
