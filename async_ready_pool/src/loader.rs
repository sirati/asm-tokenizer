//! The data-loader interface: a loader written once in Rust, run by this
//! crate's pool in-process ([`LocalPool`]), from Python ([`RustPreloader`]
//! with `loader=`), or in another process over the loader bridge
//! ([`crate::bridge`]; the `loader-bridge` crate serves it to a container).
//!
//! A [`Loader`] declares typed parameters and how many inputs a registration
//! takes, and opens a [`Source`] per registration. A source has a fixed
//! layout (named arrays with element types and shapes) and fills one item at
//! a time into a [`BatchMut`]. Inputs are files, or directories the loader
//! walks itself through [`DataDir`], always opened beneath the data roots the
//! runner was configured with; the loader never sees a path the consumer
//! chose that was not resolved that way.
//!
//! A source's `fill` must write every byte of every array: runners reuse
//! the memory of released items without clearing it.
//!
//! [`RustPreloader`]: crate#python

use std::collections::BTreeMap;
use std::fs::File;
use std::sync::Arc;

pub use crate::bridge::proto::{
    ArrayDesc, DType, DirEntry, EntryKind, ParamKind, ParamSpec, Value, valid_name,
};
#[cfg(target_os = "linux")]
pub use crate::fs::{DataDir, DataKind, PathError, Root};

use crate::bridge::proto;

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
    /// Prepares a registration. An error is reported to the consumer, which
    /// may try again.
    fn open(&self, params: &Params, files: Vec<InputFile>) -> Result<Box<dyn Source>, String>;
}

/// Produces the items of one registration, from the pool's worker threads,
/// concurrently.
pub trait Source: Send + Sync + 'static {
    /// The arrays of every item.
    fn layout(&self) -> Vec<ArraySpec>;
    /// Fills one item, every byte of every array. An error fails the
    /// registration: the consumer's next `get` for it raises this message.
    fn fill(&self, batch: &mut BatchMut<'_>) -> Result<(), String>;
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

/// Fills one item of `source` in process memory.
pub fn fill_owned(source: &dyn Source, layout: &Layout) -> Result<OwnedBatch, String> {
    let mut item = OwnedBatch::new(layout);
    let arrays = item.arrays.clone();
    let mut batch = BatchMut::new(item.buf_mut(), &arrays).ok_or("layout outside the item")?;
    source.fill(&mut batch)?;
    Ok(item)
}

/// One item of `source`, outside any pool: for a loader's tests and tools.
pub fn fill_detached(source: &dyn Source) -> Result<OwnedBatch, String> {
    let layout = Layout::of(&source.layout())?;
    fill_owned(source, &layout)
}

#[cfg(target_os = "linux")]
pub use local::LocalPool;

#[cfg(target_os = "linux")]
mod local {
    use super::*;
    use crate::core::{GetError, PreloadPool, RegistrationId};
    use std::path::Path;

    type Producer = Box<dyn Fn(RegistrationId) -> Result<OwnedBatch, String> + Send + Sync>;

    /// A loader run in this process: registrations over data roots, filled
    /// by the pool's workers into process memory.
    pub struct LocalPool<L: Loader> {
        loader: L,
        roots: Vec<Root>,
        specs: Vec<ParamSpec>,
        sources: Vec<(usize, Arc<dyn Source>, Layout)>,
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
                sources: Vec::new(),
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
            self.root(root)?.list(path, offset).map_err(|e| e.to_string())
        }

        /// Registers a source: `files` are (root, path beneath it).
        pub fn register(
            &mut self,
            keep_ready: usize,
            values: Vec<(String, Value)>,
            files: &[(&str, &str)],
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
            let source = self.loader.open(&params, opened)?;
            let layout = Layout::of(&source.layout())?;
            self.sources.push((keep_ready, Arc::from(source), layout));
            Ok(self.sources.len() as RegistrationId - 1)
        }

        pub fn start(&mut self, workers: usize) -> Result<(), String> {
            if self.pool.is_some() {
                return Err("pool has already been started".into());
            }
            if self.sources.is_empty() {
                return Err("register at least one producer before start".into());
            }
            let jobs: Vec<(Arc<dyn Source>, Layout)> = self
                .sources
                .iter()
                .map(|(_, s, l)| (Arc::clone(s), l.clone()))
                .collect();
            let producer: Producer = Box::new(move |id| {
                let (source, layout) = &jobs[id as usize];
                fill_owned(source.as_ref(), layout)
            });
            let mut pool = PreloadPool::new(producer, workers)?;
            for (keep_ready, _, _) in &self.sources {
                pool.register(*keep_ready)?;
            }
            pool.start()?;
            self.pool = Some(pool);
            Ok(())
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
                None => (0..self.sources.len() as u64).map(|i| (i, 0)).collect(),
            }
        }

        pub fn close(&mut self) {
            if let Some(p) = self.pool.as_mut() {
                p.close();
            }
        }
    }
}
