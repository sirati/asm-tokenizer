//! The wire protocol: one message per SOCK_SEQPACKET packet.
//!
//! ```text
//! header (16 bytes, little endian)
//!   magic   u32  "LBRG"
//!   version u16  VERSION
//!   kind    u16  message type
//!   request u64  chosen by the client; replies carry the request's
//! body: the message's fields in declaration order
//!   integers  little endian, fixed width
//!   string    u16 length + UTF-8 bytes
//!   list      u16 count + items
//!   bytes     u32 length + bytes
//!   option    u8 0|1 + value
//! ```
//!
//! Every length and count has a bound below, checked before anything is
//! allocated, and a body must be consumed exactly. Anything else is a
//! `DecodeError`, which on the host ends the session. Nothing here is generic
//! serialization: each message is a struct with its own encoder and decoder.

use std::fmt;

pub const MAGIC: u32 = u32::from_le_bytes(*b"LBRG");
pub const VERSION: u16 = 2;
pub const HEADER_LEN: usize = 16;
/// The largest packet either side sends or accepts.
pub const MAX_MESSAGE: usize = 64 * 1024;

/// Loader, root, parameter and array names.
pub const MAX_NAME: usize = 64;
/// A path relative to a data root.
pub const MAX_PATH: usize = 4096;
/// Free text: error messages.
pub const MAX_TEXT: usize = 4096;
pub const MAX_PARAMS: usize = 64;
pub const MAX_CHOICES: usize = 64;
pub const MAX_FILES: usize = 64;
pub const MAX_ROOTS: usize = 64;
pub const MAX_ARRAYS: usize = 32;
pub const MAX_DIMS: usize = 8;
/// Directory entries per `Listing` page; each is at most ~270 bytes.
pub const MAX_ENTRIES: usize = 128;
pub const MAX_REGISTRATIONS: usize = 1024;
/// Integers in one list parameter.
pub const MAX_INTS: usize = 4096;
/// A source's state to resume from, as a registration carries it.
pub const MAX_RESUME: usize = 48 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    Short,
    Oversized,
    BadMagic,
    BadVersion(u16),
    UnknownKind(u16),
    Truncated,
    Trailing,
    TooLong,
    TooMany,
    BadUtf8,
    BadName,
    BadTag,
    BadValue,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Short => write!(f, "message shorter than its header"),
            DecodeError::Oversized => write!(f, "message over {MAX_MESSAGE} bytes"),
            DecodeError::BadMagic => write!(f, "bad magic"),
            DecodeError::BadVersion(v) => write!(f, "protocol version {v}, expected {VERSION}"),
            DecodeError::UnknownKind(k) => write!(f, "unknown message kind {k:#x}"),
            DecodeError::Truncated => write!(f, "truncated message"),
            DecodeError::Trailing => write!(f, "trailing bytes after message"),
            DecodeError::TooLong => write!(f, "string over its length limit"),
            DecodeError::TooMany => write!(f, "list over its count limit"),
            DecodeError::BadUtf8 => write!(f, "string is not UTF-8"),
            DecodeError::BadName => write!(f, "malformed name"),
            DecodeError::BadTag => write!(f, "unknown tag"),
            DecodeError::BadValue => write!(f, "value out of range"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Encoding fails only when a value is over a limit the decoder would refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodeError(pub &'static str);

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot encode: {}", self.0)
    }
}

impl std::error::Error for EncodeError {}

/// `[A-Za-z0-9_.-]`, 1 to MAX_NAME bytes, not starting with a dot.
pub fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_NAME
        && !s.starts_with('.')
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.')
}

// ---------------------------------------------------------------------------
// Primitive reader and writer.

pub struct Reader<'a> {
    buf: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.buf.len() < n {
            return Err(DecodeError::Truncated);
        }
        let (head, rest) = self.buf.split_at(n);
        self.buf = rest;
        Ok(head)
    }

    pub fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }
    pub fn u16(&mut self) -> Result<u16, DecodeError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    pub fn u32(&mut self) -> Result<u32, DecodeError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> Result<u64, DecodeError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn i64(&mut self) -> Result<i64, DecodeError> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn f64(&mut self) -> Result<f64, DecodeError> {
        Ok(f64::from_bits(self.u64()?))
    }
    pub fn bool(&mut self) -> Result<bool, DecodeError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(DecodeError::BadTag),
        }
    }
    pub fn str(&mut self, max: usize) -> Result<String, DecodeError> {
        let len = self.u16()? as usize;
        if len > max {
            return Err(DecodeError::TooLong);
        }
        let bytes = self.take(len)?;
        let s = std::str::from_utf8(bytes).map_err(|_| DecodeError::BadUtf8)?;
        if s.contains('\0') {
            return Err(DecodeError::BadUtf8);
        }
        Ok(s.to_owned())
    }
    pub fn bytes(&mut self, max: usize) -> Result<Vec<u8>, DecodeError> {
        let len = self.u32()? as usize;
        if len > max {
            return Err(DecodeError::TooLong);
        }
        Ok(self.take(len)?.to_vec())
    }
    pub fn name(&mut self) -> Result<String, DecodeError> {
        let s = self.str(MAX_NAME)?;
        if !valid_name(&s) {
            return Err(DecodeError::BadName);
        }
        Ok(s)
    }
    /// A list count, checked against its limit and against the bytes left
    /// (each item takes at least `min_item` bytes) before anything is allocated.
    pub fn count(&mut self, max: usize, min_item: usize) -> Result<usize, DecodeError> {
        let n = self.u16()? as usize;
        if n > max {
            return Err(DecodeError::TooMany);
        }
        if n.saturating_mul(min_item) > self.buf.len() {
            return Err(DecodeError::Truncated);
        }
        Ok(n)
    }
    pub fn finish(self) -> Result<(), DecodeError> {
        if self.buf.is_empty() {
            Ok(())
        } else {
            Err(DecodeError::Trailing)
        }
    }
}

#[derive(Default)]
pub struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    pub fn bool(&mut self, v: bool) {
        self.u8(v as u8);
    }
    pub fn str(&mut self, s: &str, max: usize) -> Result<(), EncodeError> {
        if s.len() > max || s.contains('\0') {
            return Err(EncodeError("string over its limit"));
        }
        self.u16(s.len() as u16);
        self.buf.extend_from_slice(s.as_bytes());
        Ok(())
    }
    pub fn bytes(&mut self, b: &[u8], max: usize) -> Result<(), EncodeError> {
        if b.len() > max {
            return Err(EncodeError("bytes over their limit"));
        }
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
        Ok(())
    }
    pub fn name(&mut self, s: &str) -> Result<(), EncodeError> {
        if !valid_name(s) {
            return Err(EncodeError("malformed name"));
        }
        self.str(s, MAX_NAME)
    }
    pub fn count(&mut self, n: usize, max: usize) -> Result<(), EncodeError> {
        if n > max {
            return Err(EncodeError("list over its limit"));
        }
        self.u16(n as u16);
        Ok(())
    }
}

/// Free text is cut to MAX_TEXT at a character boundary, never refused.
pub fn clip_text(s: &str) -> String {
    let s = s.replace('\0', " ");
    if s.len() <= MAX_TEXT {
        return s;
    }
    let mut end = MAX_TEXT;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_owned()
}

// ---------------------------------------------------------------------------
// Shared value types.

/// Element types of the arrays a batch carries; numpy has each of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    U8 = 1,
    I8 = 2,
    U16 = 3,
    I16 = 4,
    U32 = 5,
    I32 = 6,
    U64 = 7,
    I64 = 8,
    F16 = 9,
    F32 = 10,
    F64 = 11,
    Bool = 12,
}

impl DType {
    pub const ALL: [DType; 12] = [
        DType::U8,
        DType::I8,
        DType::U16,
        DType::I16,
        DType::U32,
        DType::I32,
        DType::U64,
        DType::I64,
        DType::F16,
        DType::F32,
        DType::F64,
        DType::Bool,
    ];

    pub fn from_u8(v: u8) -> Option<DType> {
        DType::ALL.into_iter().find(|d| *d as u8 == v)
    }

    pub fn size(self) -> usize {
        match self {
            DType::U8 | DType::I8 | DType::Bool => 1,
            DType::U16 | DType::I16 | DType::F16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }

    /// numpy's array-interface type string.
    pub fn numpy(self) -> &'static str {
        match self {
            DType::U8 => "|u1",
            DType::I8 => "|i1",
            DType::U16 => "<u2",
            DType::I16 => "<i2",
            DType::U32 => "<u4",
            DType::I32 => "<i4",
            DType::U64 => "<u8",
            DType::I64 => "<i8",
            DType::F16 => "<f2",
            DType::F32 => "<f4",
            DType::F64 => "<f8",
            DType::Bool => "|b1",
        }
    }
}

/// A loading parameter's value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// One of a parameter's fixed choices.
    Choice(String),
    /// A list of integers.
    Ints(Vec<i64>),
}

impl Value {
    fn encode(&self, w: &mut Writer) -> Result<(), EncodeError> {
        match self {
            Value::Int(v) => {
                w.u8(1);
                w.i64(*v);
            }
            Value::Float(v) => {
                w.u8(2);
                w.f64(*v);
            }
            Value::Bool(v) => {
                w.u8(3);
                w.bool(*v);
            }
            Value::Choice(v) => {
                w.u8(4);
                w.name(v)?;
            }
            Value::Ints(v) => {
                w.u8(5);
                w.count(v.len(), MAX_INTS)?;
                for x in v {
                    w.i64(*x);
                }
            }
        }
        Ok(())
    }

    fn decode(r: &mut Reader<'_>) -> Result<Value, DecodeError> {
        Ok(match r.u8()? {
            1 => Value::Int(r.i64()?),
            2 => {
                let v = r.f64()?;
                if !v.is_finite() {
                    return Err(DecodeError::BadValue);
                }
                Value::Float(v)
            }
            3 => Value::Bool(r.bool()?),
            4 => Value::Choice(r.name()?),
            5 => {
                let n = r.count(MAX_INTS, 8)?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(r.i64()?);
                }
                Value::Ints(v)
            }
            _ => return Err(DecodeError::BadTag),
        })
    }
}

/// What a parameter accepts.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    Int {
        min: i64,
        max: i64,
    },
    Float {
        min: f64,
        max: f64,
    },
    Bool,
    Choice(Vec<String>),
    /// A list of at most `max_len` integers, each in `min..=max`.
    Ints {
        min: i64,
        max: i64,
        max_len: u16,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    pub kind: ParamKind,
    /// Used when a registration leaves the parameter out; None makes it required.
    pub default: Option<Value>,
}

impl ParamSpec {
    /// Whether `v` is a value this parameter accepts.
    pub fn accepts(&self, v: &Value) -> bool {
        match (&self.kind, v) {
            (ParamKind::Int { min, max }, Value::Int(x)) => min <= x && x <= max,
            (ParamKind::Float { min, max }, Value::Float(x)) => {
                x.is_finite() && min <= x && x <= max
            }
            (ParamKind::Bool, Value::Bool(_)) => true,
            (ParamKind::Choice(cs), Value::Choice(c)) => cs.iter().any(|x| x == c),
            (ParamKind::Ints { min, max, max_len }, Value::Ints(v)) => {
                v.len() <= *max_len as usize && v.iter().all(|x| min <= x && x <= max)
            }
            _ => false,
        }
    }

    fn encode(&self, w: &mut Writer) -> Result<(), EncodeError> {
        w.name(&self.name)?;
        match &self.kind {
            ParamKind::Int { min, max } => {
                w.u8(1);
                w.i64(*min);
                w.i64(*max);
            }
            ParamKind::Float { min, max } => {
                w.u8(2);
                w.f64(*min);
                w.f64(*max);
            }
            ParamKind::Bool => w.u8(3),
            ParamKind::Choice(cs) => {
                w.u8(4);
                w.count(cs.len(), MAX_CHOICES)?;
                for c in cs {
                    w.name(c)?;
                }
            }
            ParamKind::Ints { min, max, max_len } => {
                w.u8(5);
                w.i64(*min);
                w.i64(*max);
                w.u16(*max_len);
            }
        }
        match &self.default {
            None => w.u8(0),
            Some(v) => {
                w.u8(1);
                v.encode(w)?;
            }
        }
        Ok(())
    }

    fn decode(r: &mut Reader<'_>) -> Result<ParamSpec, DecodeError> {
        let name = r.name()?;
        let kind = match r.u8()? {
            1 => {
                let (min, max) = (r.i64()?, r.i64()?);
                if min > max {
                    return Err(DecodeError::BadValue);
                }
                ParamKind::Int { min, max }
            }
            2 => {
                let (min, max) = (r.f64()?, r.f64()?);
                if !(min.is_finite() && max.is_finite() && min <= max) {
                    return Err(DecodeError::BadValue);
                }
                ParamKind::Float { min, max }
            }
            3 => ParamKind::Bool,
            4 => {
                let n = r.count(MAX_CHOICES, 3)?;
                let mut cs = Vec::with_capacity(n);
                for _ in 0..n {
                    cs.push(r.name()?);
                }
                ParamKind::Choice(cs)
            }
            5 => {
                let (min, max, max_len) = (r.i64()?, r.i64()?, r.u16()?);
                if min > max || max_len as usize > MAX_INTS {
                    return Err(DecodeError::BadValue);
                }
                ParamKind::Ints { min, max, max_len }
            }
            _ => return Err(DecodeError::BadTag),
        };
        let default = match r.u8()? {
            0 => None,
            1 => Some(Value::decode(r)?),
            _ => return Err(DecodeError::BadTag),
        };
        Ok(ParamSpec {
            name,
            kind,
            default,
        })
    }
}

/// A file under one of the loader's data roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRef {
    pub root: String,
    pub path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntryKind {
    File = 1,
    Dir = 2,
    Symlink = 3,
    Other = 4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub kind: EntryKind,
    pub size: u64,
}

/// One array of a batch, at `offset` bytes into the batch's region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArrayDesc {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub offset: u64,
}

impl ArrayDesc {
    /// Its size in bytes; None on overflow.
    pub fn nbytes(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(self.dtype.size() as u64, |acc, d| acc.checked_mul(*d))
    }
}

/// A prepared item: a region of a shared segment and the arrays in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchDesc {
    /// Names the item for `Release`.
    pub batch: u64,
    pub registration: u64,
    pub segment: u32,
    pub offset: u64,
    pub len: u64,
    /// Set when this message introduces `segment`: its size, and the packet
    /// carries exactly one fd, the segment's.
    pub new_segment: Option<u64>,
    pub arrays: Vec<ArrayDesc>,
}

impl BatchDesc {
    /// Whether every array lies inside the region and none overlap the
    /// region's end, each aligned for its element type.
    pub fn arrays_in_bounds(&self) -> bool {
        self.arrays.iter().all(|a| {
            let Some(n) = a.nbytes() else { return false };
            a.offset % a.dtype.size() as u64 == 0
                && a.offset.checked_add(n).is_some_and(|end| end <= self.len)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum RefuseCode {
    /// Hello named a loader the host has not registered.
    UnknownLoader = 1,
    /// The client speaks another protocol version.
    BadVersion = 2,
    /// The file does not exist or is not a regular file / directory.
    NotFound = 3,
    /// The loader refused the registration (its own message).
    Loader = 4,
}

impl RefuseCode {
    fn from_u16(v: u16) -> Option<RefuseCode> {
        match v {
            1 => Some(RefuseCode::UnknownLoader),
            2 => Some(RefuseCode::BadVersion),
            3 => Some(RefuseCode::NotFound),
            4 => Some(RefuseCode::Loader),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Messages.

#[derive(Debug, Clone, PartialEq)]
pub enum ClientMsg {
    /// First message: which loader. The broker reads it; the session answers.
    Hello {
        loader: String,
    },
    /// One page of a directory under a root; `path` empty is the root itself.
    ListDir {
        root: String,
        path: String,
        offset: u32,
    },
    /// A registration: its ready depth, typed parameters and input files;
    /// the state to resume from (a `resume_state` an earlier item carried);
    /// whether its items are made on request (`Request`) or kept ready.
    Register {
        keep_ready: u32,
        params: Vec<(String, Value)>,
        files: Vec<FileRef>,
        resume: Option<Vec<u8>>,
        on_request: bool,
    },
    /// The next item of a registration on request, with its parameters.
    /// No reply; items come back to `Get`s in the order they were requested.
    Request {
        registration: u64,
        params: Vec<(String, Value)>,
    },
    Start {
        workers: u32,
    },
    Get {
        registration: u64,
    },
    /// The client no longer looks at a batch; the host may reuse its region.
    Release {
        batch: u64,
    },
    ReadyCounts,
    Close,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HostMsg {
    Welcome {
        loader: String,
        roots: Vec<String>,
        params: Vec<ParamSpec>,
        min_files: u16,
        max_files: u16,
        max_shared_bytes: u64,
        max_workers: u32,
    },
    /// A request that was refused without ending the session, or a refused
    /// Hello (the connection then closes).
    Refused {
        code: RefuseCode,
        message: String,
    },
    Listing {
        entries: Vec<DirEntry>,
        more: bool,
    },
    /// The registration's ID and the parameters each of its requests takes.
    Registered {
        registration: u64,
        request_params: Vec<ParamSpec>,
    },
    Started,
    Batch(BatchDesc),
    /// `get` could not return an item: the producer failed, or the pool closed.
    GetFailed {
        closing: bool,
        message: String,
    },
    ReadyCounts(Vec<(u64, u64)>),
    Closed,
    /// The session ends; sent best-effort just before the host closes.
    Fatal {
        message: String,
    },
}

mod kind {
    pub const HELLO: u16 = 0x01;
    pub const LIST_DIR: u16 = 0x02;
    pub const REGISTER: u16 = 0x03;
    pub const START: u16 = 0x04;
    pub const GET: u16 = 0x05;
    pub const RELEASE: u16 = 0x06;
    pub const READY_COUNTS: u16 = 0x07;
    pub const CLOSE: u16 = 0x08;
    pub const REQUEST: u16 = 0x09;

    pub const WELCOME: u16 = 0x81;
    pub const REFUSED: u16 = 0x82;
    pub const LISTING: u16 = 0x83;
    pub const REGISTERED: u16 = 0x84;
    pub const STARTED: u16 = 0x85;
    pub const BATCH: u16 = 0x86;
    pub const GET_FAILED: u16 = 0x87;
    pub const READY_COUNTS_REPLY: u16 = 0x88;
    pub const CLOSED: u16 = 0x89;
    pub const FATAL: u16 = 0x8a;
}

fn encode_values(w: &mut Writer, params: &[(String, Value)]) -> Result<(), EncodeError> {
    w.count(params.len(), MAX_PARAMS)?;
    for (name, value) in params {
        w.name(name)?;
        value.encode(w)?;
    }
    Ok(())
}

fn decode_values(r: &mut Reader<'_>) -> Result<Vec<(String, Value)>, DecodeError> {
    let n = r.count(MAX_PARAMS, 4)?;
    let mut params = Vec::with_capacity(n);
    for _ in 0..n {
        params.push((r.name()?, Value::decode(r)?));
    }
    Ok(params)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub version: u16,
    pub kind: u16,
    pub request: u64,
}

/// Splits a packet into its header and body. The version is returned, not
/// checked: the broker answers a wrong one with `Refused`.
pub fn split(packet: &[u8]) -> Result<(Header, &[u8]), DecodeError> {
    if packet.len() > MAX_MESSAGE {
        return Err(DecodeError::Oversized);
    }
    if packet.len() < HEADER_LEN {
        return Err(DecodeError::Short);
    }
    let mut r = Reader::new(&packet[..HEADER_LEN]);
    if r.u32()? != MAGIC {
        return Err(DecodeError::BadMagic);
    }
    let header = Header {
        version: r.u16()?,
        kind: r.u16()?,
        request: r.u64()?,
    };
    Ok((header, &packet[HEADER_LEN..]))
}

fn frame(kind: u16, request: u64, body: Writer) -> Result<Vec<u8>, EncodeError> {
    let mut out = Vec::with_capacity(HEADER_LEN + body.buf.len());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&request.to_le_bytes());
    out.extend_from_slice(&body.buf);
    if out.len() > MAX_MESSAGE {
        return Err(EncodeError("message over MAX_MESSAGE"));
    }
    Ok(out)
}

fn check_version(h: &Header) -> Result<(), DecodeError> {
    if h.version != VERSION {
        return Err(DecodeError::BadVersion(h.version));
    }
    Ok(())
}

impl ClientMsg {
    pub fn encode(&self, request: u64) -> Result<Vec<u8>, EncodeError> {
        let mut w = Writer::default();
        let kind = match self {
            ClientMsg::Hello { loader } => {
                w.name(loader)?;
                kind::HELLO
            }
            ClientMsg::ListDir { root, path, offset } => {
                w.name(root)?;
                w.str(path, MAX_PATH)?;
                w.u32(*offset);
                kind::LIST_DIR
            }
            ClientMsg::Register {
                keep_ready,
                params,
                files,
                resume,
                on_request,
            } => {
                w.u32(*keep_ready);
                encode_values(&mut w, params)?;
                w.count(files.len(), MAX_FILES)?;
                for f in files {
                    w.name(&f.root)?;
                    w.str(&f.path, MAX_PATH)?;
                }
                match resume {
                    None => w.u8(0),
                    Some(b) => {
                        w.u8(1);
                        w.bytes(b, MAX_RESUME)?;
                    }
                }
                w.bool(*on_request);
                kind::REGISTER
            }
            ClientMsg::Request {
                registration,
                params,
            } => {
                w.u64(*registration);
                encode_values(&mut w, params)?;
                kind::REQUEST
            }
            ClientMsg::Start { workers } => {
                w.u32(*workers);
                kind::START
            }
            ClientMsg::Get { registration } => {
                w.u64(*registration);
                kind::GET
            }
            ClientMsg::Release { batch } => {
                w.u64(*batch);
                kind::RELEASE
            }
            ClientMsg::ReadyCounts => kind::READY_COUNTS,
            ClientMsg::Close => kind::CLOSE,
        };
        frame(kind, request, w)
    }

    /// Decodes a whole packet; the version must be VERSION.
    pub fn decode(packet: &[u8]) -> Result<(u64, ClientMsg), DecodeError> {
        let (h, body) = split(packet)?;
        check_version(&h)?;
        Ok((h.request, ClientMsg::decode_body(h.kind, body)?))
    }

    pub fn decode_body(kind: u16, body: &[u8]) -> Result<ClientMsg, DecodeError> {
        let mut r = Reader::new(body);
        let msg = match kind {
            kind::HELLO => ClientMsg::Hello { loader: r.name()? },
            kind::LIST_DIR => ClientMsg::ListDir {
                root: r.name()?,
                path: r.str(MAX_PATH)?,
                offset: r.u32()?,
            },
            kind::REGISTER => {
                let keep_ready = r.u32()?;
                let params = decode_values(&mut r)?;
                let n = r.count(MAX_FILES, 5)?;
                let mut files = Vec::with_capacity(n);
                for _ in 0..n {
                    files.push(FileRef {
                        root: r.name()?,
                        path: r.str(MAX_PATH)?,
                    });
                }
                let resume = match r.u8()? {
                    0 => None,
                    1 => Some(r.bytes(MAX_RESUME)?),
                    _ => return Err(DecodeError::BadTag),
                };
                ClientMsg::Register {
                    keep_ready,
                    params,
                    files,
                    resume,
                    on_request: r.bool()?,
                }
            }
            kind::REQUEST => ClientMsg::Request {
                registration: r.u64()?,
                params: decode_values(&mut r)?,
            },
            kind::START => ClientMsg::Start { workers: r.u32()? },
            kind::GET => ClientMsg::Get {
                registration: r.u64()?,
            },
            kind::RELEASE => ClientMsg::Release { batch: r.u64()? },
            kind::READY_COUNTS => ClientMsg::ReadyCounts,
            kind::CLOSE => ClientMsg::Close,
            other => return Err(DecodeError::UnknownKind(other)),
        };
        r.finish()?;
        Ok(msg)
    }
}

impl HostMsg {
    pub fn encode(&self, request: u64) -> Result<Vec<u8>, EncodeError> {
        let mut w = Writer::default();
        let kind = match self {
            HostMsg::Welcome {
                loader,
                roots,
                params,
                min_files,
                max_files,
                max_shared_bytes,
                max_workers,
            } => {
                w.name(loader)?;
                w.count(roots.len(), MAX_ROOTS)?;
                for r in roots {
                    w.name(r)?;
                }
                w.count(params.len(), MAX_PARAMS)?;
                for p in params {
                    p.encode(&mut w)?;
                }
                w.u16(*min_files);
                w.u16(*max_files);
                w.u64(*max_shared_bytes);
                w.u32(*max_workers);
                kind::WELCOME
            }
            HostMsg::Refused { code, message } => {
                w.u16(*code as u16);
                w.str(&clip_text(message), MAX_TEXT)?;
                kind::REFUSED
            }
            HostMsg::Listing { entries, more } => {
                w.count(entries.len(), MAX_ENTRIES)?;
                for e in entries {
                    w.str(&e.name, 255)?;
                    w.u8(e.kind as u8);
                    w.u64(e.size);
                }
                w.bool(*more);
                kind::LISTING
            }
            HostMsg::Registered {
                registration,
                request_params,
            } => {
                w.u64(*registration);
                w.count(request_params.len(), MAX_PARAMS)?;
                for p in request_params {
                    p.encode(&mut w)?;
                }
                kind::REGISTERED
            }
            HostMsg::Started => kind::STARTED,
            HostMsg::Batch(b) => {
                w.u64(b.batch);
                w.u64(b.registration);
                w.u32(b.segment);
                w.u64(b.offset);
                w.u64(b.len);
                match b.new_segment {
                    None => w.u8(0),
                    Some(size) => {
                        w.u8(1);
                        w.u64(size);
                    }
                }
                w.count(b.arrays.len(), MAX_ARRAYS)?;
                for a in &b.arrays {
                    w.name(&a.name)?;
                    w.u8(a.dtype as u8);
                    w.count(a.shape.len(), MAX_DIMS)?;
                    for d in &a.shape {
                        w.u64(*d);
                    }
                    w.u64(a.offset);
                }
                kind::BATCH
            }
            HostMsg::GetFailed { closing, message } => {
                w.bool(*closing);
                w.str(&clip_text(message), MAX_TEXT)?;
                kind::GET_FAILED
            }
            HostMsg::ReadyCounts(counts) => {
                w.count(counts.len(), MAX_REGISTRATIONS)?;
                for (id, n) in counts {
                    w.u64(*id);
                    w.u64(*n);
                }
                kind::READY_COUNTS_REPLY
            }
            HostMsg::Closed => kind::CLOSED,
            HostMsg::Fatal { message } => {
                w.str(&clip_text(message), MAX_TEXT)?;
                kind::FATAL
            }
        };
        frame(kind, request, w)
    }

    pub fn decode(packet: &[u8]) -> Result<(u64, HostMsg), DecodeError> {
        let (h, body) = split(packet)?;
        check_version(&h)?;
        Ok((h.request, HostMsg::decode_body(h.kind, body)?))
    }

    pub fn decode_body(kind: u16, body: &[u8]) -> Result<HostMsg, DecodeError> {
        let mut r = Reader::new(body);
        let msg = match kind {
            kind::WELCOME => {
                let loader = r.name()?;
                let n = r.count(MAX_ROOTS, 3)?;
                let mut roots = Vec::with_capacity(n);
                for _ in 0..n {
                    roots.push(r.name()?);
                }
                let n = r.count(MAX_PARAMS, 5)?;
                let mut params = Vec::with_capacity(n);
                for _ in 0..n {
                    params.push(ParamSpec::decode(&mut r)?);
                }
                let min_files = r.u16()?;
                let max_files = r.u16()?;
                if min_files > max_files || max_files as usize > MAX_FILES {
                    return Err(DecodeError::BadValue);
                }
                HostMsg::Welcome {
                    loader,
                    roots,
                    params,
                    min_files,
                    max_files,
                    max_shared_bytes: r.u64()?,
                    max_workers: r.u32()?,
                }
            }
            kind::REFUSED => {
                let code = RefuseCode::from_u16(r.u16()?).ok_or(DecodeError::BadTag)?;
                HostMsg::Refused {
                    code,
                    message: r.str(MAX_TEXT)?,
                }
            }
            kind::LISTING => {
                let n = r.count(MAX_ENTRIES, 11)?;
                let mut entries = Vec::with_capacity(n);
                for _ in 0..n {
                    let name = r.str(255)?;
                    if name.is_empty() || name.contains('/') || name == "." || name == ".." {
                        return Err(DecodeError::BadName);
                    }
                    let kind = match r.u8()? {
                        1 => EntryKind::File,
                        2 => EntryKind::Dir,
                        3 => EntryKind::Symlink,
                        4 => EntryKind::Other,
                        _ => return Err(DecodeError::BadTag),
                    };
                    entries.push(DirEntry {
                        name,
                        kind,
                        size: r.u64()?,
                    });
                }
                HostMsg::Listing {
                    entries,
                    more: r.bool()?,
                }
            }
            kind::REGISTERED => {
                let registration = r.u64()?;
                let n = r.count(MAX_PARAMS, 5)?;
                let mut request_params = Vec::with_capacity(n);
                for _ in 0..n {
                    request_params.push(ParamSpec::decode(&mut r)?);
                }
                HostMsg::Registered {
                    registration,
                    request_params,
                }
            }
            kind::STARTED => HostMsg::Started,
            kind::BATCH => {
                let batch = r.u64()?;
                let registration = r.u64()?;
                let segment = r.u32()?;
                let offset = r.u64()?;
                let len = r.u64()?;
                let new_segment = match r.u8()? {
                    0 => None,
                    1 => Some(r.u64()?),
                    _ => return Err(DecodeError::BadTag),
                };
                let n = r.count(MAX_ARRAYS, 14)?;
                let mut arrays = Vec::with_capacity(n);
                for _ in 0..n {
                    let name = r.name()?;
                    let dtype = DType::from_u8(r.u8()?).ok_or(DecodeError::BadTag)?;
                    let dims = r.count(MAX_DIMS, 8)?;
                    let mut shape = Vec::with_capacity(dims);
                    for _ in 0..dims {
                        shape.push(r.u64()?);
                    }
                    arrays.push(ArrayDesc {
                        name,
                        dtype,
                        shape,
                        offset: r.u64()?,
                    });
                }
                let desc = BatchDesc {
                    batch,
                    registration,
                    segment,
                    offset,
                    len,
                    new_segment,
                    arrays,
                };
                if offset.checked_add(len).is_none() || !desc.arrays_in_bounds() {
                    return Err(DecodeError::BadValue);
                }
                HostMsg::Batch(desc)
            }
            kind::GET_FAILED => HostMsg::GetFailed {
                closing: r.bool()?,
                message: r.str(MAX_TEXT)?,
            },
            kind::READY_COUNTS_REPLY => {
                let n = r.count(MAX_REGISTRATIONS, 16)?;
                let mut counts = Vec::with_capacity(n);
                for _ in 0..n {
                    counts.push((r.u64()?, r.u64()?));
                }
                HostMsg::ReadyCounts(counts)
            }
            kind::CLOSED => HostMsg::Closed,
            kind::FATAL => HostMsg::Fatal {
                message: r.str(MAX_TEXT)?,
            },
            other => return Err(DecodeError::UnknownKind(other)),
        };
        r.finish()?;
        Ok(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_round_trip() {
        let p = ClientMsg::Hello {
            loader: "tokens".into(),
        }
        .encode(7)
        .unwrap();
        assert_eq!(
            ClientMsg::decode(&p).unwrap(),
            (
                7,
                ClientMsg::Hello {
                    loader: "tokens".into()
                }
            )
        );
    }

    #[test]
    fn version_is_checked() {
        let mut p = ClientMsg::Close.encode(1).unwrap();
        p[4] = 9;
        assert_eq!(ClientMsg::decode(&p), Err(DecodeError::BadVersion(9)));
    }

    #[test]
    fn names_are_restricted() {
        assert!(valid_name("tok-ens_1.u16"));
        for bad in ["", ".x", "a/b", "a b", "ä", &"x".repeat(65)] {
            assert!(!valid_name(bad), "{bad:?}");
        }
    }

    #[test]
    fn counts_are_checked_before_allocation() {
        // Register with a param count of 64 but no bytes behind it.
        let mut p = ClientMsg::Register {
            keep_ready: 1,
            params: vec![],
            files: vec![],
            resume: None,
            on_request: false,
        }
        .encode(1)
        .unwrap();
        let at = HEADER_LEN + 4;
        p[at] = 64;
        assert_eq!(ClientMsg::decode(&p), Err(DecodeError::Truncated));
        p[at] = 65;
        assert_eq!(ClientMsg::decode(&p), Err(DecodeError::TooMany));
    }
}
