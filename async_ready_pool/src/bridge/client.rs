//! The client, in the container: connects, sends requests, maps the segments
//! it is given read-only and hands out batches as views into them.
//!
//! The host is checked too: a reply that does not decode, a segment that is
//! not sealed or not the size announced, or arrays outside their region end
//! the connection here rather than becoming a bad pointer.

use std::collections::HashMap;
use std::fmt;
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::bridge::proto::{
    self, ArrayDesc, BatchDesc, ClientMsg, DType, DirEntry, FileRef, HostMsg, ParamSpec,
    RefuseCode, Value,
};
use crate::bridge::SEALS;
use crate::bridge::wire;

pub const DEFAULT_SOCKET: &str = "/run/loader-bridge/sock";

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The host does not serve a loader by that name.
    UnknownLoader(String),
    /// A request was refused, the session goes on.
    Refused(RefuseCode, String),
    /// The producer of a registration failed.
    ProducerFailed(String),
    /// The pool closed while waiting.
    Closing,
    /// The session is over: the host ended it (its reason), or the
    /// connection broke.
    Disconnected(String),
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnknownLoader(m) => write!(f, "unknown loader: {m}"),
            Error::Refused(_, m) => write!(f, "{m}"),
            Error::ProducerFailed(m) => write!(f, "{m}"),
            Error::Closing => write!(f, "preloader is closed or not started"),
            Error::Disconnected(m) => write!(f, "loader-bridge session ended: {m}"),
            Error::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

/// A read-only mapping of one segment.
struct Mapping {
    ptr: *const u8,
    size: u64,
    /// The segment's read-only fd, held as long as the mapping.
    _fd: OwnedFd,
}

unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    fn new(fd: OwnedFd, size: u64) -> Result<Mapping, String> {
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        if unsafe { libc::fstat(fd.as_raw_fd(), &mut st) } != 0
            || st.st_size as u64 != size
            || size == 0
        {
            return Err("segment is not the size announced".into());
        }
        let seals = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 || seals & SEALS != SEALS {
            return Err("segment is not sealed".into());
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size as usize,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap: {}", std::io::Error::last_os_error()));
        }
        Ok(Mapping {
            ptr: ptr as *const u8,
            size,
            _fd: fd,
        })
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut _, self.size as usize) };
    }
}

enum Reply {
    Msg(HostMsg),
    Batch(Batch),
}

struct Shared {
    conn: OwnedFd,
    send_lock: Mutex<()>,
    next_request: AtomicU64,
    pending: Mutex<HashMap<u64, mpsc::Sender<Reply>>>,
    /// Why the session ended, once it has.
    dead: Mutex<Option<String>>,
}

impl Shared {
    fn send(&self, request: u64, msg: &ClientMsg) -> Result<(), Error> {
        let packet = msg.encode(request).map_err(|e| Error::Io(e.to_string()))?;
        let _g = self.send_lock.lock().unwrap();
        wire::send(self.conn.as_fd(), &packet, &[]).map_err(|e| self.disconnected(&e.to_string()))
    }

    fn disconnected(&self, fallback: &str) -> Error {
        Error::Disconnected(
            self.dead
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| fallback.to_owned()),
        )
    }

    fn die(&self, why: String) {
        let mut dead = self.dead.lock().unwrap();
        if dead.is_none() {
            *dead = Some(why);
        }
        drop(dead);
        // Dropping the senders wakes every waiter.
        self.pending.lock().unwrap().clear();
        unsafe { libc::shutdown(self.conn.as_raw_fd(), libc::SHUT_RDWR) };
    }
}

/// What the host said about the loader at connect.
#[derive(Debug, Clone)]
pub struct LoaderInfo {
    pub loader: String,
    pub roots: Vec<String>,
    pub params: Vec<ParamSpec>,
    pub min_files: u16,
    pub max_files: u16,
    pub max_shared_bytes: u64,
    pub max_workers: u32,
}

/// One session with a loader on the host.
pub struct Client {
    shared: Arc<Shared>,
    info: LoaderInfo,
    receiver: Option<std::thread::JoinHandle<()>>,
    /// The loader process of a local session ([`Client::spawn_local`]).
    child: Mutex<Option<std::process::Child>>,
}

impl Client {
    pub fn connect(socket: &Path, loader: &str) -> Result<Client, Error> {
        if !proto::valid_name(loader) {
            return Err(Error::UnknownLoader(format!(
                "{loader:?} is not a loader name"
            )));
        }
        let conn = wire::connect(socket)
            .map_err(|e| Error::Io(format!("connect {}: {e}", socket.display())))?;
        let hello = ClientMsg::Hello {
            loader: loader.to_owned(),
        }
        .encode(0)
        .unwrap();
        wire::send(conn.as_fd(), &hello, &[]).map_err(|e| Error::Io(format!("send: {e}")))?;
        Client::welcome(conn, None)
    }

    /// Starts `program`, a loader binary, as a child serving one session on
    /// a private socket pair, with `roots` as its data roots: the bridge
    /// protocol and its checks without a broker or a sandbox, for running a
    /// loader on the machine the data is on. The child gets an empty
    /// environment and dies with this client (and with this process).
    pub fn spawn_local(
        program: &Path,
        loader: &str,
        roots: &[(String, std::path::PathBuf)],
        max_shared_bytes: u64,
        max_workers: u32,
    ) -> Result<Client, Error> {
        use std::os::unix::process::CommandExt;
        if !proto::valid_name(loader) || roots.iter().any(|(n, _)| !proto::valid_name(n)) {
            return Err(Error::UnknownLoader(format!(
                "{loader:?} or one of its root names is not a valid name"
            )));
        }
        let config = crate::bridge::SessionConfig {
            request: 0,
            loader: loader.to_owned(),
            max_shared_bytes,
            max_workers: max_workers.max(1),
            roots: roots.to_vec(),
        };
        let (ours, theirs) = wire::pair().map_err(|e| Error::Io(format!("socketpair: {e}")))?;
        let child_fd = theirs.as_raw_fd();
        let mut cmd = std::process::Command::new(program);
        cmd.args(config.to_args())
            .env_clear()
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null());
        // Between fork and exec: only async-signal-safe calls.
        unsafe {
            cmd.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if child_fd == crate::bridge::CONN_FD {
                    // dup2 onto itself keeps CLOEXEC: clear it.
                    let flags = libc::fcntl(child_fd, libc::F_GETFD);
                    if flags < 0
                        || libc::fcntl(child_fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0
                    {
                        return Err(std::io::Error::last_os_error());
                    }
                } else if libc::dup2(child_fd, crate::bridge::CONN_FD) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = cmd
            .spawn()
            .map_err(|e| Error::Io(format!("start {}: {e}", program.display())))?;
        drop(theirs);
        let client = Client::welcome(ours, Some(child))?;
        if client.info.loader != loader {
            client.shared.die("wrong loader".into());
            return Err(Error::UnknownLoader(format!(
                "{} serves {:?}, not {loader:?}",
                program.display(),
                client.info.loader
            )));
        }
        Ok(client)
    }

    /// Reads the Welcome (request 0) on `conn` and starts the receiver.
    fn welcome(conn: OwnedFd, child: Option<std::process::Child>) -> Result<Client, Error> {
        let info = match read_welcome(&conn) {
            Ok(info) => info,
            Err(e) => {
                if let Some(mut c) = child {
                    let _ = c.kill();
                    let _ = c.wait();
                }
                return Err(e);
            }
        };
        let shared = Arc::new(Shared {
            conn,
            send_lock: Mutex::new(()),
            next_request: AtomicU64::new(1),
            pending: Mutex::new(HashMap::new()),
            dead: Mutex::new(None),
        });
        let s = Arc::clone(&shared);
        let receiver = std::thread::spawn(move || receive_loop(s));
        Ok(Client {
            shared,
            info,
            receiver: Some(receiver),
            child: Mutex::new(child),
        })
    }
}

fn read_welcome(conn: &OwnedFd) -> Result<LoaderInfo, Error> {
    {
        let packet = match wire::recv(conn.as_fd()) {
            Ok(Some(p)) if p.is_clean() => p,
            Ok(Some(_)) => {
                return Err(Error::Disconnected(
                    "unexpected fds in the handshake".into(),
                ))
            }
            Ok(None) => return Err(Error::Disconnected("the host closed the connection".into())),
            Err(e) => return Err(Error::Io(format!("receive: {e}"))),
        };
        let info = match HostMsg::decode(&packet.data) {
            Ok((
                0,
                HostMsg::Welcome {
                    loader,
                    roots,
                    params,
                    min_files,
                    max_files,
                    max_shared_bytes,
                    max_workers,
                },
            )) => LoaderInfo {
                loader,
                roots,
                params,
                min_files,
                max_files,
                max_shared_bytes,
                max_workers,
            },
            Ok((
                _,
                HostMsg::Refused {
                    code: RefuseCode::UnknownLoader,
                    message,
                },
            )) => return Err(Error::UnknownLoader(message)),
            Ok((_, HostMsg::Refused { code, message })) => {
                return Err(Error::Refused(code, message))
            }
            Ok((_, HostMsg::Fatal { message })) => return Err(Error::Disconnected(message)),
            Ok(_) => return Err(Error::Disconnected("unexpected handshake reply".into())),
            Err(e) => {
                return Err(Error::Disconnected(format!(
                    "malformed handshake reply: {e}"
                )))
            }
        };
        Ok(info)
    }
}

impl Client {

    pub fn info(&self) -> &LoaderInfo {
        &self.info
    }

    fn request(&self, msg: &ClientMsg) -> Result<Reply, Error> {
        let id = self.shared.next_request.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel();
        {
            // Registered under the lock die() clears, so a waiter is either
            // seen by die() or sees dead itself.
            let mut pending = self.shared.pending.lock().unwrap();
            if self.shared.dead.lock().unwrap().is_some() {
                return Err(self.shared.disconnected(""));
            }
            pending.insert(id, tx);
        }
        self.shared.send(id, msg)?;
        rx.recv()
            .map_err(|_| self.shared.disconnected("connection closed"))
    }

    fn expect(&self, msg: &ClientMsg) -> Result<HostMsg, Error> {
        match self.request(msg)? {
            Reply::Msg(HostMsg::Refused { code, message }) => Err(Error::Refused(code, message)),
            Reply::Msg(m) => Ok(m),
            Reply::Batch(_) => Err(Error::Disconnected("unexpected batch".into())),
        }
    }

    /// One page of a directory under a data root.
    pub fn list_dir(
        &self,
        root: &str,
        path: &str,
        offset: u32,
    ) -> Result<(Vec<DirEntry>, bool), Error> {
        match self.expect(&ClientMsg::ListDir {
            root: root.into(),
            path: path.into(),
            offset,
        })? {
            HostMsg::Listing { entries, more } => Ok((entries, more)),
            _ => Err(Error::Disconnected("unexpected reply".into())),
        }
    }

    pub fn register(
        &self,
        keep_ready: u32,
        params: Vec<(String, Value)>,
        files: Vec<FileRef>,
    ) -> Result<u64, Error> {
        match self.expect(&ClientMsg::Register {
            keep_ready,
            params,
            files,
        })? {
            HostMsg::Registered { registration } => Ok(registration),
            _ => Err(Error::Disconnected("unexpected reply".into())),
        }
    }

    pub fn start(&self, workers: u32) -> Result<(), Error> {
        match self.expect(&ClientMsg::Start { workers })? {
            HostMsg::Started => Ok(()),
            _ => Err(Error::Disconnected("unexpected reply".into())),
        }
    }

    /// The next item of a registration; blocks until one is ready.
    pub fn get(&self, registration: u64) -> Result<Batch, Error> {
        match self.request(&ClientMsg::Get { registration })? {
            Reply::Batch(b) if b.registration == registration => Ok(b),
            Reply::Msg(HostMsg::GetFailed {
                closing: false,
                message,
            }) => Err(Error::ProducerFailed(message)),
            Reply::Msg(HostMsg::GetFailed { closing: true, .. }) => Err(Error::Closing),
            _ => Err(Error::Disconnected("unexpected reply".into())),
        }
    }

    pub fn ready_counts(&self) -> Result<Vec<(u64, u64)>, Error> {
        match self.expect(&ClientMsg::ReadyCounts)? {
            HostMsg::ReadyCounts(c) => Ok(c),
            _ => Err(Error::Disconnected("unexpected reply".into())),
        }
    }

    /// Ends the session. Batches already taken stay readable.
    pub fn close(&self) -> Result<(), Error> {
        let r = match self.expect(&ClientMsg::Close) {
            Ok(HostMsg::Closed) => Ok(()),
            Ok(_) => Err(Error::Disconnected("unexpected reply".into())),
            Err(e) => Err(e),
        };
        self.shared.die("closed".into());
        r
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.shared.die("closed".into());
        if let Some(h) = self.receiver.take() {
            let _ = h.join();
        }
        // A local loader exits when its connection closes; one that does
        // not within a second is killed.
        if let Some(mut c) = self.child.lock().unwrap().take() {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
            loop {
                match c.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(std::time::Duration::from_millis(5))
                    }
                    _ => {
                        let _ = c.kill();
                        let _ = c.wait();
                        break;
                    }
                }
            }
        }
    }
}

fn receive_loop(shared: Arc<Shared>) {
    let mut segments: HashMap<u32, Arc<Mapping>> = HashMap::new();
    let why = loop {
        let packet = match wire::recv(shared.conn.as_fd()) {
            Ok(Some(p)) => p,
            Ok(None) => break "the host closed the connection".to_owned(),
            Err(e) => break format!("receive: {e}"),
        };
        if packet.truncated
            || packet.ctrunc
            || packet.foreign_cmsg
            || packet.fds.len() > wire::MAX_FDS
        {
            break "malformed packet from the host".into();
        }
        let (request, msg) = match HostMsg::decode(&packet.data) {
            Ok(m) => m,
            Err(e) => break format!("malformed message from the host: {e}"),
        };
        let reply = match msg {
            HostMsg::Fatal { message } => break message,
            HostMsg::Batch(desc) => {
                let mut fds = packet.fds;
                match accept_batch(&shared, &mut segments, desc, fds.pop()) {
                    Ok(b) => Reply::Batch(b),
                    Err(e) => break e,
                }
            }
            HostMsg::Welcome { .. } => break "a second Welcome".into(),
            other => {
                if !packet.fds.is_empty() {
                    break "fds on a message that carries none".into();
                }
                Reply::Msg(other)
            }
        };
        let Some(tx) = shared.pending.lock().unwrap().remove(&request) else {
            break format!("reply to unknown request {request}");
        };
        let _ = tx.send(reply);
    };
    shared.die(why);
}

fn accept_batch(
    shared: &Arc<Shared>,
    segments: &mut HashMap<u32, Arc<Mapping>>,
    desc: BatchDesc,
    fd: Option<OwnedFd>,
) -> Result<Batch, String> {
    match (desc.new_segment, fd) {
        (Some(size), Some(fd)) => {
            if segments.contains_key(&desc.segment) {
                return Err("segment announced twice".into());
            }
            segments.insert(desc.segment, Arc::new(Mapping::new(fd, size)?));
        }
        (None, None) => {}
        _ => return Err("segment announcement and fds disagree".into()),
    }
    let map = segments
        .get(&desc.segment)
        .ok_or("batch in an unknown segment")?
        .clone();
    let end = desc
        .offset
        .checked_add(desc.len)
        .ok_or("region overflows")?;
    if end > map.size || !desc.arrays_in_bounds() {
        return Err("batch outside its segment".into());
    }
    let lease = Arc::new(BatchLease {
        shared: Arc::downgrade(shared),
        batch: desc.batch,
        map,
        base: desc.offset,
    });
    Ok(Batch {
        registration: desc.registration,
        arrays: desc.arrays,
        lease,
    })
}

/// Keeps a batch's region mapped and, once dropped, tells the host it may
/// reuse it.
pub struct BatchLease {
    shared: std::sync::Weak<Shared>,
    batch: u64,
    map: Arc<Mapping>,
    base: u64,
}

impl Drop for BatchLease {
    fn drop(&mut self) {
        if let Some(s) = self.shared.upgrade() {
            if s.dead.lock().unwrap().is_none() {
                let _ = s.send(0, &ClientMsg::Release { batch: self.batch });
            }
        }
    }
}

/// One item: named arrays in shared memory.
pub struct Batch {
    pub registration: u64,
    arrays: Vec<ArrayDesc>,
    lease: Arc<BatchLease>,
}

/// One array of a batch, read-only. Holds the batch's lease.
#[derive(Clone)]
pub struct ArrayView {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    ptr: *const u8,
    len: usize,
    lease: Arc<BatchLease>,
}

unsafe impl Send for ArrayView {}
unsafe impl Sync for ArrayView {}

impl ArrayView {
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    pub fn len_bytes(&self) -> usize {
        self.len
    }
    /// The bytes. The host may still write into a region the client no
    /// longer holds, never into one it does.
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    /// The lease that keeps this array valid; dropping the last one releases the batch.
    pub fn lease(&self) -> Arc<BatchLease> {
        Arc::clone(&self.lease)
    }
}

impl Batch {
    pub fn arrays(&self) -> Vec<ArrayView> {
        self.arrays
            .iter()
            .map(|a| ArrayView {
                name: a.name.clone(),
                dtype: a.dtype,
                shape: a.shape.clone(),
                // In bounds: checked in accept_batch.
                ptr: unsafe {
                    self.lease
                        .map
                        .ptr
                        .add((self.lease.base + a.offset) as usize)
                },
                len: a.nbytes().unwrap() as usize,
                lease: Arc::clone(&self.lease),
            })
            .collect()
    }

    pub fn array(&self, name: &str) -> Option<ArrayView> {
        self.arrays().into_iter().find(|a| a.name == name)
    }
}
