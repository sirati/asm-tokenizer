//! The loader bridge: a [`Loader`](crate::loader::Loader) running in another
//! process (on a container's host, or a local child) serves its batches over
//! a SOCK_SEQPACKET connection as read-only, sealed shared memory.
//!
//! - [`proto`]: the versioned, bounds-checked messages.
//! - [`wire`]: one message per packet, fds by SCM_RIGHTS.
//! - [`client`]: the consuming side; it checks the server's messages too.
//!
//! The serving side (the broker, the session adapter, the segment allocator)
//! is the `loader-bridge` crate, which links this one.

pub mod proto;

#[cfg(target_os = "linux")]
pub mod client;
#[cfg(target_os = "linux")]
pub mod wire;

/// The seals every segment carries before its fd is sent: no shrinking,
/// growing, or writing through any new mapping or write(), and no more seals.
#[cfg(target_os = "linux")]
pub const SEALS: libc::c_int =
    libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_FUTURE_WRITE | libc::F_SEAL_SEAL;

/// The argument a loader binary is started with to serve one session.
pub const SESSION_FLAG: &str = "--loader-bridge-session";
/// The connection's fd in a session.
pub const CONN_FD: i32 = 3;

/// A session's configuration, as its loader binary's argv:
/// `--loader-bridge-session <request> <loader> <max_shared_bytes> <max_workers> [<root>=<path>]...`
/// `request` is the request ID the Welcome answers (the Hello's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionConfig {
    pub request: u64,
    pub loader: String,
    pub max_shared_bytes: u64,
    pub max_workers: u32,
    pub roots: Vec<(String, std::path::PathBuf)>,
}

impl SessionConfig {
    pub fn from_args(args: &[String]) -> Option<SessionConfig> {
        let [flag, request, loader, max_shared, max_workers, roots @ ..] = args else {
            return None;
        };
        if flag != SESSION_FLAG || !proto::valid_name(loader) {
            return None;
        }
        let roots = roots
            .iter()
            .map(|r| {
                let (n, p) = r.split_once('=')?;
                proto::valid_name(n).then(|| (n.to_owned(), std::path::PathBuf::from(p)))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(SessionConfig {
            request: request.parse().ok()?,
            loader: loader.clone(),
            max_shared_bytes: max_shared.parse().ok()?,
            max_workers: max_workers.parse().ok().filter(|w| *w > 0)?,
            roots,
        })
    }

    pub fn to_args(&self) -> Vec<String> {
        let mut v = vec![
            SESSION_FLAG.to_owned(),
            self.request.to_string(),
            self.loader.clone(),
            self.max_shared_bytes.to_string(),
            self.max_workers.to_string(),
        ];
        v.extend(
            self.roots
                .iter()
                .map(|(n, p)| format!("{n}={}", p.display())),
        );
        v
    }
}
