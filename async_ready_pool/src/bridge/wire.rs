//! Packets over a SOCK_SEQPACKET unix socket, with SCM_RIGHTS fds.

use std::io;
use std::mem;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;

use crate::bridge::proto::MAX_MESSAGE;

/// The most fds either side ever puts on one packet. More is a violation.
pub const MAX_FDS: usize = 1;
/// Room for more than MAX_FDS, so that a peer sending more is seen (and its
/// fds closed) rather than silently cut off.
const CMSG_FDS: usize = 8;

pub struct Packet {
    pub data: Vec<u8>,
    /// Every fd that arrived; closed on drop unless taken.
    pub fds: Vec<OwnedFd>,
    /// The packet was longer than MAX_MESSAGE.
    pub truncated: bool,
    /// Control data was cut off (more fds than fit, or other ancillary data).
    pub ctrunc: bool,
    /// Ancillary data other than SCM_RIGHTS was present.
    pub foreign_cmsg: bool,
}

impl Packet {
    /// A packet the host accepts from a client: within size, with no fds or
    /// other ancillary data.
    pub fn is_clean(&self) -> bool {
        !self.truncated && !self.ctrunc && !self.foreign_cmsg && self.fds.is_empty()
    }
}

fn cmsg_space(fds: usize) -> usize {
    unsafe { libc::CMSG_SPACE((fds * mem::size_of::<RawFd>()) as u32) as usize }
}

/// Sends one packet with `fds` attached. MSG_NOSIGNAL: a closed peer is an
/// error, not SIGPIPE.
pub fn send(sock: BorrowedFd<'_>, data: &[u8], fds: &[BorrowedFd<'_>]) -> io::Result<()> {
    assert!(fds.len() <= CMSG_FDS);
    let mut iov = libc::iovec {
        iov_base: data.as_ptr() as *mut _,
        iov_len: data.len(),
    };
    let mut msg: libc::msghdr = unsafe { mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    // u64 for alignment.
    let mut cbuf = [0u64; 16];
    if !fds.is_empty() {
        let space = cmsg_space(fds.len());
        assert!(space <= mem::size_of_val(&cbuf));
        msg.msg_control = cbuf.as_mut_ptr() as *mut _;
        msg.msg_controllen = space as _;
        unsafe {
            let c = libc::CMSG_FIRSTHDR(&msg);
            (*c).cmsg_level = libc::SOL_SOCKET;
            (*c).cmsg_type = libc::SCM_RIGHTS;
            (*c).cmsg_len = libc::CMSG_LEN((fds.len() * mem::size_of::<RawFd>()) as u32) as _;
            let p = libc::CMSG_DATA(c) as *mut RawFd;
            for (i, fd) in fds.iter().enumerate() {
                ptr::write_unaligned(p.add(i), fd.as_raw_fd());
            }
        }
    }
    loop {
        let n = unsafe { libc::sendmsg(sock.as_raw_fd(), &msg, libc::MSG_NOSIGNAL) };
        if n >= 0 {
            if n as usize != data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "short seqpacket send",
                ));
            }
            return Ok(());
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

/// Receives one packet. Ok(None) is end of stream.
pub fn recv(sock: BorrowedFd<'_>) -> io::Result<Option<Packet>> {
    // One byte more than the largest message, so an oversized one shows.
    let mut data = vec![0u8; MAX_MESSAGE + 1];
    let mut iov = libc::iovec {
        iov_base: data.as_mut_ptr() as *mut _,
        iov_len: data.len(),
    };
    let mut cbuf = [0u64; 64];
    let space = cmsg_space(CMSG_FDS);
    assert!(space <= mem::size_of_val(&cbuf));
    let mut msg: libc::msghdr = unsafe { mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cbuf.as_mut_ptr() as *mut _;
    msg.msg_controllen = space as _;
    let n = loop {
        let n = unsafe { libc::recvmsg(sock.as_raw_fd(), &mut msg, libc::MSG_CMSG_CLOEXEC) };
        if n >= 0 {
            break n as usize;
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    };
    let mut fds = Vec::new();
    let mut foreign_cmsg = false;
    unsafe {
        let mut c = libc::CMSG_FIRSTHDR(&msg);
        while !c.is_null() {
            if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                let payload = (*c).cmsg_len as usize - libc::CMSG_LEN(0) as usize;
                let p = libc::CMSG_DATA(c) as *const RawFd;
                for i in 0..payload / mem::size_of::<RawFd>() {
                    fds.push(OwnedFd::from_raw_fd(ptr::read_unaligned(p.add(i))));
                }
            } else {
                foreign_cmsg = true;
            }
            c = libc::CMSG_NXTHDR(&msg, c);
        }
    }
    let truncated = msg.msg_flags & libc::MSG_TRUNC != 0 || n > MAX_MESSAGE;
    let ctrunc = msg.msg_flags & libc::MSG_CTRUNC != 0;
    if n == 0 && !truncated && fds.is_empty() {
        // A zero-length packet is never a message; on SEQPACKET recvmsg also
        // returns 0 at end of stream. Both end the conversation.
        return Ok(None);
    }
    data.truncate(n.min(MAX_MESSAGE));
    Ok(Some(Packet {
        data,
        fds,
        truncated,
        ctrunc,
        foreign_cmsg,
    }))
}

/// A connected SOCK_SEQPACKET pair (tests, and nothing else).
pub fn pair() -> io::Result<(OwnedFd, OwnedFd)> {
    let mut fds = [0; 2];
    let r = unsafe {
        libc::socketpair(
            libc::AF_UNIX,
            libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC,
            0,
            fds.as_mut_ptr(),
        )
    };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    unsafe { Ok((OwnedFd::from_raw_fd(fds[0]), OwnedFd::from_raw_fd(fds[1]))) }
}

/// Connects to a SOCK_SEQPACKET listener at `path`.
pub fn connect(path: &std::path::Path) -> io::Result<OwnedFd> {
    use std::os::unix::ffi::OsStrExt;
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let (addr, len) = sockaddr(path.as_os_str().as_bytes())?;
    let r = unsafe {
        libc::connect(
            fd.as_raw_fd(),
            &addr as *const _ as *const libc::sockaddr,
            len,
        )
    };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

/// Binds and listens on a SOCK_SEQPACKET socket at `path`.
pub fn listen(path: &std::path::Path) -> io::Result<OwnedFd> {
    use std::os::unix::ffi::OsStrExt;
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let (addr, len) = sockaddr(path.as_os_str().as_bytes())?;
    if unsafe {
        libc::bind(
            fd.as_raw_fd(),
            &addr as *const _ as *const libc::sockaddr,
            len,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::listen(fd.as_raw_fd(), 64) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(fd)
}

fn sockaddr(path: &[u8]) -> io::Result<(libc::sockaddr_un, libc::socklen_t)> {
    let mut addr: libc::sockaddr_un = unsafe { mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as _;
    if path.len() >= addr.sun_path.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "socket path too long",
        ));
    }
    for (d, s) in addr.sun_path.iter_mut().zip(path) {
        *d = *s as libc::c_char;
    }
    let len = mem::size_of::<libc::sa_family_t>() + path.len() + 1;
    Ok((addr, len as libc::socklen_t))
}

/// Sets SO_RCVTIMEO.
pub fn set_recv_timeout(sock: BorrowedFd<'_>, ms: u64) -> io::Result<()> {
    let tv = libc::timeval {
        tv_sec: (ms / 1000) as _,
        tv_usec: ((ms % 1000) * 1000) as _,
    };
    let r = unsafe {
        libc::setsockopt(
            sock.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_RCVTIMEO,
            &tv as *const _ as *const _,
            mem::size_of_val(&tv) as _,
        )
    };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Whether `fd` is a SOCK_SEQPACKET socket.
pub fn is_seqpacket(fd: BorrowedFd<'_>) -> bool {
    let mut ty: libc::c_int = 0;
    let mut len = mem::size_of::<libc::c_int>() as libc::socklen_t;
    let r = unsafe {
        libc::getsockopt(
            fd.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_TYPE,
            &mut ty as *mut _ as *mut _,
            &mut len,
        )
    };
    r == 0 && ty == libc::SOCK_SEQPACKET
}
