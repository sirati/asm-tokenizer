//! Files under a loader's data roots, as the client names them.
//!
//! A client path is checked as text first (relative, no empty, `.` or `..`
//! components), then opened with openat2 from the root's directory fd with
//! RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS: the kernel refuses any step,
//! including through a symlink, that would leave the root. Either refusal is
//! hostile and ends the session; a path that simply does not exist is not.

use std::ffi::{CStr, CString};
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::Path;

use crate::bridge::proto::{DirEntry, EntryKind, MAX_ENTRIES, MAX_PATH};

#[derive(Debug)]
pub enum PathError {
    /// Traversal, an absolute path, a symlink out of the root: ends the session.
    Hostile(String),
    /// Does not exist, or is not what was asked for (file vs directory).
    NotFound(String),
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathError::Hostile(m) | PathError::NotFound(m) => f.write_str(m),
        }
    }
}

#[repr(C)]
struct OpenHow {
    flags: u64,
    mode: u64,
    resolve: u64,
}

const RESOLVE_NO_MAGICLINKS: u64 = 0x02;
const RESOLVE_BENEATH: u64 = 0x08;

/// Text checks: relative, components neither empty nor `.`/`..`, no NUL.
/// The empty path is the root itself.
pub fn check_relative(path: &str) -> Result<(), PathError> {
    let hostile = |why: &str| Err(PathError::Hostile(format!("path {path:?}: {why}")));
    if path.len() > MAX_PATH || path.contains('\0') {
        return hostile("malformed");
    }
    if path.is_empty() {
        return Ok(());
    }
    if path.starts_with('/') {
        return hostile("absolute");
    }
    for c in path.split('/') {
        if c.is_empty() || c == "." || c == ".." || c.len() > 255 {
            return hostile("empty, `.` or `..` component");
        }
    }
    Ok(())
}

/// One data root: a name and the directory, held open.
pub struct Root {
    pub name: String,
    dir: OwnedFd,
}

impl Root {
    pub fn open(name: &str, path: &Path) -> io::Result<Root> {
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(path.as_os_str().as_bytes())?;
        let fd = unsafe {
            libc::open(
                c.as_ptr(),
                libc::O_PATH | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Root {
            name: name.to_owned(),
            dir: unsafe { OwnedFd::from_raw_fd(fd) },
        })
    }

    fn openat2(&self, path: &str, flags: libc::c_int) -> Result<OwnedFd, PathError> {
        check_relative(path)?;
        openat2_beneath(&self.dir, path, flags).map_err(|e| {
            let what = format!("{}/{path}", self.name);
            match e.raw_os_error() {
                Some(libc::EXDEV) | Some(libc::ELOOP) => {
                    PathError::Hostile(format!("{what}: leaves the data root"))
                }
                Some(libc::ENOENT) | Some(libc::ENOTDIR) | Some(libc::EISDIR) => {
                    PathError::NotFound(format!("{what}: no such file"))
                }
                _ => PathError::NotFound(format!("{what}: {e}")),
            }
        })
    }

    /// A directory under the root, for a loader that walks a tree itself.
    pub fn open_dir(&self, path: &str) -> Result<DataDir, PathError> {
        let fd = self.openat2(path, libc::O_RDONLY | libc::O_DIRECTORY)?;
        Ok(DataDir { fd })
    }

    /// A regular file under the root, opened read-only.
    pub fn open_file(&self, path: &str) -> Result<File, PathError> {
        let fd = self.openat2(path, libc::O_RDONLY | libc::O_NONBLOCK)?;
        let file = File::from(fd);
        let md = file
            .metadata()
            .map_err(|e| PathError::NotFound(format!("{}/{path}: {e}", self.name)))?;
        if !md.is_file() {
            return Err(PathError::NotFound(format!(
                "{}/{path}: not a regular file",
                self.name
            )));
        }
        Ok(file)
    }

    /// One page of a directory: entries sorted by name from `offset`, and
    /// whether there are more. Names that are not UTF-8 are left out.
    pub fn list(&self, path: &str, offset: u32) -> Result<(Vec<DirEntry>, bool), PathError> {
        let fd = self.openat2(path, libc::O_RDONLY | libc::O_DIRECTORY)?;
        let raw = fd.as_raw_fd();
        let dir = unsafe { libc::fdopendir(raw) };
        if dir.is_null() {
            return Err(PathError::NotFound(format!(
                "{}/{path}: {}",
                self.name,
                io::Error::last_os_error()
            )));
        }
        // fdopendir owns the fd now; closedir closes it.
        std::mem::forget(fd);
        let mut names = Vec::new();
        loop {
            let ent = unsafe { libc::readdir64(dir) };
            if ent.is_null() {
                break;
            }
            let name = unsafe { CStr::from_ptr((*ent).d_name.as_ptr()) };
            let b = name.to_bytes();
            if b == b"." || b == b".." {
                continue;
            }
            if let Ok(s) = std::str::from_utf8(b) {
                names.push((s.to_owned(), name.to_owned()));
            }
        }
        names.sort();
        let start = (offset as usize).min(names.len());
        let end = (start + MAX_ENTRIES).min(names.len());
        let mut out = Vec::with_capacity(end - start);
        for (s, c) in &names[start..end] {
            let mut st: libc::stat64 = unsafe { std::mem::zeroed() };
            let r = unsafe { libc::fstatat64(raw, c.as_ptr(), &mut st, libc::AT_SYMLINK_NOFOLLOW) };
            let (kind, size) = if r != 0 {
                (EntryKind::Other, 0)
            } else {
                match st.st_mode & libc::S_IFMT {
                    libc::S_IFREG => (EntryKind::File, st.st_size as u64),
                    libc::S_IFDIR => (EntryKind::Dir, 0),
                    libc::S_IFLNK => (EntryKind::Symlink, 0),
                    _ => (EntryKind::Other, 0),
                }
            };
            out.push(DirEntry {
                name: s.clone(),
                kind,
                size,
            });
        }
        unsafe { libc::closedir(dir) };
        Ok((out, end < names.len()))
    }
}

/// openat2 of `path` (checked as text by the caller) beneath `dir`, with
/// RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS. The empty path is `dir` itself.
fn openat2_beneath(dir: &OwnedFd, path: &str, flags: libc::c_int) -> io::Result<OwnedFd> {
    let rel = if path.is_empty() { "." } else { path };
    let c = CString::new(rel).map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
    // openat2 refuses O_PATH with any flag but O_DIRECTORY, O_NOFOLLOW and
    // O_CLOEXEC.
    let extra = if flags & libc::O_PATH != 0 {
        libc::O_CLOEXEC
    } else {
        libc::O_CLOEXEC | libc::O_NOCTTY
    };
    let how = OpenHow {
        flags: (flags | extra) as u64,
        mode: 0,
        resolve: RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS,
    };
    let fd = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            dir.as_raw_fd(),
            c.as_ptr(),
            &how as *const OpenHow,
            std::mem::size_of::<OpenHow>(),
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd as libc::c_int) })
}

/// What a directory entry is, not following a symlink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataKind {
    File,
    Dir,
    Symlink,
    Other,
}

/// A directory of the host's data that a loader walks itself. Everything
/// under it is opened beneath it with openat2, as the client's paths are;
/// a name that is not a plain relative path, or a step that would leave the
/// directory, is an error.
pub struct DataDir {
    fd: OwnedFd,
}

fn data_path_ok(path: &str) -> io::Result<()> {
    check_relative(path).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{path:?} is not a plain relative path"),
        )
    })
}

impl DataDir {
    /// A directory on the host, outside a session (tests and tools).
    pub fn open_path(path: &Path) -> io::Result<DataDir> {
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(path.as_os_str().as_bytes())?;
        let fd = unsafe {
            libc::open(
                c.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(DataDir {
            fd: unsafe { OwnedFd::from_raw_fd(fd) },
        })
    }

    /// The directory itself, as a (read-only, unreadable) file.
    pub fn as_file(&self) -> io::Result<File> {
        Ok(File::from(self.fd.try_clone()?))
    }

    /// A regular file beneath this directory, read-only.
    pub fn open_file(&self, path: &str) -> io::Result<File> {
        data_path_ok(path)?;
        let file = File::from(openat2_beneath(
            &self.fd,
            path,
            libc::O_RDONLY | libc::O_NONBLOCK,
        )?);
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{path}: not a regular file"),
            ));
        }
        Ok(file)
    }

    /// A directory beneath this one.
    pub fn open_dir(&self, path: &str) -> io::Result<DataDir> {
        data_path_ok(path)?;
        Ok(DataDir {
            fd: openat2_beneath(&self.fd, path, libc::O_RDONLY | libc::O_DIRECTORY)?,
        })
    }

    /// Whether `path` exists beneath this directory, as anything.
    pub fn exists(&self, path: &str) -> bool {
        data_path_ok(path).is_ok() && openat2_beneath(&self.fd, path, libc::O_PATH).is_ok()
    }

    /// The entries in the order the kernel returns them (the order of
    /// Python's `os.scandir` and `Path.glob`), without `.` and `..`. Names
    /// that are not UTF-8 are left out.
    pub fn entries(&self) -> io::Result<Vec<(String, DataKind)>> {
        let fd = openat2_beneath(&self.fd, "", libc::O_RDONLY | libc::O_DIRECTORY)?;
        let raw = fd.as_raw_fd();
        let dir = unsafe { libc::fdopendir(raw) };
        if dir.is_null() {
            return Err(io::Error::last_os_error());
        }
        // fdopendir owns the fd now; closedir closes it.
        std::mem::forget(fd);
        let mut out = Vec::new();
        loop {
            let ent = unsafe { libc::readdir64(dir) };
            if ent.is_null() {
                break;
            }
            let name = unsafe { CStr::from_ptr((*ent).d_name.as_ptr()) };
            let b = name.to_bytes();
            if b == b"." || b == b".." {
                continue;
            }
            let Ok(s) = std::str::from_utf8(b) else {
                continue;
            };
            let kind = match unsafe { (*ent).d_type } {
                libc::DT_REG => DataKind::File,
                libc::DT_DIR => DataKind::Dir,
                libc::DT_LNK => DataKind::Symlink,
                libc::DT_UNKNOWN => {
                    let mut st: libc::stat64 = unsafe { std::mem::zeroed() };
                    let r = unsafe {
                        libc::fstatat64(raw, name.as_ptr(), &mut st, libc::AT_SYMLINK_NOFOLLOW)
                    };
                    match (r, st.st_mode & libc::S_IFMT) {
                        (0, libc::S_IFREG) => DataKind::File,
                        (0, libc::S_IFDIR) => DataKind::Dir,
                        (0, libc::S_IFLNK) => DataKind::Symlink,
                        _ => DataKind::Other,
                    }
                }
                _ => DataKind::Other,
            };
            out.push((s.to_owned(), kind));
        }
        unsafe { libc::closedir(dir) };
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_checks() {
        for ok in ["", "a", "a/b.bin", "a..b", "...x"] {
            assert!(check_relative(ok).is_ok(), "{ok}");
        }
        for bad in [
            "/etc/passwd",
            "..",
            "a/..",
            "../a",
            "a//b",
            "a/",
            "./a",
            "a/./b",
            "a\0b",
        ] {
            assert!(
                matches!(check_relative(bad), Err(PathError::Hostile(_))),
                "{bad}"
            );
        }
    }

    #[test]
    fn data_dir_stays_beneath() {
        use std::os::unix::fs::symlink;
        let base = std::env::temp_dir().join(format!("loader-bridge-datadir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("root");
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/a"), b"a").unwrap();
        std::fs::write(base.join("secret"), b"s").unwrap();
        symlink("../secret", root.join("out")).unwrap();
        symlink("/etc", root.join("abs")).unwrap();
        symlink("sub/a", root.join("in")).unwrap();

        let d = DataDir::open_path(&root).unwrap();
        let sub = d.open_dir("sub").unwrap();
        assert_eq!(std::io::read_to_string(sub.open_file("a").unwrap()).unwrap(), "a");
        assert!(d.open_file("in").is_ok());
        for bad in ["out", "abs/passwd", "../secret", "/etc/passwd", "sub/../out", "sub"] {
            assert!(d.open_file(bad).is_err(), "{bad}");
        }
        assert!(sub.open_file("../a").is_err());
        assert!(d.open_dir("abs").is_err());
        let mut names: Vec<_> = d.entries().unwrap();
        names.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            names,
            vec![
                ("abs".into(), DataKind::Symlink),
                ("in".into(), DataKind::Symlink),
                ("out".into(), DataKind::Symlink),
                ("sub".into(), DataKind::Dir),
            ]
        );
        assert!(d.exists("sub/a") && !d.exists("sub/b") && !d.exists("out"));
        std::fs::remove_dir_all(&base).unwrap();
    }
}
