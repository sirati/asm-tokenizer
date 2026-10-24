//! The loader interface run in-process: parameters, inputs beneath the data
//! roots, layouts, and items as the pool hands them out.

use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};

use _native::loader::{
    ArraySpec, BatchMut, DType, InputFile, Layout, LocalPool, Loader, ParamKind, ParamSpec,
    Params, Source, Value, fill_detached,
};
use _native::GetError;

/// Items: `bytes` [n] u8, the first n bytes of the file, and `count` [1]
/// u64, the item's index. A directory input yields the names it holds.
struct Head;

impl Loader for Head {
    fn params(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec {
                name: "n".into(),
                kind: ParamKind::Int { min: 1, max: 64 },
                default: None,
            },
            ParamSpec {
                name: "fail_at".into(),
                kind: ParamKind::Int { min: -1, max: 1000 },
                default: Some(Value::Int(-1)),
            },
        ]
    }
    fn files(&self) -> (u16, u16) {
        (1, 1)
    }
    fn takes_dirs(&self) -> bool {
        true
    }
    fn open(&self, params: &Params, mut files: Vec<InputFile>) -> Result<Box<dyn Source>, String> {
        let input = files.pop().unwrap();
        let n = params.int("n") as usize;
        let mut data = match &input.dir {
            Some(d) => {
                let mut names: Vec<_> = d
                    .entries()
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .map(|(n, _)| n)
                    .collect();
                names.sort();
                names.join(",").into_bytes()
            }
            None => {
                let mut v = Vec::new();
                (&input.file).read_to_end(&mut v).map_err(|e| e.to_string())?;
                v
            }
        };
        if data.len() < n {
            return Err(format!("{}: shorter than {n}", input.path));
        }
        data.truncate(n);
        Ok(Box::new(HeadSource {
            data,
            next: AtomicU64::new(0),
            fail_at: params.int("fail_at"),
        }))
    }
}

struct HeadSource {
    data: Vec<u8>,
    next: AtomicU64,
    fail_at: i64,
}

impl Source for HeadSource {
    fn layout(&self) -> Vec<ArraySpec> {
        vec![
            ArraySpec::new("bytes", DType::U8, &[self.data.len() as u64]),
            ArraySpec::new("count", DType::U64, &[1]),
        ]
    }
    fn fill(&self, out: &mut BatchMut<'_>) -> Result<(), String> {
        let i = self.next.fetch_add(1, Ordering::SeqCst);
        if i as i64 == self.fail_at {
            return Err(format!("item {i} fails"));
        }
        out.bytes_mut(0).copy_from_slice(&self.data);
        out.array_mut::<u64>(1).unwrap()[0] = i;
        Ok(())
    }
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("arp-loader-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("data/sub")).unwrap();
    std::fs::write(d.join("data/hello.txt"), b"hello, world").unwrap();
    std::fs::write(d.join("secret"), b"outside").unwrap();
    std::os::unix::fs::symlink("../secret", d.join("data/escape")).unwrap();
    d
}

#[test]
fn params_resolve_like_the_bridge() {
    let specs = Head.params();
    let p = Params::resolve(&specs, vec![("n".into(), Value::Int(3))]).unwrap();
    assert_eq!((p.int("n"), p.int("fail_at")), (3, -1));
    for bad in [
        vec![],
        vec![("n".into(), Value::Int(0))],
        vec![("n".into(), Value::Bool(true))],
        vec![("m".into(), Value::Int(3))],
        vec![("n".into(), Value::Int(3)), ("n".into(), Value::Int(4))],
    ] {
        assert!(Params::resolve(&specs, bad).is_err());
    }
}

#[test]
fn local_pool_serves_items_and_failures() {
    let d = scratch("pool");
    let data = d.join("data");
    let mut pool = LocalPool::new(Head, &[("corpus".into(), data.as_path())]).unwrap();
    let a = pool
        .register(2, vec![("n".into(), Value::Int(5))], &[("corpus", "hello.txt")])
        .unwrap();
    let b = pool
        .register(1, vec![("n".into(), Value::Int(9))], &[("corpus", "")])
        .unwrap();
    let c = pool
        .register(
            1,
            vec![("n".into(), Value::Int(1)), ("fail_at".into(), Value::Int(0))],
            &[("corpus", "hello.txt")],
        )
        .unwrap();
    // Inputs stay beneath the root; a short input is the loader's refusal.
    for (root, path) in [("corpus", "escape"), ("corpus", "../secret"), ("other", "x")] {
        assert!(pool.register(1, vec![("n".into(), Value::Int(1))], &[(root, path)]).is_err());
    }
    assert!(pool.register(1, vec![("n".into(), Value::Int(64))], &[("corpus", "hello.txt")]).is_err());
    let (entries, more) = pool.list("corpus", "", 0).unwrap();
    assert!(!more);
    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["escape", "hello.txt", "sub"]);

    pool.start(2).unwrap();
    let mut seen = Vec::new();
    for _ in 0..6 {
        let item = pool.get(a).unwrap();
        assert_eq!(item.array("bytes").unwrap().1, b"hello");
        seen.push(u64::from_le_bytes(item.array("count").unwrap().1.try_into().unwrap()));
    }
    seen.sort();
    assert_eq!(seen, [0, 1, 2, 3, 4, 5]);
    assert_eq!(pool.get(b).unwrap().array("bytes").unwrap().1, b"escape,he");
    assert!(matches!(pool.get(c), Err(GetError::Failed(m)) if m.contains("item 0 fails")));
    pool.close();
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn detached_fill_and_layout_checks() {
    let src = HeadSource {
        data: b"abc".to_vec(),
        next: AtomicU64::new(7),
        fail_at: -1,
    };
    let item = fill_detached(&src).unwrap();
    assert_eq!(item.array("bytes").unwrap().1, b"abc");
    assert_eq!(item.arrays[1].offset, 64);
    assert!(Layout::of(&[]).is_err());
    assert!(Layout::of(&[ArraySpec::new("../x", DType::U8, &[1])]).is_err());
    assert!(Layout::of(&[ArraySpec::new("x", DType::U64, &[u64::MAX, 2])]).is_err());
    // An item's memory must hold its arrays, aligned.
    let layout = Layout::of(&src.layout()).unwrap();
    let mut short = vec![0u8; 8];
    assert!(BatchMut::new(&mut short, &layout.arrays).is_none());
}
