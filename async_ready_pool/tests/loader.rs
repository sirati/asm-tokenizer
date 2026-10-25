//! The loader interface run in-process: parameters, inputs beneath the data
//! roots, layouts, items in order, requests and resuming.

use std::io::Read;

use _native::GetError;
use _native::loader::{
    ArraySpec, BatchMut, DType, InputFile, Item, Layout, Loader, LocalPool, ParamKind, ParamSpec,
    Params, RESUME_STATE, Source, Value, fill_detached, next_detached,
};

/// Items: `bytes` [take] u8, the first `take` bytes of the file (`take` a
/// request parameter, `n` by default), and `count` [1] u64, the item's
/// index. A directory input yields the names it holds. Its state is the
/// next index.
struct Head;

fn int(name: &str, min: i64, max: i64, default: Option<i64>) -> ParamSpec {
    ParamSpec {
        name: name.into(),
        kind: ParamKind::Int { min, max },
        default: default.map(Value::Int),
    }
}

impl Loader for Head {
    fn params(&self) -> Vec<ParamSpec> {
        vec![int("n", 1, 64, None), int("fail_at", -1, 1000, Some(-1))]
    }
    fn files(&self) -> (u16, u16) {
        (1, 1)
    }
    fn takes_dirs(&self) -> bool {
        true
    }
    fn open(
        &self,
        params: &Params,
        mut files: Vec<InputFile>,
        resume: Option<&[u8]>,
    ) -> Result<Box<dyn Source>, String> {
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
                (&input.file)
                    .read_to_end(&mut v)
                    .map_err(|e| e.to_string())?;
                v
            }
        };
        if data.len() < n {
            return Err(format!("{}: shorter than {n}", input.path));
        }
        data.truncate(n);
        let next = match resume {
            None => 0,
            Some(b) => u64::from_le_bytes(b.try_into().map_err(|_| "not a Head state")?),
        };
        Ok(Box::new(HeadSource {
            data,
            next,
            fail_at: params.int("fail_at"),
        }))
    }
}

struct HeadSource {
    data: Vec<u8>,
    next: u64,
    fail_at: i64,
}

struct HeadItem {
    bytes: Vec<u8>,
    count: u64,
}

impl Source for HeadSource {
    fn request_params(&self) -> Vec<ParamSpec> {
        let n = self.data.len() as i64;
        vec![int("take", 1, n, Some(n))]
    }
    fn max_layout(&self) -> Vec<ArraySpec> {
        vec![
            ArraySpec::new("bytes", DType::U8, &[self.data.len() as u64]),
            ArraySpec::new("count", DType::U64, &[1]),
        ]
    }
    fn next(&mut self, request: &Params) -> Result<Box<dyn Item>, String> {
        let i = self.next;
        self.next += 1;
        if i as i64 == self.fail_at {
            return Err(format!("item {i} fails"));
        }
        let take = request.int("take") as usize;
        Ok(Box::new(HeadItem {
            bytes: self.data[..take].to_vec(),
            count: i,
        }))
    }
    fn state(&self) -> Option<Vec<u8>> {
        Some(self.next.to_le_bytes().to_vec())
    }
}

impl Item for HeadItem {
    fn layout(&self) -> Vec<ArraySpec> {
        vec![
            ArraySpec::new("bytes", DType::U8, &[self.bytes.len() as u64]),
            ArraySpec::new("count", DType::U64, &[1]),
        ]
    }
    fn fill(self: Box<Self>, out: &mut BatchMut<'_>) -> Result<(), String> {
        // Later items finish first now and then: items still come in order.
        if self.count % 3 == 0 {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        out.bytes_mut(0).copy_from_slice(&self.bytes);
        out.array_mut::<u64>(1).unwrap()[0] = self.count;
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

fn count(item: &_native::loader::OwnedBatch) -> u64 {
    u64::from_le_bytes(item.array("count").unwrap().1.try_into().unwrap())
}

#[test]
fn a_planning_panic_unparks_later_tickets() {
    struct Panics;
    impl Source for Panics {
        fn max_layout(&self) -> Vec<ArraySpec> {
            vec![ArraySpec::new("value", DType::U8, &[1])]
        }
        fn next(&mut self, _: &Params) -> Result<Box<dyn Item>, String> {
            panic!("broken loader")
        }
    }
    let stream =
        std::sync::Arc::new(_native::loader::Stream::new(Box::new(Panics), false).unwrap());
    let later = stream.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || tx.send(later.next(1).is_err()).unwrap());
    assert!(stream.next(0).is_err());
    assert!(rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap());
    waiter.join().unwrap();
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
        vec![("n".into(), Value::Ints(vec![3]))],
    ] {
        assert!(Params::resolve(&specs, bad).is_err());
    }
    let ints = [ParamSpec {
        name: "xs".into(),
        kind: ParamKind::Ints {
            min: 0,
            max: 9,
            max_len: 3,
        },
        default: Some(Value::Ints(vec![])),
    }];
    assert_eq!(
        Params::resolve(&ints, vec![]).unwrap().ints("xs"),
        &[] as &[i64]
    );
    let p = Params::resolve(&ints, vec![("xs".into(), Value::Ints(vec![1, 9]))]).unwrap();
    assert_eq!(p.ints("xs"), [1, 9]);
    assert!(Params::resolve(&ints, vec![("xs".into(), Value::Ints(vec![1, 2, 3, 4]))]).is_err());
    assert!(Params::resolve(&ints, vec![("xs".into(), Value::Ints(vec![10]))]).is_err());
}

#[test]
fn local_pool_serves_items_in_order_and_failures() {
    let d = scratch("pool");
    let data = d.join("data");
    let mut pool = LocalPool::new(Head, &[("corpus".into(), data.as_path())]).unwrap();
    let a = pool
        .register(
            2,
            vec![("n".into(), Value::Int(5))],
            &[("corpus", "hello.txt")],
        )
        .unwrap();
    let b = pool
        .register(1, vec![("n".into(), Value::Int(9))], &[("corpus", "")])
        .unwrap();
    let c = pool
        .register(
            1,
            vec![
                ("n".into(), Value::Int(1)),
                ("fail_at".into(), Value::Int(0)),
            ],
            &[("corpus", "hello.txt")],
        )
        .unwrap();
    // Inputs stay beneath the root; a short input is the loader's refusal.
    for (root, path) in [
        ("corpus", "escape"),
        ("corpus", "../secret"),
        ("other", "x"),
    ] {
        assert!(
            pool.register(1, vec![("n".into(), Value::Int(1))], &[(root, path)])
                .is_err()
        );
    }
    assert!(
        pool.register(
            1,
            vec![("n".into(), Value::Int(64))],
            &[("corpus", "hello.txt")]
        )
        .is_err()
    );
    // A resume state the loader does not know is its refusal.
    assert!(
        pool.register_with(
            1,
            vec![("n".into(), Value::Int(1))],
            &[("corpus", "hello.txt")],
            Some(b"x"),
            false
        )
        .is_err()
    );
    let (entries, more) = pool.list("corpus", "", 0).unwrap();
    assert!(!more);
    let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["escape", "hello.txt", "sub"]);
    assert_eq!(
        pool.request_params(a).unwrap()[0].kind,
        ParamKind::Int { min: 1, max: 5 }
    );

    pool.start(3).unwrap();
    for i in 0..30 {
        let item = pool.get(a).unwrap();
        assert_eq!(item.array("bytes").unwrap().1, b"hello");
        assert_eq!(count(&item), i);
        assert_eq!(item.array(RESUME_STATE).unwrap().1, (i + 1).to_le_bytes());
    }
    assert!(
        pool.request(a, vec![]).is_err(),
        "a registration kept ready takes no requests"
    );
    assert_eq!(pool.get(b).unwrap().array("bytes").unwrap().1, b"escape,he");
    assert!(matches!(pool.get(c), Err(GetError::Failed(m)) if m.contains("item 0 fails")));
    pool.close();
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn requests_shape_items_and_resume_goes_on() {
    let d = scratch("requests");
    let data = d.join("data");
    let mut pool = LocalPool::new(Head, &[("corpus".into(), data.as_path())]).unwrap();
    let files = [("corpus", "hello.txt")];
    let n = || vec![("n".into(), Value::Int(12))];
    let r = pool.register_with(2, n(), &files, None, true).unwrap();
    // Resumed after item 6 of an earlier registration.
    let s = pool
        .register_with(1, n(), &files, Some(&7u64.to_le_bytes()), true)
        .unwrap();
    pool.start(4).unwrap();
    let takes = [3i64, 12, 1, 7, 5, 2, 9, 4];
    for &t in &takes {
        pool.request(r, vec![("take".into(), Value::Int(t))])
            .unwrap();
    }
    assert!(
        pool.request(r, vec![("take".into(), Value::Int(13))])
            .is_err()
    );
    assert!(
        pool.request(r, vec![("other".into(), Value::Int(1))])
            .is_err()
    );
    let mut state = Vec::new();
    for (i, &t) in takes.iter().enumerate() {
        let item = pool.get(r).unwrap();
        assert_eq!(
            item.array("bytes").unwrap().1,
            &b"hello, world"[..t as usize]
        );
        assert_eq!(item.array("bytes").unwrap().0.shape, [t as u64]);
        assert_eq!(count(&item), i as u64);
        state = item.array(RESUME_STATE).unwrap().1.to_vec();
    }
    assert_eq!(state, 8u64.to_le_bytes());
    pool.request(s, vec![]).unwrap();
    assert_eq!(count(&pool.get(s).unwrap()), 7);
    pool.close();
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn detached_fill_and_layout_checks() {
    let mut src = HeadSource {
        data: b"abc".to_vec(),
        next: 7,
        fail_at: -1,
    };
    let item = fill_detached(&mut src).unwrap();
    assert_eq!(item.array("bytes").unwrap().1, b"abc");
    assert_eq!(item.arrays[1].offset, 64);
    assert_eq!(count(&item), 7);
    let item = next_detached(&mut src, vec![("take".into(), Value::Int(2))]).unwrap();
    assert_eq!(
        (item.array("bytes").unwrap().1, count(&item)),
        (&b"ab"[..], 8)
    );
    assert!(Layout::of(&[]).is_err());
    assert!(Layout::of(&[ArraySpec::new("../x", DType::U8, &[1])]).is_err());
    assert!(Layout::of(&[ArraySpec::new("x", DType::U64, &[u64::MAX, 2])]).is_err());
    // An item's memory must hold its arrays, aligned.
    let layout = Layout::of(&src.max_layout()).unwrap();
    let mut short = vec![0u8; 8];
    assert!(BatchMut::new(&mut short, &layout.arrays).is_none());
}
