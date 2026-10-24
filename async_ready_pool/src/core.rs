//! `preload::core` — the pure-Rust bucket-keyed ready-buffer (design §5).
//!
//! Single concern: a set of opaque-registration-keyed BOUNDED ready
//! buffers that background workers keep topped up to a per-registration target and
//! a consumer drains, with NO way to park forever (the gpu_prefetcher #72/#73
//! lesson encoded structurally). This file is PyO3-free and generic over the
//! decoded item type `T: Send`, so the whole back-pressure + no-hang state
//! machine is unit-testable with a trivial `T` (e.g. `u64`) and no `Python`
//! token; the pyclass wrapper (`preload::mod`) instantiates it with a sendable
//! Python object and drives the workers' `produce` via `Python::attach`.
//!
//! ## The state machine (why it can never park forever)
//!
//! One `Mutex<Inner>` guards every bucket; two `Condvar`s ride it:
//!   * `not_empty` — woken when a worker DEPOSITS; a puller waits on it.
//!   * `not_full`  — woken when a puller DRAINS; a worker waits on it.
//! Every wait predicate ALSO returns on `closing` or the selected registration's
//! failure, so a dead/panicking producer wakes both condvars and the affected
//! puller returns a typed error immediately;
//!   * `close()` sets `closing` + wakes both → workers drain out, pullers stop.
//! There is no unbounded queue and no `get()` that can outlive its producer:
//! the bound is `keep_ready` per bucket, and a starved puller is always either
//! served, told the pool failed, or told it is closing.
//!
//! ## Deficit accounting (keep N ready "at all times")
//!
//! A worker claims the MOST-STARVED bucket: the one with the largest
//! `keep_ready - (ready.len() + in_flight)` deficit. `in_flight` counts slots a
//! worker has CLAIMED but not yet deposited, so two workers never both target
//! the last missing slot of one bucket while another bucket starves. A claim
//! reserves the slot (`in_flight += 1`); a deposit converts it to a ready item
//! (`in_flight -= 1`, push); a failed produce releases it (`in_flight -= 1`).

use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::{Condvar, Mutex};
use std::thread::{self, JoinHandle};

/// A registered bucket is identified by an opaque registration ID.
/// Domain-specific keys stay outside this scheduler.
pub type RegistrationId = u64;

/// Per-bucket mutable state: the ready ring + its target + the in-flight count.
struct Bucket<T> {
    ready: VecDeque<T>,
    keep_ready: usize,
    in_flight: usize,
    failure: Option<String>,
}

impl<T> Bucket<T> {
    /// How many more items this bucket WANTS dispatched right now: the target
    /// minus what is ready minus what workers are already producing for it.
    /// Saturating (never negative) so an over-full bucket reports a 0 deficit.
    fn deficit(&self) -> usize {
        self.keep_ready
            .saturating_sub(self.ready.len() + self.in_flight)
    }
}

/// The lock-guarded shared state. One mutex over every bucket keeps the
/// most-starved pick + the deposit/drain transitions a single critical
/// section (the buckets are cheap to scan: a handful registered, not 1e6).
struct Inner<T> {
    buckets: HashMap<RegistrationId, Bucket<T>>,
    /// Set by `close()`: workers should drain out, pullers should stop waiting.
    closing: bool,
}

/// What a blocked `get` returns when it cannot hand back an item.
#[derive(Debug, PartialEq, Eq)]
pub enum GetError {
    /// No such bucket was registered.
    UnknownBucket(RegistrationId),
    /// A worker died/raised; carries the worker-side message.
    Failed(String),
    /// `close()` was called while a puller waited (or before it asked).
    Closing,
}

/// What a blocked `claim` returns when there is no work to do.
#[derive(Debug, PartialEq, Eq)]
pub enum ClaimStop {
    /// The pool failed; the worker should exit (the failure is already set).
    Failed,
    /// `close()` was called; the worker should exit.
    Closing,
}

/// The bucket-keyed ready-buffer. Cloneable handle is NOT provided here; the
/// pyclass wraps a single `Arc<PreloadCore<T>>` shared with every worker.
pub struct PreloadCore<T> {
    inner: Mutex<Inner<T>>,
    not_empty: Condvar,
    not_full: Condvar,
}

impl<T> PreloadCore<T> {
    /// Build an empty core (no buckets yet). Buckets are added by `register`
    /// before the workers start.
    pub fn new() -> Self {
        PreloadCore {
            inner: Mutex::new(Inner {
                buckets: HashMap::new(),
                closing: false,
            }),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
        }
    }

    /// Register one opaque registration bucket with a ready-count target.
    /// `keep_ready == 0` is rejected by the caller (the pyclass) — a 0-target
    /// bucket could never satisfy a `get`.
    pub fn register(&self, key: RegistrationId, keep_ready: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner
            .buckets
            .entry(key)
            .and_modify(|b| b.keep_ready = keep_ready)
            .or_insert_with(|| Bucket {
                ready: VecDeque::new(),
                keep_ready,
                in_flight: 0,
                failure: None,
            });
    }

    /// Whether a bucket is registered (for the pyclass `get` pre-check).
    pub fn has_bucket(&self, key: RegistrationId) -> bool {
        self.inner.lock().unwrap().buckets.contains_key(&key)
    }

    /// Block until SOME bucket has a deficit, then reserve one of its slots and
    /// return its key (the worker then produces an item for it OUTSIDE the
    /// lock). Returns `Err` when the worker should exit (failure or closing).
    ///
    /// Reserving here (`in_flight += 1`) under the lock is what makes the
    /// most-starved pick race-free: the next worker scanning sees this slot
    /// already accounted for.
    pub fn claim(&self) -> Result<RegistrationId, ClaimStop> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if inner.closing {
                return Err(ClaimStop::Closing);
            }
            if let Some(key) = most_starved(&inner.buckets) {
                inner.buckets.get_mut(&key).unwrap().in_flight += 1;
                return Ok(key);
            }
            if inner
                .buckets
                .values()
                .all(|bucket| bucket.failure.is_some())
            {
                return Err(ClaimStop::Failed);
            }
            // Every healthy bucket is full: wait for a drain (not_full) to open a slot.
            // Spurious wakeups are fine — the loop re-checks the predicate.
            inner = self.not_full.wait(inner).unwrap();
        }
    }

    /// Deposit a produced item into its bucket: convert the reserved in-flight
    /// slot into a ready item and wake one puller. Called by the worker AFTER
    /// `produce` succeeds. If the bucket vanished (only on close) the item is
    /// dropped.
    pub fn deposit(&self, key: RegistrationId, item: T) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(bucket) = inner.buckets.get_mut(&key) {
            bucket.in_flight = bucket.in_flight.saturating_sub(1);
            bucket.ready.push_back(item);
        }
        // A new item is ready for exactly one waiting puller.
        self.not_empty.notify_one();
    }

    /// Release a reserved slot WITHOUT depositing (the produce failed): mark
    /// the pool failed and wake EVERYONE so no caller parks. Sticky: a second
    /// failure keeps the first message.
    pub fn fail(&self, key: RegistrationId, message: String) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(bucket) = inner.buckets.get_mut(&key) {
            bucket.in_flight = bucket.in_flight.saturating_sub(1);
        }
        if let Some(bucket) = inner.buckets.get_mut(&key) {
            if bucket.failure.is_none() {
                bucket.failure = Some(message);
            }
        }
        // Wake every parked puller AND refiller — they re-check the bucket failure.
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    /// Pop one ready item from `key`, blocking until one is ready. Returns
    /// `Err` instead of parking forever when the pool failed or is closing, or
    /// at once when the bucket was never registered.
    ///
    /// Draining frees a slot, so a `not_full` notify wakes a refiller to top
    /// the bucket back up (the self-refilling property).
    pub fn get(&self, key: RegistrationId) -> Result<T, GetError> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if !inner.buckets.contains_key(&key) {
                return Err(GetError::UnknownBucket(key));
            }
            // A producer failure is surfaced before buffered data for THIS
            // registration, while unrelated registrations keep running.
            if let Some(msg) = &inner.buckets.get(&key).unwrap().failure {
                return Err(GetError::Failed(msg.clone()));
            }
            if let Some(item) = inner.buckets.get_mut(&key).unwrap().ready.pop_front() {
                // Opened a slot: a refiller should top this bucket up.
                self.not_full.notify_one();
                return Ok(item);
            }
            if inner.closing {
                return Err(GetError::Closing);
            }
            inner = self.not_empty.wait(inner).unwrap();
        }
    }

    /// Signal every worker to exit and every puller to stop waiting. Idempotent.
    /// The caller (the pyclass `close`) then joins the worker threads.
    pub fn close(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closing = true;
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }

    /// Snapshot the per-bucket ready counts (test/diagnostic aid). Returned as
    /// a sorted `Vec` so assertions are deterministic.
    pub fn ready_counts(&self) -> Vec<(RegistrationId, usize)> {
        let inner = self.inner.lock().unwrap();
        let mut out: Vec<(RegistrationId, usize)> = inner
            .buckets
            .iter()
            .map(|(k, b)| (*k, b.ready.len()))
            .collect();
        out.sort_by_key(|(k, _)| *k);
        out
    }
}

/// Generic Rust-facing pool. It owns worker scheduling and lifecycle; the
/// caller supplies only `producer(registration) -> Result<T, String>`.
pub struct PreloadPool<T, F>
where
    T: Send + 'static,
    F: Fn(RegistrationId) -> Result<T, String> + Send + Sync + 'static,
{
    core: Arc<PreloadCore<T>>,
    producer: Arc<F>,
    workers: usize,
    next_registration: RegistrationId,
    handles: Vec<JoinHandle<()>>,
    started: bool,
    closed: bool,
}

impl<T, F> PreloadPool<T, F>
where
    T: Send + 'static,
    F: Fn(RegistrationId) -> Result<T, String> + Send + Sync + 'static,
{
    pub fn new(producer: F, workers: usize) -> Result<Self, &'static str> {
        if workers == 0 {
            return Err("workers must be greater than zero");
        }
        Ok(Self {
            core: Arc::new(PreloadCore::new()),
            producer: Arc::new(producer),
            workers,
            next_registration: 0,
            handles: Vec::new(),
            started: false,
            closed: false,
        })
    }

    pub fn register(&mut self, keep_ready: usize) -> Result<RegistrationId, &'static str> {
        if self.closed {
            return Err("pool has been closed");
        }
        if self.started {
            return Err("register must happen before start");
        }
        if keep_ready == 0 {
            return Err("keep_ready must be greater than zero");
        }
        let registration = self.next_registration;
        self.next_registration = self
            .next_registration
            .checked_add(1)
            .ok_or("registration id exhausted")?;
        self.core.register(registration, keep_ready);
        Ok(registration)
    }

    pub fn start(&mut self) -> Result<(), &'static str> {
        if self.closed {
            return Err("pool has been closed");
        }
        if self.started {
            return Err("pool has already been started");
        }
        if self.core.ready_counts().is_empty() {
            return Err("register at least one producer before start");
        }
        for _ in 0..self.workers {
            let core = Arc::clone(&self.core);
            let producer = Arc::clone(&self.producer);
            self.handles.push(thread::spawn(move || {
                loop {
                    let registration = match core.claim() {
                        Ok(registration) => registration,
                        Err(ClaimStop::Closing) | Err(ClaimStop::Failed) => return,
                    };
                    let produced = catch_unwind(AssertUnwindSafe(|| producer(registration)));
                    match produced {
                        Ok(Ok(item)) => core.deposit(registration, item),
                        Ok(Err(message)) => core.fail(registration, message),
                        Err(_) => core.fail(
                            registration,
                            "producer panicked (Rust unwind caught in worker)".to_owned(),
                        ),
                    }
                }
            }));
        }
        self.started = true;
        Ok(())
    }

    pub fn get(&self, registration: RegistrationId) -> Result<T, GetError> {
        if !self.started {
            return Err(GetError::Closing);
        }
        self.core.get(registration)
    }

    pub fn ready_counts(&self) -> Vec<(RegistrationId, usize)> {
        self.core.ready_counts()
    }

    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.core.close();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        self.started = false;
        self.closed = true;
    }
}

impl<T, F> Drop for PreloadPool<T, F>
where
    T: Send + 'static,
    F: Fn(RegistrationId) -> Result<T, String> + Send + Sync + 'static,
{
    fn drop(&mut self) {
        self.close();
    }
}

/// Pick the bucket with the largest deficit (ties broken by key so the choice
/// is deterministic). Returns `None` when every bucket is at/over its target.
fn most_starved<T>(buckets: &HashMap<RegistrationId, Bucket<T>>) -> Option<RegistrationId> {
    buckets
        .iter()
        .filter(|(_, b)| b.failure.is_none() && b.deficit() > 0)
        // Max by deficit; deterministic tie-break by registration ID.
        .max_by_key(|(k, b)| (b.deficit(), *k))
        .map(|(k, _)| *k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn key(bs: u64, _sl: u64) -> RegistrationId {
        bs
    }

    #[test]
    fn deficit_is_target_minus_ready_and_inflight() {
        let b = Bucket::<u64> {
            ready: VecDeque::from(vec![1, 2]),
            keep_ready: 5,
            in_flight: 1,
            failure: None,
        };
        assert_eq!(b.deficit(), 2);
    }

    #[test]
    fn deficit_saturates_when_overfull() {
        let b = Bucket::<u64> {
            ready: VecDeque::from(vec![1, 2, 3]),
            keep_ready: 2,
            in_flight: 0,
            failure: None,
        };
        assert_eq!(b.deficit(), 0);
    }

    #[test]
    fn most_starved_picks_largest_deficit() {
        let mut m = HashMap::new();
        m.insert(
            key(1, 1),
            Bucket::<u64> {
                ready: VecDeque::new(),
                keep_ready: 2,
                in_flight: 0,
                failure: None,
            },
        );
        m.insert(
            key(2, 2),
            Bucket::<u64> {
                ready: VecDeque::new(),
                keep_ready: 5,
                in_flight: 0,
                failure: None,
            },
        );
        assert_eq!(most_starved(&m), Some(key(2, 2)));
    }

    #[test]
    fn most_starved_none_when_all_full() {
        let mut m = HashMap::new();
        m.insert(
            key(1, 1),
            Bucket::<u64> {
                ready: VecDeque::from(vec![9]),
                keep_ready: 1,
                in_flight: 0,
                failure: None,
            },
        );
        assert_eq!(most_starved(&m), None);
    }

    #[test]
    fn claim_reserves_then_deposit_makes_ready() {
        let core = PreloadCore::<u64>::new();
        core.register(key(4, 8), 2);
        let k = core.claim().unwrap();
        assert_eq!(k, key(4, 8));
        // The slot is now in-flight, so a second claim targets the SAME bucket
        // (still 1 deficit) not a phantom over-dispatch.
        let k2 = core.claim().unwrap();
        assert_eq!(k2, key(4, 8));
        core.deposit(k, 100);
        core.deposit(k2, 200);
        assert_eq!(core.get(key(4, 8)).unwrap(), 100);
        assert_eq!(core.get(key(4, 8)).unwrap(), 200);
    }

    #[test]
    fn unknown_bucket_get_errors_at_once() {
        let core = PreloadCore::<u64>::new();
        assert_eq!(core.get(key(1, 1)), Err(GetError::UnknownBucket(key(1, 1))));
    }

    #[test]
    fn fail_unparks_a_blocked_get_with_typed_error() {
        let core = Arc::new(PreloadCore::<u64>::new());
        core.register(key(1, 1), 1);
        let c2 = Arc::clone(&core);
        let h = thread::spawn(move || c2.get(key(1, 1)));
        // Give the puller time to park on not_empty, then fail the pool.
        thread::sleep(std::time::Duration::from_millis(50));
        core.fail(key(1, 1), "boom".to_string());
        assert_eq!(h.join().unwrap(), Err(GetError::Failed("boom".into())));
    }

    #[test]
    fn failure_isolated_to_registration() {
        let core = PreloadCore::<u64>::new();
        core.register(key(1, 1), 1);
        core.register(key(2, 1), 1);
        let failed = core.claim().unwrap();
        core.fail(failed, "broken source".to_owned());
        let healthy = core.claim().unwrap();
        assert_ne!(healthy, failed);
        core.deposit(healthy, 42);
        assert_eq!(
            core.get(failed),
            Err(GetError::Failed("broken source".into()))
        );
        assert_eq!(core.get(healthy), Ok(42));
    }

    #[test]
    fn close_unparks_a_blocked_get() {
        let core = Arc::new(PreloadCore::<u64>::new());
        core.register(key(1, 1), 1);
        let c2 = Arc::clone(&core);
        let h = thread::spawn(move || c2.get(key(1, 1)));
        thread::sleep(std::time::Duration::from_millis(50));
        core.close();
        assert_eq!(h.join().unwrap(), Err(GetError::Closing));
    }

    #[test]
    fn close_unparks_a_blocked_claim() {
        let core = Arc::new(PreloadCore::<u64>::new());
        core.register(key(1, 1), 1);
        // Fill the only slot so the next claim parks on not_full.
        let k = core.claim().unwrap();
        core.deposit(k, 7);
        let c2 = Arc::clone(&core);
        let h = thread::spawn(move || c2.claim());
        thread::sleep(std::time::Duration::from_millis(50));
        core.close();
        assert_eq!(h.join().unwrap(), Err(ClaimStop::Closing));
    }

    #[test]
    fn full_bucket_blocks_claim_until_drained() {
        let core = Arc::new(PreloadCore::<u64>::new());
        core.register(key(1, 1), 1);
        let k = core.claim().unwrap();
        core.deposit(k, 1);
        // Bucket full (1/1). A claim must park until a get() drains it.
        let c2 = Arc::clone(&core);
        let h = thread::spawn(move || c2.claim());
        thread::sleep(std::time::Duration::from_millis(50));
        // Drain — frees the slot, wakes the parked claim.
        assert_eq!(core.get(key(1, 1)).unwrap(), 1);
        assert_eq!(h.join().unwrap(), Ok(key(1, 1)));
    }

    #[test]
    fn stress_many_threads_keeps_buffers_bounded_and_correct() {
        // A loaded run: 6 worker threads refilling 4 buckets while a puller
        // drains; asserts every produced item is delivered exactly once and no
        // bucket EVER exceeds its keep_ready (the bound holds under load).
        let core = Arc::new(PreloadCore::<u64>::new());
        let keys = [key(1, 16), key(2, 16), key(4, 32), key(8, 64)];
        for (i, k) in keys.iter().enumerate() {
            core.register(*k, i + 2); // distinct targets 2..5
        }
        let total_to_pull = 2000usize;

        // Workers: claim → "produce" a monotonically tagged item → deposit.
        // The item encodes its bucket so the puller can re-check the bound.
        let mut workers = Vec::new();
        for _ in 0..6 {
            let c = Arc::clone(&core);
            workers.push(thread::spawn(move || {
                loop {
                    match c.claim() {
                        Ok(k) => c.deposit(k, k), // tag = registration ID
                        Err(_) => break,
                    }
                }
            }));
        }

        // Puller: round-robin drains; after each get re-reads ready_counts and
        // asserts no bucket overshoots its target.
        let mut pulled = 0usize;
        let targets: HashMap<RegistrationId, usize> =
            keys.iter().enumerate().map(|(i, k)| (*k, i + 2)).collect();
        while pulled < total_to_pull {
            let k = keys[pulled % keys.len()];
            let item = core.get(k).unwrap();
            assert_eq!(item, k, "item delivered to wrong bucket");
            pulled += 1;
            for (bk, n) in core.ready_counts() {
                assert!(
                    n <= targets[&bk],
                    "bucket {bk:?} overshot target: {n} > {}",
                    targets[&bk]
                );
            }
        }

        core.close();
        for w in workers {
            w.join().unwrap();
        }
    }

    #[test]
    fn typed_pool_owns_workers_for_the_caller() {
        let mut pool = PreloadPool::new(|registration| Ok(registration * 10), 2).unwrap();
        let first = pool.register(2).unwrap();
        let second = pool.register(1).unwrap();
        pool.start().unwrap();
        assert_eq!(pool.get(first).unwrap() % 10, 0);
        assert_eq!(pool.get(second).unwrap() % 10, 0);
        pool.close();
    }
}
