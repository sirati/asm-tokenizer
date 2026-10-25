//! Synchronous native task groups on the existing bounded persistent pool.
//! The caller owns task data. Every callback finishes before run returns.
use crate::{PreloadPool, RegistrationId};
use std::{
    ffi::{c_int, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex},
};

pub type Callback = unsafe extern "C" fn(*mut c_void, usize) -> c_int;
type Producer = Box<dyn Fn(RegistrationId) -> Result<bool, String> + Send + Sync>;

struct Work {
    callback: Callback,
    context: usize,
    next: usize,
    count: usize,
}
struct Control {
    pool: PreloadPool<bool, Producer>,
    registration: RegistrationId,
    workers: usize,
}
struct Executor {
    control: Mutex<Control>,
    work: Arc<Mutex<Option<Work>>>,
}
impl Executor {
    fn new(workers: usize) -> Result<Self, &'static str> {
        let work = Arc::new(Mutex::new(None::<Work>));
        let request = work.clone();
        let produce: Producer = Box::new(move |_| {
            let (callback, context, index) = {
                let mut guard = request.lock().unwrap();
                let current = guard.as_mut().ok_or("native CPU request is absent")?;
                if current.next >= current.count {
                    return Err("native CPU request exhausted".into());
                }
                let index = current.next;
                current.next += 1;
                (current.callback, current.context, index)
            };
            // The synchronous run barrier retains the caller's context until
            // this callback and every other in-flight callback have returned.
            Ok(unsafe { callback(context as *mut c_void, index) } == 0)
        });
        let mut pool = PreloadPool::with_persistence(produce, workers, true)?;
        let registration = pool.register_limit(workers, Some(0))?;
        pool.start()?;
        Ok(Self {
            control: Mutex::new(Control {
                pool,
                registration,
                workers,
            }),
            work,
        })
    }

    unsafe fn run(&self, callback: Callback, context: *mut c_void, count: usize) -> bool {
        if count == 0 {
            return true;
        }
        let control = self.control.lock().unwrap();
        let core = control
            .pool
            .consumer()
            .expect("native CPU executor is not running");
        core.cancel(control.registration)
            .expect("native CPU request cannot be cancelled");
        *self.work.lock().unwrap() = Some(Work {
            callback,
            context: context as usize,
            next: 0,
            count,
        });
        // Even an unexpected controller panic must quiesce callbacks before
        // the caller can release borrowed task data. Callback panics must be
        // caught by the callback itself; unwinding through C is not permitted.
        let completed = catch_unwind(AssertUnwindSafe(|| {
            control
                .pool
                .rearm_limit(control.registration, control.workers, Some(count))
                .expect("native CPU request cannot be rearmed");
            let mut successful = true;
            for _ in 0..count {
                match control.pool.get(control.registration) {
                    Ok(status) => successful &= status,
                    Err(_) => {
                        successful = false;
                        break;
                    }
                }
            }
            successful
        }));
        core.cancel(control.registration)
            .expect("native CPU completion barrier failed");
        *self.work.lock().unwrap() = None;
        completed.unwrap_or(false)
    }
}

/// Public C capsule ABI. Callbacks may borrow stack data because run always
/// waits for all callbacks, including on failure. Callbacks must not unwind or
/// recursively run the same executor. The capsule owner must outlive run.
#[repr(C)]
pub struct Api {
    version: u64,
    workers: usize,
    context: *mut c_void,
    run: unsafe extern "C" fn(*mut c_void, Callback, *mut c_void, usize) -> c_int,
}
impl Api {
    pub fn new(workers: usize) -> Result<Box<Self>, &'static str> {
        let workers = workers.min(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        );
        let executor = Box::new(Executor::new(workers)?);
        Ok(Box::new(Self {
            version: 1,
            workers,
            context: Box::into_raw(executor).cast(),
            run,
        }))
    }
}
impl Drop for Api {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.context.cast::<Executor>()));
        }
    }
}
unsafe extern "C" fn run(
    executor: *mut c_void,
    callback: Callback,
    context: *mut c_void,
    count: usize,
) -> c_int {
    let executor = unsafe { &*executor.cast::<Executor>() };
    i32::from(!unsafe { executor.run(callback, context, count) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    unsafe extern "C" fn increment(context: *mut c_void, index: usize) -> c_int {
        let counts = unsafe { &*context.cast::<Vec<AtomicUsize>>() };
        counts[index].fetch_add(1, Ordering::SeqCst);
        0
    }
    #[test]
    fn native_groups_complete_every_index_once_across_reuse() {
        let executor = Api::new(4).unwrap();
        for count in [0, 1, 7, 256, 32] {
            let mut counts: Vec<_> = (0..count).map(|_| AtomicUsize::new(0)).collect();
            assert_eq!(
                unsafe {
                    (executor.run)(
                        executor.context,
                        increment,
                        (&mut counts as *mut Vec<_>).cast(),
                        count,
                    )
                },
                0
            );
            assert!(counts.iter().all(|value| value.load(Ordering::SeqCst) == 1));
        }
    }
    unsafe extern "C" fn fail_one(context: *mut c_void, index: usize) -> c_int {
        unsafe {
            increment(context, index);
        }
        i32::from(index == 2)
    }
    #[test]
    fn failed_group_still_completes_before_return_and_can_be_reused() {
        let executor = Api::new(4).unwrap();
        let mut counts: Vec<_> = (0..16).map(|_| AtomicUsize::new(0)).collect();
        assert_ne!(
            unsafe {
                (executor.run)(
                    executor.context,
                    fail_one,
                    (&mut counts as *mut Vec<_>).cast(),
                    16,
                )
            },
            0
        );
        assert!(counts.iter().all(|value| value.load(Ordering::SeqCst) == 1));
        assert_eq!(
            unsafe {
                (executor.run)(
                    executor.context,
                    increment,
                    (&mut counts as *mut Vec<_>).cast(),
                    16,
                )
            },
            0
        );
        assert!(counts.iter().all(|value| value.load(Ordering::SeqCst) == 2));
    }

    #[test]
    fn native_group_uses_the_available_worker_slots_concurrently() {
        use std::{
            sync::{Condvar, mpsc},
            time::Duration,
        };
        struct Context {
            entered: mpsc::Sender<usize>,
            release: Mutex<bool>,
            changed: Condvar,
        }
        unsafe extern "C" fn wait(context: *mut c_void, index: usize) -> c_int {
            let context = unsafe { &*context.cast::<Context>() };
            context.entered.send(index).unwrap();
            let mut released = context.release.lock().unwrap();
            while !*released {
                released = context.changed.wait(released).unwrap();
            }
            0
        }
        let executor = Executor::new(4).unwrap();
        let (entered, receive) = mpsc::channel();
        let context = Context {
            entered,
            release: Mutex::new(false),
            changed: Condvar::new(),
        };
        std::thread::scope(|scope| {
            let running = scope.spawn(|| unsafe {
                executor.run(wait, (&context as *const Context).cast_mut().cast(), 4)
            });
            let observed: Result<Vec<_>, _> = (0..4)
                .map(|_| receive.recv_timeout(Duration::from_secs(2)))
                .collect();
            *context.release.lock().unwrap() = true;
            context.changed.notify_all();
            assert!(running.join().unwrap());
            let mut observed = observed.unwrap();
            observed.sort_unstable();
            assert_eq!(observed, vec![0, 1, 2, 3]);
        });
    }
}
