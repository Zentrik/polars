//! Stress driver for the polars-stream async executor, targeting
//! pola-rs/polars#27842 (capacity underflow / corruption in crossbeam-deque's
//! `steal_batch_and_pop` under heavy work stealing).
//!
//! The `async_executor` module here is copied byte-for-byte from polars
//! py-1.37.1 (only its `use` lines for three polars-internal symbols were
//! repointed at `shims`). Built with overflow-checks (see Cargo.toml), the
//! underflow panics at `crossbeam-deque .../deque.rs:65` exactly as in the
//! issue's backtrace instead of silently corrupting memory.
//!
//! Strategy: oversubscribe the executor (threads >> cores) so workers are
//! almost always steal-hungry, then drive the per-thread LIFO queues through
//! rapid grow/shrink cycles (deep spawn bursts drained by concurrent steals)
//! and race `task_scope` teardown cancellation against in-flight stealing.

#![allow(clippy::disallowed_names)]

mod async_executor;
mod shims;

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::{Duration, Instant};

use async_executor::{spawn, task_scope, AbortOnDropHandle, JoinHandle, TaskPriority};

static COMPLETED: AtomicU64 = AtomicU64::new(0);
static SCOPES: AtomicU64 = AtomicU64::new(0);
static SPAWNED: AtomicU64 = AtomicU64::new(0);
static POLLS: AtomicU64 = AtomicU64::new(0);

// --- a self-rescheduling yield, to churn the scheduler queues -------------

struct YieldNow(u32);
impl Future for YieldNow {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 == 0 {
            Poll::Ready(())
        } else {
            self.0 -= 1;
            POLLS.fetch_add(1, Ordering::Relaxed);
            // Reschedule ourselves: drives schedule_task -> local slot/queue
            // eviction and makes this task stealable again next round.
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}
fn yield_times(n: u32) -> YieldNow {
    YieldNow(n)
}

// --- block the calling (main) thread on a JoinHandle ----------------------
// The task itself runs on the executor's worker threads; this only parks the
// main thread until the root task signals completion.

struct ThreadWaker(std::thread::Thread);
impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<T>(handle: JoinHandle<T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut handle = handle;
    let mut pinned = unsafe { Pin::new_unchecked(&mut handle) };
    loop {
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::thread::park_timeout(Duration::from_millis(50)),
        }
    }
}

fn leaf_work(seed: u64) -> u64 {
    // A little non-trivial work so tasks don't compile away to nothing.
    let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
    for _ in 0..8 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x ^= x >> 29;
    }
    x
}

// --- workload modes -------------------------------------------------------

/// Deep local spawn bursts drained by concurrent steals: maximizes buffer
/// grow/shrink churn in the per-thread LIFO queues (the exact crash site).
/// `generators` concurrent producers each fill their *own* worker's local
/// queue, so many deque buffers grow and shrink at once under cross-thread
/// stealing, widening the epoch-reclamation UAF window.
fn mode_fanout(generators: usize, burst: usize, rounds: usize) {
    let handle = spawn(TaskPriority::High, async move {
        task_scope(|scope| {
            let root = scope.spawn_task(TaskPriority::High, async move {
                let mut gens = Vec::with_capacity(generators);
                for g in 0..generators {
                    gens.push(scope.spawn_task(TaskPriority::High, async move {
                        for r in 0..rounds {
                            // Wide burst -> this worker's local LIFO queue grows.
                            for i in 0..burst {
                                let s = ((g * rounds + r) * burst + i) as u64;
                                SPAWNED.fetch_add(1, Ordering::Relaxed);
                                scope.spawn_task(TaskPriority::High, async move {
                                    // Near-zero work + rare yield so the queue is
                                    // drained fast by steals -> forces shrink.
                                    if s % 8 == 0 {
                                        yield_times(1).await;
                                    }
                                    std::hint::black_box(leaf_work(s));
                                    COMPLETED.fetch_add(1, Ordering::Relaxed);
                                });
                            }
                            // Yield so steals drain our queue before next burst.
                            yield_times(1).await;
                        }
                    }));
                }
                for h in gens {
                    h.await;
                }
            });
            block_on(root);
        });
        SCOPES.fetch_add(1, Ordering::Relaxed);
    });
    block_on(handle);
}

/// Many tasks that reschedule themselves repeatedly: steady steal pressure
/// with tasks constantly re-entering queues from many workers at once.
fn mode_yield_storm(count: usize, yields: u32) {
    let handle = spawn(TaskPriority::High, async move {
        task_scope(|scope| {
            let root = scope.spawn_task(TaskPriority::High, async move {
                let mut handles = Vec::with_capacity(count);
                for i in 0..count {
                    SPAWNED.fetch_add(1, Ordering::Relaxed);
                    handles.push(scope.spawn_task(TaskPriority::High, async move {
                        yield_times(yields).await;
                        std::hint::black_box(leaf_work(i as u64));
                        COMPLETED.fetch_add(1, Ordering::Relaxed);
                    }));
                }
                for h in handles {
                    h.await;
                }
            });
            block_on(root);
        });
        SCOPES.fetch_add(1, Ordering::Relaxed);
    });
    block_on(handle);
}

/// Race scope teardown against live stealing: spawn a swarm of long-yielding
/// tasks, wait only until some have started, then let the scope end so
/// destroy() cancels the rest while they are being stolen/run. Prime suspect
/// path for the corruption.
fn mode_scope_churn(iters: usize, count: usize) {
    for _ in 0..iters {
        let started = Arc::new(AtomicUsize::new(0));
        let s2 = started.clone();
        // Run the whole scope on an executor worker so spawns land in a
        // worker-local queue (stealable), not the global injector.
        let handle = spawn(TaskPriority::High, async move {
            task_scope(|scope| {
                for i in 0..count {
                    let st = s2.clone();
                    SPAWNED.fetch_add(1, Ordering::Relaxed);
                    scope.spawn_task(TaskPriority::High, async move {
                        st.fetch_add(1, Ordering::Relaxed);
                        // Long yield loop so most are still in-flight at teardown.
                        yield_times(10_000).await;
                        std::hint::black_box(leaf_work(i as u64));
                        COMPLETED.fetch_add(1, Ordering::Relaxed);
                    });
                }
                // Give workers a moment to pick up and start stealing, then
                // return -> TaskScope::destroy() cancels the rest mid-flight.
                let spin_until = (count / 4).max(1);
                let deadline = Instant::now() + Duration::from_millis(50);
                while s2.load(Ordering::Relaxed) < spin_until && Instant::now() < deadline {
                    std::hint::spin_loop();
                }
            });
            SCOPES.fetch_add(1, Ordering::Relaxed);
        });
        block_on(handle);
        let _ = started;
    }
}

/// Foreign-thread wake/spawn storm: OS threads inject tasks via the global
/// `spawn` entry point (unknown-thread scheduling -> global queue + unpark),
/// racing the executor's own stealing.
fn mode_foreign(threads: usize, per_thread: usize) {
    let stop = Arc::new(AtomicBool::new(false));
    let mut os = Vec::new();
    for t in 0..threads {
        let stop = stop.clone();
        os.push(std::thread::spawn(move || {
            let mut n = 0usize;
            while !stop.load(Ordering::Relaxed) && n < per_thread {
                let h = spawn(TaskPriority::High, async move {
                    SPAWNED.fetch_add(1, Ordering::Relaxed);
                    yield_times((n % 3) as u32).await;
                    std::hint::black_box(leaf_work((t * per_thread + n) as u64));
                    COMPLETED.fetch_add(1, Ordering::Relaxed);
                });
                // Randomly cancel some in-flight to exercise cancel-vs-steal.
                // Drop the handle either way: the scheduler keeps its own ref so
                // the task still runs to completion, and dropping frees the
                // task's Arc once done. (Do NOT mem::forget it — that leaks the
                // Arc of every spawned task and OOMs a long soak.)
                if n % 7 == 0 {
                    drop(AbortOnDropHandle::new(h));
                } else {
                    drop(h);
                }
                n += 1;
                if n % 256 == 0 {
                    std::thread::yield_now();
                }
            }
        }));
    }
    for h in os {
        let _ = h.join();
    }
}

fn run_mixed(threads: usize, rng: &mut impl rand::Rng) {
    match rng.random_range(0..4) {
        0 => mode_fanout(threads, 512 + rng.random_range(0..1536), 4),
        1 => mode_yield_storm(4096 + rng.random_range(0..4096), 3 + rng.random_range(0..6)),
        2 => mode_scope_churn(6, 2048 + rng.random_range(0..2048)),
        _ => mode_foreign(threads.min(8).max(2), 20_000),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) * 16;
    let mut secs = 120u64;
    let mut mode = "mixed".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 2; }
            "--secs" => { secs = args[i + 1].parse().unwrap(); i += 2; }
            "--mode" => { mode = args[i + 1].clone(); i += 2; }
            other => { eprintln!("unknown arg {other}"); std::process::exit(2); }
        }
    }

    async_executor::set_num_threads(threads);
    println!(
        "executor-hammer: threads={threads} cores={} secs={secs} mode={mode}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
    );

    use rand::SeedableRng;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(0xC0FFEE);
    let start = Instant::now();
    let deadline = start + Duration::from_secs(secs);
    let mut iters = 0u64;
    let mut last_report = Instant::now();

    while Instant::now() < deadline {
        match mode.as_str() {
            "fanout" => mode_fanout(threads, 1536, 4),
            "yield_storm" => mode_yield_storm(6144, 5),
            "scope_churn" => mode_scope_churn(8, 3072),
            "foreign" => mode_foreign(threads.min(8).max(2), 40_000),
            _ => run_mixed(threads, &mut rng),
        }
        iters += 1;
        if last_report.elapsed() >= Duration::from_secs(5) {
            let el = start.elapsed().as_secs_f64();
            let done = COMPLETED.load(Ordering::Relaxed);
            let sp = SPAWNED.load(Ordering::Relaxed);
            let polls = POLLS.load(Ordering::Relaxed);
            println!(
                "[{el:6.1}s] iters={iters} scopes={} spawned={sp} completed={done} \
                 polls={polls} ({:.2}M sched-ops/s)",
                SCOPES.load(Ordering::Relaxed),
                (done + polls) as f64 / el.max(1e-9) / 1e6,
            );
            last_report = Instant::now();
        }
    }

    let el = start.elapsed().as_secs_f64();
    println!(
        "DONE ok: iters={iters} scopes={} spawned={} completed={} in {el:.1}s ({:.0} tasks/s)",
        SCOPES.load(Ordering::Relaxed),
        SPAWNED.load(Ordering::Relaxed),
        COMPLETED.load(Ordering::Relaxed),
        COMPLETED.load(Ordering::Relaxed) as f64 / el.max(1e-9),
    );
}
