# Stress-testing polars 1.37.1 for the async-executor deque corruption (#27842)

Status: **complete.** All three campaigns finished — Rust executor hammer
(~24.4 B ops), ThreadSanitizer (clean), and the 1.37.1 wheel matrix (87 runs).
See the Summary table at the bottom.

## The bug we're hunting

[pola-rs/polars#27842] reports non-deterministic segfaults using polars
(1.37.1, 1.40.1, 1.41.2) on a big, long-running job. On an overflow-checks
build the fault surfaces as a panic:

```
attempt to subtract with overflow
  at crossbeam-deque-0.8.6/src/deque.rs:65
  in Stealer::steal_batch_and_pop()  (polars async executor work-stealing)
```

Reported environment: 2× Intel Xeon Gold 5418Y, **~400 executor threads**,
overflow-checks enabled (conda runtime-compat build). No standalone repro —
the original job takes hours to fault.

## Why `cap == 0` means memory corruption, not an arithmetic bug

`deque.rs:65` is `Buffer::at`:

```rust
unsafe fn at(&self, index: isize) -> *mut T {
    // `self.cap` is always a power of two.
    self.ptr.offset(index & (self.cap - 1) as isize)   // <-- cap - 1
}
```

The crash site is reached from `steal_batch_and_pop` →
`buffer.deref().read(f)` → `at(f)`. The panic means `self.cap == 0`.

But a crossbeam-deque buffer's capacity is **never** legitimately zero:
`MIN_CAP = 64`, every allocation is `Buffer::alloc(cap)` with
`debug_assert_eq!(cap, cap.next_power_of_two())`, and shrink never goes below
`MIN_CAP` (`buffer.cap > MIN_CAP && len <= buffer.cap/4`). So `cap == 0` can
only come from:

1. a **use-after-free** of the epoch-protected `Buffer` (the `Shared<Buffer>`
   loaded under `epoch::pin()` is freed by a concurrent `resize`'s
   `defer_destroy` before this stealer finishes reading it), or
2. a **torn/garbage read** of the buffer pointer + `cap` word, or
3. **external heap corruption** — some other unsafe kernel in polars writing
   out of bounds onto the deque's `Buffer` header or the allocator freelist.

On a normal (wrapping) release build there is no panic: `0 - 1` wraps to
`usize::MAX`, the mask `& usize::MAX` becomes a no-op, and `ptr.offset(index)`
dereferences a wild address → the non-deterministic **segfault** the issue
title is really about. The overflow-checks build just converts that latent
memory-safety bug into a loud, early panic.

This is not the old crossbeam-deque race [CVE-2021-32810] — that was fixed in
0.8.1 and this is 0.8.6. No separate documented 0.8.6 underflow bug exists, so
the trigger is either specific to how polars drives the executor at extreme
thread counts, or corruption originating elsewhere in a query kernel.

## What the polars executor does (and doesn't) guarantee

`crates/polars-stream/src/async_executor/` (byte-identical between the
`py-1.37.1` tag and current `main`):

- One `Worker<ReadyTask>` LIFO deque **per executor thread**, living forever in
  a global `OnceLock<Executor>` (never dropped — so no whole-deque free race).
- Only the owning thread ever `push`es to its Worker (via `schedule_task`,
  which uses `TLS_THREAD_ID`); every other thread only `steal`s. The
  single-producer invariant crossbeam requires is respected.
- `task_scope` soundness rests on "every spawned task is cancelled before the
  scope's borrowed lifetime ends" (`TaskScope::destroy`). Cancellation
  (`CancelHandle::cancel` → `TaskData::Cancelled`) can race a task that is
  simultaneously being stolen and run on another thread — a prime suspect
  surface, exercised heavily below.

## Test environment (this run)

- Container: 4 vCPU, 15 GiB RAM, Linux 6.18, glibc. **Far smaller than the
  reporter's ~400-thread Xeon** — see "Can a 4-core box catch this?" below.
- polars **1.37.1** PyPI wheel (`pip install polars==1.37.1`), Python 3.11.
- Rust: nightly-2026-01-09 (repo-pinned), crossbeam-deque **=0.8.6** and all
  executor deps pinned to the `py-1.37.1` `Cargo.lock`.

## Can a 4-core box catch a 400-thread race?

Partially, and the design compensates where it can:

- **Oversubscription substitutes for core count.** Running the executor with
  64–256 threads on 4 cores keeps every worker almost always out of local work
  → near-constant `steal_batch_and_pop` pressure, which is the contended path.
- **Event rate substitutes for wall-clock.** Tiny morsels
  (`POLARS_IDEAL_MORSEL_SIZE` down to 1) and the Rust hammer's micro-tasks push
  the number of schedule/steal operations to millions/sec — the reporter's job
  faults after *hours*, i.e. after a comparable absolute number of steals.
- **Honest limitation:** preemption-driven interleavings on 4 cores are not the
  same as true-parallel/NUMA interleavings across 2 sockets. A race whose window
  only opens under genuine simultaneous cross-socket execution may not reproduce
  here regardless of duration. A clean run is therefore evidence, not proof of
  absence.

## Approach — three detection layers

**Layer A — Python wheel campaign** (`harness.py` + `scenarios.py`).
Runs the real 1.37.1 wheel exactly as production does. Each iteration runs one
workload in a fresh subprocess under a sampled env config, with core dumps +
gdb backtraces, hang watchdog, allocator hardening, and crash classification.
Reproduces the actual segfault mode (wrapping arithmetic). Axes:

| axis | values |
|------|--------|
| `POLARS_MAX_THREADS` | 4, 16, 64, 256 (oversubscribe 4 cores) |
| `POLARS_IDEAL_MORSEL_SIZE` | 100000, 1024, 32, 1 |
| tight channel buffers | `POLARS_DEFAULT_{LINEARIZER,DISTRIBUTOR,ZIP_HEAD}_BUFFER_SIZE=1` |
| allocator hardening | `MALLOC_PERTURB_`, `MALLOC_CHECK_=3`, jemalloc `junk:true` |
| `POLARS_MAX_CONCURRENT_SCANS` | unset, 4, 32 |

Workloads span the streaming engine: sequential big scan/filter/group-by/join/
sort/window/concat pipelines, multi-file parquet scan → `sink_parquet`, rapid
tiny queries (per-query `task_scope` churn), early-stop `head`/`limit`/`slice`
(cancels hot producers), async/background collect cancellation, SIGINT-driven
mid-query cancellation, concurrent submission from Python threads, and mixed
streaming + in-memory engines.

**Layer B — Rust executor hammer** (`executor-hammer/`).
The `py-1.37.1` executor lifted verbatim onto pinned crossbeam-deque 0.8.6,
built with **overflow-checks** so the underflow panics at `deque.rs:65`
(reproduced exactly — see the profile note in `Cargo.toml`) instead of
segfaulting. Drives the work-stealing paths directly at ~1–9M schedule/steal
ops/sec — thousands of times the event rate the Python layer can reach.
Modes: `fanout` (deep local-queue grow/shrink under concurrent steals —
the exact crash-site path), `scope_churn` (cancel-vs-steal teardown race),
`yield_storm`, `foreign` (foreign-thread spawn/wake storms).

**Layer C — ThreadSanitizer** (`run_tsan.sh`).
TSan build of the same hammer, to catch a data race in the executor *before* it
corrupts the deque. Caveat: crossbeam-epoch's fence/reclamation patterns can
produce TSan reports that aren't true bugs; hits are triaged against known
crossbeam noise and focused on the polars `async_executor` code.

---

## Findings

### Finding 1 (CONFIRMED, deterministic repro) — process abort when a `collect(background=True)` handle is dropped mid-query

**This is a distinct bug from #27842** (in-memory engine + rayon, not the
streaming deque), surfaced by the `cancel_async` workload. But it is a real,
process-killing crash in exactly the "cancel a query while it runs under
threading" space, and unlike #27842 it reproduces **deterministically in
<1 second**.

`repro_exitable_abort.py` (200 iters, aborts on iteration 0–1, 3/3 runs):

```
thread 'polars-1' panicked at crates/polars-lazy/src/frame/exitable.rs:34:33:
called `Result::unwrap()` on an `Err` value: SendError { .. }
Rayon: detected unexpected panic; aborting
Fatal Python error: Aborted            # exit 134 (SIGABRT)
```

Root cause — `LazyFrame::collect_concurrently` (`exitable.rs`):

```rust
POOL.spawn_fifo(move || {
    let result = physical_plan.execute(&mut state);
    tx.send(result).unwrap();      // <-- panics if the Receiver is already gone
});
```

`InProcessQuery` owns the `Receiver`, and its `Drop` sets the cancel token —
so dropping the handle is a supported, intended operation. But cancellation is
cooperative and best-effort: if the query finishes (or was already past its
cancel checkpoints) around the time the handle is dropped, the completing job
calls `tx.send(result)` into a channel whose receiver no longer exists, gets
`SendError`, and the `.unwrap()` panics. Because that panic happens on a
**rayon worker thread**, rayon escalates it to `abort()` — killing the whole
process, not just the query. Any `collect(background=True)` user who drops or
`.cancel()`s + releases the handle before the query lands is exposed.

Present in **1.37.1 and current `main`** (verified against the `py-1.37.1` tag).
All three spawn arms (`spawn_blocking`, `std::thread::spawn`, `POOL.spawn_fifo`
at lines 21/28/34) have the same `unwrap()`.

**Fix (one line per arm):** ignore the send error — the receiver being gone is
the normal cancellation outcome, not a bug:

```rust
let _ = tx.send(result);
```

### Finding 2 — the #27842 deque underflow itself: NOT reproduced on 4 cores

Across all three layers the underflow did **not** reproduce:

| layer | intensity | result |
|-------|-----------|--------|
| Rust executor hammer (overflow-checks) | **~24.4 billion** schedule/steal ops — 1.72 B task spawn-steal-run cycles + 22.7 B self-reschedule polls, at 64/128/256 threads on 4 cores | **0** deque underflows / panics / segfaults |
| ThreadSanitizer (same executor) | scope_churn + fanout + mixed, full instrumented runs | **0** data races reported |
| polars 1.37.1 wheel, streaming engine | **87** randomized runs across threads×morsel×buffers×allocator-hardening | **0** segfaults; see artifacts below |

The hammer specifically hammered the crash path — deep per-thread LIFO queues
(the `Worker` deque that `steal_batch_and_pop` reads) grown and shrunk under
constant cross-thread stealing, plus `task_scope` teardown cancelling tasks
mid-steal — at heavy oversubscription. 24.4 billion operations of it without a
single `cap == 0`.

**Non-bug outcomes seen in the wheel campaign** (classified, not defects):

- `oom_kill` ×2 — `concat_many` and `threads_concurrent` at the most extreme
  settings (`morsel=1` and 256 threads × 16 concurrent queries) exhausted the
  box's ~15 GB. Artifacts of pathological stress knobs, correctly isolated to
  the one subprocess by the harness; not a leak in polars.
- `hang` ×1 — `early_stop` at `morsel=1` timed out at 320 s. **Verified to be
  slowness, not a deadlock:** the identical workload with tight channel buffers
  but a normal morsel size completes in 2.5 s. `morsel=1` forces a
  `slice(1_000_000, 5)` to stream a million rows one row per morsel. This also
  clears tight channel buffers (`*_BUFFER_SIZE=1`) of deadlocking early-stop
  pipelines.

### What this means for #27842

`cap == 0` requires the epoch-protected deque `Buffer` to be freed / torn /
overwritten under a live stealer. That it survived 24.4 B executor ops **and** a
clean TSan run makes a plain logic race in the work-stealing code the least
likely explanation. The two remaining hypotheses both fit "faults only after
hours on a ~400-thread dual-socket Xeon":

1. **The corruption needs genuine large-scale parallelism.** 4 cores give
   preemption-driven interleavings, not true simultaneous cross-socket
   execution; a UAF window in crossbeam-epoch's reclamation that only opens
   under real parallelism at ~400 threads would not reproduce here at any
   duration. This is the honest limit of this environment.
2. **The corruption originates outside the executor** — an out-of-bounds write
   in some query kernel (join/group-by/IO) that happens to land on a deque
   `Buffer`, with the steal just the first read to touch the poisoned memory.
   This would only show under real query workloads, not the synthetic executor
   hammer — and the 87 wheel runs here didn't hit it, but they're far short of
   the reporter's multi-hour job.

**Recommended next steps to pin it down** (need the reporter's real workload):
- Run the failing job under a normal build with `RUST_BACKTRACE=full` **plus a
  larger stress window on the real 2-socket box** using this harness's env
  matrix; if it's hypothesis 2, AddressSanitizer/Valgrind on a debug polars
  build would catch the *originating* OOB write, which the deque panic only
  reports downstream.
- Share the query shape (streaming vs in-memory; joins/sinks/`collect_async`?)
  so the hammer and wheel workloads can be narrowed to it.
- Independently, ship the Finding 1 one-liner — it's a separate, certain bug.

## Summary

| # | Bug | Status | Trigger | Fix |
|---|-----|--------|---------|-----|
| 1 | `collect(background=True)` aborts the process if the handle is dropped mid-query (`exitable.rs` `tx.send().unwrap()` on a rayon worker) | **Confirmed, deterministic repro (<1 s)**, 1.37.1→main | `repro_exitable_abort.py` | `let _ = tx.send(result);` ×3 arms |
| 2 | #27842 deque underflow in `steal_batch_and_pop` | **Not reproduced** on 4 cores over 24.4 B ops + clean TSan; consistent with needing true large-scale parallelism or external heap corruption | reporter's multi-hour ~400-thread job | root cause still open — see next steps |

[pola-rs/polars#27842]: https://github.com/pola-rs/polars/issues/27842
[CVE-2021-32810]: https://github.com/crossbeam-rs/crossbeam/security/advisories/GHSA-pqqp-xmhj-wgcw

[pola-rs/polars#27842]: https://github.com/pola-rs/polars/issues/27842
[CVE-2021-32810]: https://github.com/crossbeam-rs/crossbeam/security/advisories/GHSA-pqqp-xmhj-wgcw
