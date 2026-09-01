# Stress-testing polars 1.37.1 for the async-executor deque corruption (#27842)

Status of this document: **methodology + root-cause analysis are final; the
Findings section is updated as each campaign completes.**

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

_Campaigns in progress; this section is filled in as results land._

<!-- RESULTS_PLACEHOLDER -->

[pola-rs/polars#27842]: https://github.com/pola-rs/polars/issues/27842
[CVE-2021-32810]: https://github.com/crossbeam-rs/crossbeam/security/advisories/GHSA-pqqp-xmhj-wgcw
