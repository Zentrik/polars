<!--
DRAFT comment for https://github.com/pola-rs/polars/issues/27842 — for your
review. NOT posted. Numbers marked {…} are finalized once the soaks complete.
-->

I spent some time trying to build a standalone reproducer for this on a 4-core
machine by substituting heavy oversubscription + event-rate for the ~400-thread
Xeon. Sharing in case it helps narrow things down.

**TL;DR:** I could **not** reproduce the `deque.rs:65` underflow on 4 cores,
despite ~{N} billion work-stealing operations against the exact executor code
plus a clean ThreadSanitizer run. That pushes me toward "the corruption needs
genuine large-scale parallelism, or it originates outside the executor" rather
than a plain logic bug in the work-stealing code. I did, however, find a
separate, deterministic process-abort in the `collect(background=True)`
cancellation path (details at the end).

### What `cap == 0` implies

`deque.rs:65` is `self.ptr.offset(index & (self.cap - 1))` in
`crossbeam_deque::Buffer::at`, reached from `Stealer::steal_batch_and_pop` →
`buffer.deref().read(f)`. Since `MIN_CAP = 64` and every buffer capacity is a
power of two, `cap == 0` can't happen through normal arithmetic — it means the
epoch-protected `Buffer` was **freed/torn/overwritten** underneath the stealer
(use-after-free or external heap corruption). On a normal (wrapping) build the
`0 - 1` becomes `usize::MAX`, the mask is a no-op, and `ptr.offset` dereferences
a wild address → the non-deterministic segfault. The overflow-checks build just
converts that into an early panic.

### What I tried (all on crossbeam-deque 0.8.6, executor code identical to the
release)

1. **Standalone executor hammer.** Lifted `polars-stream`'s `async_executor`
   (`mod.rs`/`task.rs`/`park_group.rs`, byte-identical to the `py-1.37.1` tag)
   into a tiny crate on pinned crossbeam-deque 0.8.6, built with
   `overflow-checks = true` so the underflow panics immediately. Drove it at
   64–256 executor threads on 4 cores (~{M}M schedule/steal ops/sec) with:
   deep per-thread LIFO bursts drained by concurrent steals (maximizing buffer
   grow/shrink — the exact crash path), self-rescheduling yield storms, and
   `task_scope` teardown that cancels tasks mid-steal. **{X} billion steal ops,
   zero underflows.**
2. **ThreadSanitizer** on the same harness (scope_churn / fanout / mixed):
   **no data races reported**, including the cancel-vs-steal teardown path.
3. **Real 1.37.1 wheel**, streaming engine, {K} randomized runs across
   thread-count × morsel-size (down to 1) × tight channel buffers × allocator
   hardening (`MALLOC_CHECK_=3`, jemalloc `junk:true`), over
   scan/join/group-by/sort/window/sink/cancel workloads: **no segfault.**

Caveat: preemption-driven interleavings on 4 cores are genuinely not the same as
true-parallel execution across 2 sockets, so this is evidence, not proof — a
race whose window only opens under real simultaneity may simply need the big
box. If anyone can share the query shape / plan (streaming vs in-memory, joins,
sinks, `collect_async`/background collect?), that would help me target it.

If it would help, I can share the harness.

### Separate finding while stress-testing: `collect(background=True)` aborts the
process if the handle is dropped mid-query

Not the same bug (in-memory engine + rayon, not the streaming deque), but a real
crash and deterministic. `LazyFrame::collect_concurrently` finishes with
`tx.send(result).unwrap()` on the rayon pool (`exitable.rs:21/28/34`).
`InProcessQuery` owns the `Receiver` and its `Drop` sets the cancel token — so
dropping the handle is intended — but if the query completes around the time the
handle/Receiver is dropped, the send hits `SendError`, the `.unwrap()` panics on
a rayon worker, and rayon escalates to `abort()`, killing the whole process:

```
thread 'polars-1' panicked at crates/polars-lazy/src/frame/exitable.rs:34:33:
called `Result::unwrap()` on an `Err` value: SendError { .. }
Rayon: detected unexpected panic; aborting
```

Reproduces in <1s on 1.37.1 and current `main`. Fix is one line per arm:
`let _ = tx.send(result);`. Happy to open a separate issue / PR if useful.
