# Stress harness for pola-rs/polars#27842

Hunts the crash class behind [pola-rs/polars#27842]: `crossbeam-deque 0.8.6`
panicking with *"attempt to subtract with overflow"* at `deque.rs:65`
(`Buffer::at`: `index & (self.cap - 1)` with `cap == 0`) inside
`steal_batch_and_pop()` in the polars-stream async executor — caught on an
overflow-checks build, and the presumptive identity of non-deterministic
segfaults on normal wrapping builds (`0 - 1` → `usize::MAX` → wild pointer).

A zero-capacity deque buffer is never legitimate (min capacity 64, power of
two), so any reproduction implies memory corruption / use-after-free in or
around the executor.

## Layout

- `scenarios.py` — workload library. Each scenario runs in a fresh process:
  sequential big streaming pipelines (scan/filter/group_by/join/sort/window/
  concat/sink), multiscan over many parquet files, rapid tiny queries
  (per-query `task_scope` churn), early-stop (`head`/`limit`/`slice` cancels
  hot producers), async/background collect cancellation, SIGINT-interrupt
  storms, concurrent submission from Python threads, mixed
  streaming + in-memory engines.
- `harness.py` — driver: samples an env config per iteration, spawns the
  scenario subprocess, classifies the exit (ok / segfault / rust_panic /
  heap_abort / hang / oom_kill / ...), enables core dumps, runs gdb for
  backtraces, appends to `results.jsonl`, keeps artifacts of bad runs.

## Stress axes

| axis | values | purpose |
|------|--------|---------|
| `POLARS_MAX_THREADS` | 4, 16, 64, 256 | oversubscribe 4 cores → workers permanently steal-hungry |
| `POLARS_IDEAL_MORSEL_SIZE` | 100000, 1024, 32, 1 | tiny morsels multiply task/steal events by orders of magnitude (row counts auto-scale) |
| tight buffers | `POLARS_DEFAULT_{LINEARIZER,DISTRIBUTOR,ZIP_HEAD}_BUFFER_SIZE=1` | maximize blocking/wakeup churn |
| hardened alloc | `MALLOC_PERTURB_`, `MALLOC_CHECK_=3`, jemalloc `junk:true` | turn latent UAF reads into loud crashes |
| `POLARS_MAX_CONCURRENT_SCANS` | unset, 4, 32 | vary multiscan concurrency |

## Run

```sh
python3 harness.py \
  --python /path/to/venv-with-polars/bin/python \
  --out /path/to/rundir --data /path/to/datadir \
  --minutes 90 --seed 1

# repeat one exact config:
python3 harness.py ... --iters 50 \
  --pin "scenario=early_stop,threads=256,morsel=32,tight=1,hardened=1"
```

Any outcome other than `ok`/`py_error` is a finding; grep stderr/backtraces
for `overflow`, `deque`, `crossbeam` for the #27842 signature.
