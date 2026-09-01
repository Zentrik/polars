#!/usr/bin/env python3
"""Minimal deterministic reproducer for a process-abort in polars'
`collect(background=True)` cancellation path (polars-lazy exitable.rs).

Root cause: `LazyFrame::collect_concurrently` runs the query on the rayon pool
and finishes with `tx.send(result).unwrap()`. The returned `InProcessQuery`
owns the `Receiver` and its `Drop` sets the cancel token -- so dropping the
handle is intended. But if the handle (hence the `Receiver`) is dropped while
the query is still executing, the completing job's `tx.send()` returns
`SendError`, the `.unwrap()` panics *on a rayon worker thread*, and rayon turns
that into `abort()` -- taking down the whole process (SIGABRT), not just the
query.

Trigger: submit a background query, drop the handle immediately, let the query
finish. Present in polars 1.37.1 through current main. Fix is one line per arm:
`let _ = tx.send(result);` (ignore the error when the receiver is gone).

Run: python3 repro_exitable_abort.py          # expect: Aborted (core dumped)
"""
import sys
import time

import polars as pl


def heavy_lazy(seed: int) -> pl.LazyFrame:
    n = 3_000_000
    return (
        pl.select(pl.int_range(0, n, dtype=pl.Int64).alias("id"))
        .lazy()
        .with_columns(((pl.col("id") * 2654435761 + seed) % 1_000_003).alias("k"))
        .group_by("k")
        .agg(pl.sum("id").alias("s"), pl.mean("id").alias("m"), pl.len())
        .sort("k")
    )


def main():
    print(f"polars {pl.__version__}, threads={pl.thread_pool_size()}", flush=True)
    for i in range(200):
        # Submit a background (in-process) query...
        q = heavy_lazy(i).collect(background=True)
        # ...and drop the handle right away. Drop sets the cancel token and
        # drops the Receiver; the query is still running on the rayon pool.
        del q
        # Let the in-flight query finish and hit tx.send().unwrap().
        time.sleep(0.05)
        if i % 20 == 0:
            print(f"submitted+dropped {i}", flush=True)
    print("completed all iterations WITHOUT abort", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
