#!/usr/bin/env python3
"""
BUG D (NEW / unreported): pl.col(...).rle() on the STREAMING engine panics with
`assertion failed: chunks.len() == 1` whenever the input column has >= 5 chunks.
This is a hard assert! in polars-core/src/series/builder.rs:168 (gather_extend),
reached from the streaming run-length-encoding node (polars-stream .../nodes/rle.rs),
so it fires on the RELEASE wheel too -- not just debug builds.

Affected: polars 1.37.1 through 1.44.1 (latest). Streaming engine only; the
in-memory engine is always fine. rle_id() is fine; only rle().

Why it looks "sporadic": it needs the column to be physically multi-chunk (>= 5
chunks) at the rle node, which happens naturally after reading many files,
concatenating/vstacking many frames, etc. -- so in a varied pipeline it fires only
sometimes.

Found with a debug-assertions build of 1.44.1; confirmed on the release wheel.
"""
import polars as pl

print("polars", pl.__version__)

def run(df, label):
    try:
        df.lazy().select(pl.col("p").rle()).collect(engine="streaming")
        print(f"  {label:38s} ok")
    except pl.exceptions.PanicException as e:
        print(f"  {label:38s} PANIC: {str(e).splitlines()[-1][:60]}")

# Minimal: a column with >= 5 chunks (single-element chunks here).
s = pl.concat([pl.Series("p", [i % 3]) for i in range(5)], rechunk=False)
run(pl.DataFrame({"p": s}), f"{s.n_chunks()} chunks, streaming rle()")

# Boundary: 4 chunks ok, 5 chunks panics.
for nchunks in (4, 5):
    s = pl.concat([pl.Series("p", [i % 3]) for i in range(nchunks)], rechunk=False)
    run(pl.DataFrame({"p": s}), f"{nchunks} chunks")

# Natural triggers (no manual rechunk=False needed):
import tempfile, os
d = tempfile.mkdtemp()
for i in range(6):
    pl.DataFrame({"p": [i % 3, i % 2, i % 4]}).write_parquet(os.path.join(d, f"{i}.parquet"))
run(pl.read_parquet(os.path.join(d, "*.parquet")), "read_parquet(6 files) then rle()")

acc = pl.DataFrame({"p": [0, 1]})
for i in range(6):
    acc.vstack(pl.DataFrame({"p": [i % 3, i % 2]}), in_place=True)
run(acc, "6x vstack then rle()")

# In-memory engine is unaffected:
s = pl.concat([pl.Series("p", [i % 3]) for i in range(8)], rechunk=False)
try:
    pl.DataFrame({"p": s}).lazy().select(pl.col("p").rle()).collect(engine="in-memory")
    print("  in-memory engine, 8 chunks           ok (bug is streaming-only)")
except pl.exceptions.PanicException:
    print("  in-memory engine                     PANIC")
