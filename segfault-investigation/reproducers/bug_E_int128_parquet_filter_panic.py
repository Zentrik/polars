#!/usr/bin/env python3
"""
BUG E (NEW / unreported): filtering an Int128 column read from a Parquet file panics
(the query crashes), on both engines, on 1.44.1. You cannot filter an Int128 parquet
column at all.

Two distinct failure points, both reachable normally:
  * with statistics (the default), predicate pushdown reads the column's min/max
    stats and hits `todo!("{:?}", other)` at
    polars-parquet/src/arrow/read/statistics.rs:545 -- the stats deserializer handles
    (Decimal, FixedLenByteArray) but NOT (Int128, FixedLenByteArray(16));
  * with use_statistics=False, the read path instead fails with
    `cannot create a series of type 'i128' of arrow chunk with type 'FixedSizeBinary(16)'`.

Affected: polars 1.37.1 .. 1.44.1 (latest), in-memory AND streaming.
Int128 is a stable public dtype. A missing stats type should skip pruning, not panic.

Found with the debug-assertions 1.44.1 build; reproduces on the plain release wheel.
"""
import io
import polars as pl

print("polars", pl.__version__)
df = pl.DataFrame({"c": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int128)})
buf = io.BytesIO()
df.write_parquet(buf)  # default writer writes statistics

def go(label, fn):
    try:
        print(f"  {label:36s} -> {fn()}")
    except pl.exceptions.PanicException as e:
        print(f"  {label:36s} -> PANIC: {str(e).splitlines()[-1][:60]}")
    except Exception as e:
        print(f"  {label:36s} -> {type(e).__name__}: {str(e)[:55]}")

buf.seek(0); go("read_parquet, no filter", lambda: pl.read_parquet(buf).shape)
buf.seek(0); go("scan + filter (in-memory)", lambda: pl.scan_parquet(buf).filter(pl.col("c") > 2).collect().shape)
buf.seek(0); go("scan + filter (streaming)", lambda: pl.scan_parquet(buf).filter(pl.col("c") > 2).collect(engine="streaming").shape)
buf.seek(0); go("scan + filter, use_statistics=False", lambda: pl.scan_parquet(buf, use_statistics=False).filter(pl.col("c") > 2).collect().shape)

# Contrast: Int64 / UInt64 are fine.
for dt in (pl.Int64, pl.UInt64):
    b = io.BytesIO(); pl.DataFrame({"c": pl.Series([1, 2, 3], dtype=dt)}).write_parquet(b); b.seek(0)
    go(f"{dt} scan + filter", lambda b=b: pl.scan_parquet(b).filter(pl.col("c") > 1).collect().shape)
