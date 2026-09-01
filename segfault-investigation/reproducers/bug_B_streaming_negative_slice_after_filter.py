#!/usr/bin/env python3
"""
BUG B (NEW / unreported): on the STREAMING engine, scan_parquet().filter(...).slice(
negative_offset, length) returns TOO MANY rows when the file has small row groups.

Affected: polars 1.37.1 through 1.44.1 (latest). Streaming engine only
          (in-memory engine is correct). Silent wrong results.

Trigger conditions:
  * source is a scan (scan_parquet) with more than one row group
    (row_group_size smaller than the number of matching rows)
  * a filter is applied
  * a slice with NEGATIVE offset AND an explicit length is taken
    (df.slice(-10, 5) / df.tail() with a length). slice(-10) without a length
    is fine; positive offsets are fine.
The negative-offset slice is mis-combined with the per-row-group filter output,
so extra rows leak past the requested length.
"""
import tempfile, os
import polars as pl

print("polars", pl.__version__)
d = tempfile.mkdtemp()
p = os.path.join(d, "f.parquet")

# 100 rows, every 5th matches (20 matches); small row groups of 7.
df = pl.DataFrame({"k": [i % 5 == 0 for i in range(100)], "i": range(100)})
df.write_parquet(p, row_group_size=7)

ref = df.filter("k").select("i").slice(-10, 5)          # correct: 5 rows
for engine in ("in-memory", "streaming"):
    got = pl.scan_parquet(p).filter("k").select("i").slice(-10, 5).collect(engine=engine)
    print(f"  {engine:10s} filter + slice(-10, 5): got {got.height} rows "
          f"{got['i'].to_list()}  (expected {ref.height}: {ref['i'].to_list()})  ->",
          "BUG" if not got.equals(ref) else "ok")

print("\nonly negative offset + explicit length is affected (streaming):")
for off, ln in [(-10, 5), (-3, 1), (-10, None), (0, 5), (5, 3)]:
    q = pl.scan_parquet(p).filter("k").select("i")
    q = q.slice(off, ln) if ln is not None else q.slice(off)
    ref2 = df.filter("k").select("i")
    ref2 = ref2.slice(off, ln) if ln is not None else ref2.slice(off)
    got = q.collect(engine="streaming")
    print(f"  slice({off}, {ln}): got {got.height} expected {ref2.height}  ->",
          "BUG" if got.height != ref2.height else "ok")

print("\nrow-group-size sweep (filter + slice(-10, 5), 20 matching rows, streaming):")
for rg in (1, 3, 7, 13, 50, None):
    df.write_parquet(p, row_group_size=rg)
    got = pl.scan_parquet(p).filter("k").select("i").slice(-10, 5).collect(engine="streaming")
    print(f"  row_group_size={str(rg):4s}: got {got.height} (expected 5)  ->",
          "BUG" if got.height != 5 else "ok")
