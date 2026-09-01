#!/usr/bin/env python3
"""
BUG A (NEW / unreported): scan_parquet predicate pushdown silently DROPS matching
rows for Decimal columns with precision >= 19, and corrupts the on-disk parquet
min/max statistics for every reader (pyarrow included).

Affected: polars 1.37.1 through 1.44.1 (latest), eager read+scan, both engines.
Severity: silent data loss on a filtered scan; corrupt file metadata.

Root cause (write side):
  crates/polars-parquet/src/arrow/write/fixed_size_binary/mod.rs :: build_statistics_decimal
  i128-backed Decimal (precision > 18) is stored as FIXED_LEN_BYTE_ARRAY with
  two's-complement min/max bytes, but the statistics are compared with UNSIGNED
  byte order during pruning. When a column-chunk / data-page min and max straddle
  zero, the stored min (e.g. -1 -> 0xFF..FF) is byte-GREATER than the stored max
  (e.g. 1 -> 0x00..01). A pruner then sees min > max, treats the range as empty,
  and prunes the chunk -- dropping the rows that actually match.

Trigger conditions (all required):
  * Decimal precision >= 19 (i128-backed). Precision <= 18 (i64-backed) is fine.
  * The pruned chunk/page's values straddle zero (mix of >=0 and <0). Same-sign
    chunks are fine.
  * Statistics/pruning actually engage (many pages, e.g. data_page_size small, or
    a real multi-row-group file whose chunk spans zero).
"""
import io, tempfile, os
import polars as pl

print("polars", pl.__version__)

# --- Minimal reproducer: two values straddling zero, two data pages -----------
df = pl.DataFrame({"c": pl.Series([1, -1], dtype=pl.Int64).cast(pl.Decimal(38, 0))})
buf = io.BytesIO()
df.write_parquet(buf, data_page_size=1)   # force >1 page so page/chunk stats apply

buf.seek(0)
got = pl.scan_parquet(buf).filter(pl.col("c") == pl.lit(1, dtype=pl.Decimal(38, 0))).collect()
print(f"filter(c == 1): got {got.height} row(s), expected 1  ->",
      "BUG (row dropped)" if got.height != 1 else "ok")

# --- The corruption is in the FILE: pyarrow reads swapped stats and also drops it
try:
    import pyarrow.parquet as pq, decimal
    buf.seek(0)
    md = pq.ParquetFile(buf).metadata.row_group(0).column(0).statistics
    print(f"on-disk stats written by polars: min={md.min}  max={md.max}   "
          f"(min > max is impossible for correct stats)")
    buf.seek(0)
    n = pq.read_table(buf, filters=[("c", "=", decimal.Decimal(1))]).num_rows
    print(f"pyarrow reading the polars file, filter c==1: {n} row(s) ->",
          "BUG (other engines affected too)" if n != 1 else "ok")
except ImportError:
    pass

# --- Precision boundary and sign condition ------------------------------------
def drops(vals, prec, target):
    b = io.BytesIO()
    pl.DataFrame({"c": pl.Series(vals, dtype=pl.Int64).cast(pl.Decimal(prec, 0))}).write_parquet(b, data_page_size=1)
    b.seek(0)
    got = pl.scan_parquet(b).filter(pl.col("c") == pl.lit(target, dtype=pl.Decimal(prec, 0))).collect().height
    return got != sum(1 for v in vals if v == target)

print("\nprecision boundary (values [1, -1], filter == 1):")
for prec in (18, 19, 38):
    print(f"  precision {prec:2d}: {'BUG' if drops([1, -1], prec, 1) else 'ok'}")

print("\nsign condition (precision 38, filter == first value):")
for vals in ([10, 20], [-10, -20], [1, -1], [5, -3], [0, -1]):
    straddles = min(vals) < 0 <= max(vals)
    print(f"  vals={str(vals):10s} straddles_zero={straddles!s:5s}: "
          f"{'BUG' if drops(vals, 38, vals[0]) else 'ok'}")
