import polars as pl, pyarrow as pa, pyarrow.parquet as pq, tempfile, os, decimal
print("polars", pl.__version__, "pyarrow", pa.__version__)
d = tempfile.mkdtemp()
# 1. polars-written; inspect the page-index / column stats via pyarrow
p = os.path.join(d, "pl.parquet")
pl.DataFrame({"c": pl.Series([1, -1], dtype=pl.Int64).cast(pl.Decimal(38, 0))}).write_parquet(p, data_page_size=1)
md = pq.ParquetFile(p).metadata
rg = md.row_group(0); col = rg.column(0)
print("polars-written col stats: min", col.statistics.min, "max", col.statistics.max, "num_values", col.num_values)
# 2. pyarrow-written mixed-sign decimal128, filtered by polars pushdown
p2 = os.path.join(d, "pa.parquet")
t = pa.table({"c": pa.array([decimal.Decimal(1), decimal.Decimal(-1)], pa.decimal128(38, 0))})
pq.write_table(t, p2, data_page_size=1, write_page_index=True)
got = pl.scan_parquet(p2).filter(pl.col("c") == pl.lit(1, dtype=pl.Decimal(38, 0))).collect().height
print("pyarrow-written, polars pushdown filter ==1:", got, "exp 1", "BUG" if got != 1 else "ok")
# 3. can pyarrow itself filter it right (sanity)?
print("pyarrow read ==1:", pq.read_table(p2, filters=[("c", "=", decimal.Decimal(1))]).num_rows)
# 4. read polars-written with pyarrow filter
print("pyarrow read polars-file ==1:", pq.read_table(p, filters=[("c", "=", decimal.Decimal(1))]).num_rows)
# 5. predicate variants on polars pushdown (fixed syntax)
def scanfilter(pred): 
    return pl.scan_parquet(p).filter(pred).collect().height
L = lambda v: pl.lit(v, dtype=pl.Decimal(38, 0))
for name, pred, exp in [("== 1", pl.col("c")==L(1), 1), ("== -1", pl.col("c")==L(-1), 1),
                        (">= 0", pl.col("c")>=L(0), 1), ("<= 0", pl.col("c")<=L(0), 1),
                        ("< 0", pl.col("c")<L(0), 1), ("> 0", pl.col("c")>L(0), 1),
                        ("!= 5", pl.col("c")!=L(5), 2), ("is_in [1]", pl.col("c").is_in(pl.Series([1], dtype=pl.Decimal(38,0))), 1)]:
    got = scanfilter(pred); print(f"  pushdown {name:10s}: got {got} exp {exp} {'BUG' if got!=exp else 'ok'}")
