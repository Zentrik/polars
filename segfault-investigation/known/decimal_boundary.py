import polars as pl, tempfile, os, random
print("polars", pl.__version__)
d = tempfile.mkdtemp()
def run(vals, prec, tgt, dps=1):
    p = os.path.join(d, "f.parquet")
    pl.DataFrame({"c": pl.Series(vals, dtype=pl.Int64).cast(pl.Decimal(prec, 0))}).write_parquet(p, data_page_size=dps)
    got = pl.scan_parquet(p).filter(pl.col("c") == pl.lit(tgt, dtype=pl.Decimal(prec, 0))).collect().height
    return got, sum(1 for v in vals if v == tgt)
print("precision boundary ([1,-1] tgt 1, dps=1):")
for prec in [1, 9, 10, 18, 19, 20, 28, 38]:
    g, e = run([1, -1], prec, 1); print(f"  prec={prec:2d}: got {g} exp {e} {'BUG' if g!=e else 'ok'}")
print("sign combinations (prec 38, dps=1, tgt=first):")
for vals in [[1,-1],[10,20],[-10,-20],[1,2],[-1,-2],[5,-3],[-3,5],[0,1],[0,-1],[1,0]]:
    g, e = run(vals, 38, vals[0]); print(f"  vals={str(vals):10s} tgt={vals[0]}: got {g} exp {e} {'BUG' if g!=e else 'ok'}")
print("realistic: default page size, random mixed-sign decimal, various n:")
for n in [1000, 10000, 100000, 500000]:
    random.seed(1); vals = [random.randint(-10**15, 10**15) for _ in range(n)]
    tgt = vals[n//3]
    p = os.path.join(d, "g.parquet")
    pl.DataFrame({"c": pl.Series(vals, dtype=pl.Int64).cast(pl.Decimal(38, 0)), "i": range(n)}).write_parquet(p)  # DEFAULT settings
    import pyarrow.parquet as pq
    md = pq.ParquetFile(p).metadata
    got = pl.scan_parquet(p).filter(pl.col("c") == pl.lit(tgt, dtype=pl.Decimal(38, 0))).collect().height
    print(f"  n={n:6d} rowgroups={md.num_row_groups}: got {got} exp 1 {'BUG' if got != 1 else 'ok'}")
print("does >= or < also mis-prune? (prec38, [1,-1], dps=1):")
p = os.path.join(d, "h.parquet"); pl.DataFrame({"c": pl.Series([1,-1], dtype=pl.Int64).cast(pl.Decimal(38,0))}).write_parquet(p, data_page_size=1)
for label, pred, exp in [("== 1", pl.col("c")==pl.lit(1,dtype=pl.Decimal(38,0)), 1), ("== -1", pl.col("c")==pl.lit(-1,dtype=pl.Decimal(38,0)), 1), (">= 0", pl.col("c")>=pl.lit(0,dtype=pl.Decimal(38,0)), 1), ("< 0", pl.col("c")<pl.lit(0,dtype=pl.Decimal(38,0)), 1), ("> -5", pl.col("c")>pl.lit(-5,dtype=pl.Decimal(38,0)), 2), ("is_in[1]", pl.col("c").is_in([pl.lit(1,dtype=pl.Decimal(38,0))]), 1)]:
    got = pl.scan_parquet(p).filter(pred).collect().height
    print(f"  {label:10s}: got {got} exp {exp} {'BUG' if got!=exp else 'ok'}")
