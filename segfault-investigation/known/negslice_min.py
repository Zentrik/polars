import polars as pl, tempfile, os
print("polars", pl.__version__)
d = tempfile.mkdtemp()
# Minimal: parquet with small row groups; filter then negative slice over-returns rows (streaming).
n = 100
df = pl.DataFrame({"k": [i % 5 == 0 for i in range(n)], "i": range(n)})  # 20 rows match
p = os.path.join(d, "f.parquet"); df.write_parquet(p, row_group_size=7)
ref = df.filter("k").select("i").slice(-10, 5)   # last-10 window then take 5 -> 5 rows
for engine in ["in-memory", "streaming"]:
    got = pl.scan_parquet(p).filter("k").select("i").slice(-10, 5).collect(engine=engine)
    print(f"  {engine:10s}: got {got.height} rows {got['i'].to_list()} | ref {ref.height} {ref['i'].to_list()} {'BUG' if not got.equals(ref) else 'ok'}")
# vary row group size
print("row-group-size sweep (filter k, slice(-10,5), streaming), 20 matching rows:")
for rg in [1, 2, 3, 5, 7, 10, 13, 20, 50, 100, None]:
    p = os.path.join(d, "g.parquet"); df.write_parquet(p, row_group_size=rg)
    got = pl.scan_parquet(p).filter("k").select("i").slice(-10, 5).collect(engine="streaming")
    print(f"  rg={str(rg):5s}: got {got.height} (exp 5) {'BUG' if got.height != 5 else 'ok'}")
