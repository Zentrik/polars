import polars as pl, sys
df = pl.DataFrame({"v": [1, 1, 1, 2, 2, 3], "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
out = df.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").sum()).collect(engine="streaming")
ok = out["x"].len() == out.height
print(f"polars {pl.__version__}: height={out.height} x.len={out['x'].len()} x={out['x'].to_list()} -> {'OK' if ok else 'BUG'}")
# expression-context variant
out2 = df.lazy().select(pl.col("v"), pl.col("x").sum().rolling(index_column="v", period="1i", closed="left")).collect(engine="streaming")
print(f"   expr .rolling(): height={out2.height} x.len={out2['x'].len()} x={out2['x'].to_list()} -> {'OK' if out2['x'].len()==out2.height else 'BUG'}")
