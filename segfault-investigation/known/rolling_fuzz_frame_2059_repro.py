import polars as pl, sys
df = pl.read_ipc("rolling_fuzz_frame_2059.ipc", rechunk=False)
d = df.sort("v")
aggs = [pl.col(c).first().alias(c+"_f") for c in df.columns if c != "v"] + [pl.len()] + [pl.col(c).sum().alias(c+"_s") for c, t in df.schema.items() if t.is_numeric() and c != "v"]
out = d.lazy().rolling(index_column="v", period="1i", closed="left").agg(aggs).collect(engine=sys.argv[1])
print("collected", out.shape, flush=True)
for i in range(3):
    try:
        out.head(3).to_dicts(); print("to_dicts ok", flush=True)
    except BaseException as e:
        print("to_dicts EXC", repr(e)[:300], flush=True)
print("rows:", flush=True)
try:
    out.rows(); print("rows ok")
except BaseException as e:
    print("rows EXC", repr(e)[:300], flush=True)
print("done", flush=True)
