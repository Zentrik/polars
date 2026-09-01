import polars as pl, sys, datetime as dt
print("polars", pl.__version__)
def run(label, lf, engine):
    try:
        out = lf.collect(engine=engine)
        lens = {c: out[c].len() for c in out.columns}
        bad = {c: n for c, n in lens.items() if n != out.height}
        print(f"  {label:60s} {engine:10s} height={out.height} {'BAD ' + str(bad) if bad else 'ok'}")
    except BaseException as e:
        print(f"  {label:60s} {engine:10s} EXC {type(e).__name__}: {str(e)[:80]}")
n = 16
base = pl.DataFrame({"v": [1] * n, "x": [1.0] * n, "i": list(range(n)), "s": ["a"] * n})
mixed = pl.DataFrame({"v": [1, 1, 1, 2, 2, 3], "x": [1.0] * 6, "i": list(range(6)), "s": ["a"] * 6})
for engine in ["streaming", "in-memory"]:
    print(engine)
    run("all-empty windows: sum(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").sum()), engine)
    run("all-empty windows: sum(i) int", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("i").sum()), engine)
    run("all-empty windows: mean(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").mean()), engine)
    run("all-empty windows: min(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").min()), engine)
    run("all-empty windows: max(i)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("i").max()), engine)
    run("all-empty windows: first(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").first()), engine)
    run("all-empty windows: last(s)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("s").last()), engine)
    run("all-empty windows: len", base.lazy().rolling("v", period="1i", closed="left").agg(pl.len()), engine)
    run("all-empty windows: count(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").count()), engine)
    run("all-empty windows: n_unique(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").n_unique()), engine)
    run("all-empty windows: std(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").std()), engine)
    run("all-empty windows: quantile(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").quantile(0.5)), engine)
    run("all-empty windows: implode x", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x")), engine)
    run("all-empty windows: sum(x)+first(x)", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").sum(), pl.col("x").first()), engine)
    run("mixed windows: sum(x)", mixed.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").sum()), engine)
    run("mixed windows: mean(x)", mixed.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").mean()), engine)
    run("closed=none 1i: sum(x)", mixed.lazy().rolling("v", period="1i", closed="none").agg(pl.col("x").sum()), engine)
    run("closed=right 1i: sum(x)", mixed.lazy().rolling("v", period="1i", closed="right").agg(pl.col("x").sum()), engine)
    run("all-empty group_by_dynamic sum(x)", base.lazy().group_by_dynamic("v", every="1i", period="0i", closed="left").agg(pl.col("x").sum()), engine)
    run("rolling by group: sum(x)", base.with_columns(g=pl.Series([0,1]*(n//2))).lazy().rolling("v", period="1i", closed="left", group_by="g").agg(pl.col("x").sum()), engine)
    dates = pl.DataFrame({"d": [dt.date(2020,1,1)] * 5, "x": [1.0] * 5})
    run("dates all-empty: sum(x)", dates.lazy().rolling("d", period="1d", closed="left").agg(pl.col("x").sum()), engine)
    run("dates all-empty: sum(x) offset", dates.lazy().rolling("d", period="1d", offset="-5d").agg(pl.col("x").sum()), engine)
    run("eager DataFrame.rolling", pl.LazyFrame(base.rolling("v", period="1i", closed="left").agg(pl.col("x").sum())), engine)
    run("all-empty: sum(x) then to_dicts guard", base.lazy().rolling("v", period="1i", closed="left").agg(pl.col("x").sum(), pl.col("x").first()), engine)
