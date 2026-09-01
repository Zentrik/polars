"""Memory-pressure streaming fuzzer. Runs LARGE streaming queries (group_by / join /
over / sort on millions of rows with nulls and low-cardinality keys) so that, inside a
tight cgroup memory limit, the streaming engine's spilling / allocation-failure paths
are exercised. A graceful OOM should raise a polars error or MemoryError; a SIGSEGV
(139) or non-graceful SIGABRT (134) indicates an allocation-failure-under-pressure bug
(cf. open issue #29020). Reuses random query shapes but at scale.
Usage: python fuzz_mempressure.py SEED NITER LOGFILE [REPRODIR]
"""
import sys, os, random, warnings
warnings.simplefilter("ignore")
import polars as pl, numpy as np

seed=int(sys.argv[1]); niter=int(sys.argv[2]); logf=open(sys.argv[3],"a",buffering=1)
reprodir=sys.argv[4] if len(sys.argv)>4 else None
rng=random.Random(seed); np.random.seed(seed % (2**32))
def log(m): logf.write(f"[seed={seed}] {m}\n"); logf.flush()

def big_df(n):
    g = rng.choice([8, 64, 2000, 20000])   # cardinality
    cols = {
        "g": np.random.randint(0, g, n).astype(np.int64),
        "h": np.random.randint(0, rng.choice([2, 4, 16]), n).astype(np.int64),
        "k": np.random.randint(0, 10, n).astype(np.int64),
    }
    # value cols, some with nulls
    for i in range(rng.randint(2, 5)):
        v = np.random.randn(n) if rng.random() < 0.5 else np.random.randint(0, 10**6, n).astype(np.float64)
        s = pl.Series(f"v{i}", v)
        if rng.random() < 0.5:
            s = pl.select(pl.when(pl.Series(np.random.random(n) < 0.3)).then(None).otherwise(s).alias(f"v{i}")).to_series()
        cols[f"v{i}"] = s
    # sometimes a string col
    if rng.random() < 0.5:
        pool = ["a", "bb", "ccc", "x"*13, "y"*40, None]
        cols["s"] = pl.Series("s", np.random.choice(pool, n))
    return pl.DataFrame(cols)

for it in range(niter):
    n = rng.choice([1_000_000, 3_000_000, 6_000_000, 12_000_000])
    try:
        df = big_df(n)
    except MemoryError:
        log(f"iter {it} df alloc MemoryError n={n}"); continue
    except Exception as e:
        continue
    kind = rng.choice(["over_multi", "groupby", "join", "sort", "unique", "over_cumsum"])
    log(f"iter {it} kind={kind} n={n} schema={df.schema}")
    try:
        lf = df.lazy()
        vcols = [c for c in df.columns if c.startswith("v")]
        if kind == "over_multi":
            W = rng.choice([8, 24, 48])
            exprs = {f"c{i}": (pl.col(rng.choice(vcols)) + i).sum().over(rng.choice([["g"], ["g", "h"]])) for i in range(W)}
            lf = lf.with_columns(**exprs).select("g", *exprs)
        elif kind == "over_cumsum":
            W = rng.choice([8, 24])
            exprs = {f"c{i}": pl.col(rng.choice(vcols)).cum_sum().over(["g", "h"]) for i in range(W)}
            lf = lf.with_columns(seg=(pl.col("k") < 4).cum_sum().over("g")).with_columns(**exprs).select("g", "seg", *exprs)
        elif kind == "groupby":
            aggs = []
            for c in vcols: aggs += [pl.col(c).sum(), pl.col(c).mean(), pl.col(c).min(), pl.col(c).max(), pl.col(c).n_unique()]
            lf = lf.group_by(rng.choice([["g"], ["g", "h"], ["g", "h", "k"]]), maintain_order=rng.random()<0.5).agg(aggs)
        elif kind == "join":
            other = df.select("g", *vcols[:2]).lazy()
            lf = lf.join(other, on="g", how=rng.choice(["inner", "left"]), coalesce=True)
        elif kind == "sort":
            lf = lf.sort(["g", "h"] + vcols[:1], descending=rng.random()<0.5)
        elif kind == "unique":
            lf = lf.unique(subset=["g", "h", "k"], maintain_order=rng.random()<0.5)
        out = lf.collect(engine="streaming")
        # touch result
        out.head(3).to_dicts(); out.null_count()
        del out, df, lf
    except (pl.exceptions.PolarsError, MemoryError) as e:
        log(f"iter {it} handled {type(e).__name__}: {str(e)[:100]}")
    except pl.exceptions.PanicException as e:
        log(f"iter {it} PANIC kind={kind}: {str(e)[:250]}")
log("done")
