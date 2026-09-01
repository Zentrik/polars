"""Scaled-down variant of upstream issue #29020 (streaming .over() under pressure).
Usage: python over_stress.py generate DATA ROWS
       python over_stress.py run DATA ITERS OUTFILE [engine] [source]
Each iteration collects the query with the streaming engine and compares a checksum
against the in-memory engine result computed once.
"""
import os, sys, time, hashlib
from pathlib import Path
import numpy as np
import polars as pl

def generate(data: Path, rows: int, shards=6, groups=20_000):
    data.mkdir(exist_ok=True, parents=True)
    for shard in range(shards):
        rng = np.random.default_rng(shard)
        n = rows // shards
        pl.DataFrame({
            "g": pl.Series(rng.integers(0, groups, n), dtype=pl.Int64),
            "k": pl.Series(rng.integers(0, 10, n), dtype=pl.Int64),
            "v": pl.Series(rng.integers(0, 10**6, n), dtype=pl.Int64),
        }).with_columns(v=pl.when(pl.col("v") % 8 != 0).then("v")).write_parquet(
            data / f"{shard}.parquet", compression="zstd", compression_level=1)

def query(data: Path, windows: int, source: str) -> pl.LazyFrame:
    files = sorted(str(p) for p in data.glob("*.parquet"))
    if source == "pyarrow":
        import pyarrow.dataset as pads
        lf = pl.scan_pyarrow_dataset(pads.dataset(files, format="parquet").filter(pads.field("g") >= 0))
    else:
        lf = pl.scan_parquet(files).filter(pl.col("g") >= 0)
    lf = lf.with_columns(seg=(pl.col("k") < 4).cum_sum().over("g"))
    key = ["g", "seg"]
    exprs = {f"c{i}": (pl.col("v") + i).cum_sum().over(key) for i in range(windows)}
    return lf.with_columns(**exprs).select("g", "seg", *exprs)

def checksum(df: pl.DataFrame) -> str:
    agg = df.select([pl.col(c).sum().alias(c + "_s") for c in df.columns] + [pl.col(c).null_count().alias(c + "_n") for c in df.columns] + [pl.len()])
    return hashlib.md5(str(agg.row(0)).encode()).hexdigest()

if __name__ == "__main__":
    cmd = sys.argv[1]
    data = Path(sys.argv[2])
    if cmd == "generate":
        generate(data, int(sys.argv[3]))
    else:
        iters = int(sys.argv[3]); out = open(sys.argv[4], "a", buffering=1)
        engine = sys.argv[5] if len(sys.argv) > 5 else "streaming"
        source = sys.argv[6] if len(sys.argv) > 6 else "pyarrow"
        windows = int(os.environ.get("WINDOWS", 24))
        ref = checksum(query(data, windows, source).collect(engine="in-memory"))
        out.write(f"pid={os.getpid()} ref={ref} polars={pl.__version__}\n")
        for i in range(iters):
            t0 = time.time()
            out.write(f"pid={os.getpid()} iter={i} start\n")
            res = query(data, windows, source).collect(engine=engine)
            cs = checksum(res)
            out.write(f"pid={os.getpid()} iter={i} {'OK' if cs == ref else 'MISMATCH ' + cs} shape={res.shape} {time.time()-t0:.1f}s\n")
            del res
