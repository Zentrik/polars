"""Parquet round-trip fuzzer: random frames -> parquet (polars or pyarrow writer, random options)
-> scan with random projection/filter/slice in both engines -> compare with in-memory truth.
Usage: python fuzz_parquet.py SEED NITER LOGFILE
"""
import sys, os, random, io, datetime, warnings, tempfile
warnings.simplefilter("ignore")
import polars as pl, numpy as np
import pyarrow as pa, pyarrow.parquet as pq

seed = int(sys.argv[1]); niter = int(sys.argv[2]); logf = open(sys.argv[3], "a", buffering=1)
rng = random.Random(seed)
def log(m): logf.write(f"[seed={seed}] {m}\n")

def rand_str():
    n = rng.choice([0, 1, 5, 12, 13, 20, 64, 200])
    return "".join(rng.choice("abcXYZ é中😀") for _ in range(n))

def rand_col(name, n, depth=0):
    kinds = ["i8","i32","i64","u16","u64","f32","f64","str","bool","date","datetime","duration","time","cat","enum","dec","bin","null","i128"]
    if depth < 2: kinds += ["list","struct","array"]
    kind = rng.choice(kinds)
    nullp = rng.choice([0.0, 0.0, 0.05, 0.5, 0.95, 1.0])
    nul = lambda v: None if rng.random() < nullp else v
    if kind == "i8": return pl.Series(name, [nul(rng.randint(-128,127)) for _ in range(n)], pl.Int8)
    if kind == "i32": return pl.Series(name, [nul(rng.randint(-2**31,2**31-1)) for _ in range(n)], pl.Int32)
    if kind == "i64": return pl.Series(name, [nul(rng.choice([0,1,-1,2**63-1,-2**63,rng.randint(-10**6,10**6)])) for _ in range(n)], pl.Int64)
    if kind == "i128": return pl.Series(name, [nul(rng.choice([0,1,-1,2**100,-2**100,rng.randint(-10**6,10**6)])) for _ in range(n)], pl.Int128)
    if kind == "u16": return pl.Series(name, [nul(rng.randint(0,65535)) for _ in range(n)], pl.UInt16)
    if kind == "u64": return pl.Series(name, [nul(rng.choice([0,1,2**64-1,rng.randint(0,10**6)])) for _ in range(n)], pl.UInt64)
    if kind == "f32": return pl.Series(name, [nul(rng.choice([0.0,-0.0,1.5,float('nan'),float('inf'),rng.uniform(-1e6,1e6)])) for _ in range(n)], pl.Float32)
    if kind == "f64": return pl.Series(name, [nul(rng.choice([0.0,-0.0,1.5,float('nan'),float('inf'),rng.uniform(-1e300,1e300)])) for _ in range(n)], pl.Float64)
    if kind == "str":
        pool = [rand_str() for _ in range(rng.choice([1,2,5,50,1000]))]
        return pl.Series(name, [nul(rng.choice(pool)) for _ in range(n)], pl.String)
    if kind == "bin": return pl.Series(name, [nul(rand_str().encode()) for _ in range(n)], pl.Binary)
    if kind == "bool": return pl.Series(name, [nul(rng.random()<0.5) for _ in range(n)], pl.Boolean)
    if kind == "date": return pl.Series(name, [nul(datetime.date(1970,1,1)+datetime.timedelta(days=rng.randint(-100000,100000))) for _ in range(n)], pl.Date)
    if kind == "datetime": return pl.Series(name, [nul(datetime.datetime(1970,1,1)+datetime.timedelta(seconds=rng.randint(-10**10,10**10))) for _ in range(n)], pl.Datetime(rng.choice(["ms","us","ns"]), rng.choice([None, "UTC", "Europe/Amsterdam"])))
    if kind == "duration": return pl.Series(name, [nul(datetime.timedelta(seconds=rng.randint(-10**9,10**9))) for _ in range(n)], pl.Duration(rng.choice(["ms","us","ns"])))
    if kind == "time": return pl.Series(name, [nul(datetime.time(rng.randint(0,23),rng.randint(0,59),rng.randint(0,59))) for _ in range(n)], pl.Time)
    if kind == "cat": return pl.Series(name, [nul(rng.choice(["a","b","c","dddd","","é"])) for _ in range(n)], pl.Categorical)
    if kind == "enum": return pl.Series(name, [nul(rng.choice(["a","b","c"])) for _ in range(n)], pl.Enum(["a","b","c","d"]))
    if kind == "dec": return pl.Series(name, [nul(rng.randint(-10**12,10**12)) for _ in range(n)], pl.Int64).cast(pl.Decimal(rng.choice([18,38]), rng.choice([0,2,5])))
    if kind == "null": return pl.Series(name, [None]*n, pl.Null)
    if kind == "list":
        inner = rand_col("i", 0, depth+1).dtype
        vals = []
        for _ in range(n):
            m = rng.choice([0,0,1,2,5,30])
            vals.append(nul(rand_col("i", m, depth+1).cast(inner, strict=False).to_list()) if m or True else None)
        try: return pl.Series(name, vals, pl.List(inner))
        except Exception: return pl.Series(name, [nul([rng.randint(0,9) for _ in range(rng.randint(0,4))]) for _ in range(n)], pl.List(pl.Int64))
    if kind == "array":
        w = rng.choice([1,2,4]); inner = rand_col("i", 0, depth+1).dtype
        try: return pl.Series(name, [nul(rand_col("i", w, depth+1).cast(inner, strict=False).to_list()) for _ in range(n)], pl.Array(inner, w))
        except Exception: return pl.Series(name, [nul([rng.randint(0,9)]*w) for _ in range(n)], pl.Array(pl.Int64, w))
    if kind == "struct":
        fields = [rand_col(f"f{j}", n, depth+1) for j in range(rng.choice([1,2,3]))]
        s = pl.DataFrame(fields).to_struct(name)
        if nullp > 0:
            mask = pl.Series([rng.random() < nullp for _ in range(n)])
            s = pl.select(pl.when(mask).then(None).otherwise(s).alias(name)).to_series()
        return s

def rand_df():
    n = rng.choice([0, 1, 2, 7, 100, 1000, 5000, 20000])
    cols = []
    for i in range(rng.randint(1, 6)):
        try: cols.append(rand_col(f"c{i}", n))
        except Exception as ex: log(f"  col gen exc {type(ex).__name__}: {str(ex)[:100]}")
    if not cols: cols = [pl.Series("c0", range(n))]
    return pl.DataFrame(cols)

def write(df, path):
    if rng.random() < 0.6:
        opts = dict(compression=rng.choice(["zstd","lz4","snappy","uncompressed","gzip","brotli"]), row_group_size=rng.choice([None, 1, 7, 100, 1000, 4096]), data_page_size=rng.choice([None, 1, 100, 1024, 1<<16]), statistics=rng.choice([True, False, "full"]))
        log(f"  polars write {opts}")
        df.write_parquet(path, **opts)
        return "polars"
    else:
        t = df.to_arrow()
        opts = dict(use_dictionary=rng.choice([True, False]), data_page_version=rng.choice(["1.0", "2.0"]), version=rng.choice(["1.0", "2.4", "2.6"]), compression=rng.choice(["zstd","snappy","none","lz4","gzip"]), write_statistics=rng.choice([True, False]), data_page_size=rng.choice([None, 64, 1024, 1<<16]), row_group_size=rng.choice([None, 1, 13, 500, 5000]), use_byte_stream_split=rng.choice([True, False]), write_page_index=rng.choice([True, False]), dictionary_pagesize_limit=rng.choice([None, 128, 1<<20]))
        if opts["dictionary_pagesize_limit"] is None: del opts["dictionary_pagesize_limit"]
        if rng.random() < 0.3: opts["column_encoding"] = rng.choice(["PLAIN", "DELTA_BINARY_PACKED", "DELTA_BYTE_ARRAY", "DELTA_LENGTH_BYTE_ARRAY", "BYTE_STREAM_SPLIT", "RLE"]) ; opts["use_dictionary"] = False
        log(f"  pyarrow write {opts}")
        try:
            pq.write_table(t, path, **opts)
        except Exception as ex:
            log(f"  pyarrow write failed {type(ex).__name__}: {str(ex)[:100]}; fallback")
            pq.write_table(t, path)
        return "pyarrow"

def rand_pred(df):
    cands = [c for c, t in df.schema.items() if not t.is_nested() and t != pl.Null]
    if not cands or rng.random() < 0.2: return None
    c = rng.choice(cands); t = df.schema[c]; s = df[c].drop_nulls()
    if s.len() == 0: return pl.col(c).is_null()
    v = s[rng.randint(0, s.len()-1)]
    kind = rng.choice(["eq", "gt", "lt", "is_in", "null", "notnull", "and", "or", "ne"])
    try:
        if kind == "eq": return pl.col(c) == v
        if kind == "ne": return pl.col(c) != v
        if kind == "gt": return pl.col(c) > v
        if kind == "lt": return pl.col(c) < v
        if kind == "is_in": return pl.col(c).is_in(s.unique().head(rng.choice([1, 3, 50])).to_list())
        if kind == "null": return pl.col(c).is_null()
        if kind == "notnull": return pl.col(c).is_not_null()
        if kind == "and": return (pl.col(c) == v) & pl.col(cands[0]).is_not_null()
        if kind == "or": return (pl.col(c) == v) | (pl.col(cands[-1]).is_null())
    except Exception: return None

def norm(df):
    # canonical form for comparison
    return df.select([pl.col(c).cast(pl.String) if isinstance(t, pl.Categorical) else pl.col(c) for c, t in df.schema.items()])

tmpdir = tempfile.mkdtemp()
for it in range(niter):
    try:
        df = rand_df()
        path = os.path.join(tmpdir, f"f{it}.parquet")
        log(f"iter {it} schema={df.schema} h={df.height}")
        writer = write(df, path)
        truth = pl.read_parquet(path)
        if truth.height != df.height:
            log(f"  !!! HEIGHT MISMATCH after write/read: {truth.height} vs {df.height}")
        # random scans
        for k in range(4):
            lf_par = rng.choice(["auto", "columns", "row_groups", "prefiltered", "none"]); lf_lm = rng.random()<0.2; lf_us = rng.random()<0.8
            lf = pl.scan_parquet(path, parallel=lf_par, low_memory=lf_lm, use_statistics=lf_us, rechunk=rng.random()<0.3)
            proj = rng.sample(df.columns, rng.randint(1, len(df.columns))) if rng.random() < 0.7 else None
            pred = rand_pred(df)
            sl = rng.choice([None, (0, 5), (3, 100), (-10, 5), (df.height // 2, 3), (1, None)])
            desc = f"scan proj={proj} pred={pred} slice={sl}"
            ref = truth
            if pred is not None: ref = ref.filter(pred)
            if proj: ref = ref.select(proj)
            if sl: ref = ref.slice(sl[0], sl[1]) if sl[1] is not None else ref.slice(sl[0])
            q = lf
            if pred is not None: q = q.filter(pred)
            if proj: q = q.select(proj)
            if sl: q = q.slice(sl[0], sl[1]) if sl[1] is not None else q.slice(sl[0])
            for engine in ["in-memory", "streaming"]:
                log(f"  {desc} engine={engine}")
                try:
                    got = q.collect(engine=engine)
                except Exception as ex:
                    log(f"    exc {type(ex).__name__}: {str(ex)[:150]}"); continue
                try:
                    if not norm(got).equals(norm(ref)):
                        log(f"    !!! DATA MISMATCH ({writer} writer): got shape={got.shape} ref shape={ref.shape}")
                        import shutil, json
                        keep = os.path.splitext(sys.argv[3])[0] + f"_mismatch_{it}_{k}_{engine}.parquet"
                        shutil.copy(path, keep)
                        log(f"    saved {keep}; query: proj={proj!r} pred={pred!r} slice={sl!r} scan_kwargs={{'parallel': {lf_par!r}, 'low_memory': {lf_lm!r}, 'use_statistics': {lf_us!r}}}")
                        log(f"    got head: {got.head(5).to_dicts()}")
                        log(f"    ref head: {ref.head(5).to_dicts()}")
                except Exception as ex:
                    log(f"    compare exc {type(ex).__name__}: {str(ex)[:150]}")
        os.remove(path)
    except pl.exceptions.PanicException as ex:
        log(f"  PANIC: {str(ex)[:300]}")
    except Exception as ex:
        log(f"  exc {type(ex).__name__}: {str(ex)[:200]}")
log("done")
