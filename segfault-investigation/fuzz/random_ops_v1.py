"""Random-op fuzzer for polars. Logs each op before running so a crash can be attributed.
Usage: python fuzz1.py SEED NOPS LOGFILE
"""
import sys, random, os, traceback, datetime, itertools
import polars as pl
import numpy as np

seed = int(sys.argv[1]); nops = int(sys.argv[2]); logf = open(sys.argv[3], "a", buffering=1)
rng = random.Random(seed)
np.random.seed(seed % (2**32))

def log(msg):
    logf.write(f"[seed={seed}] {msg}\n"); logf.flush()

def rand_str():
    n = rng.choice([0, 1, 3, 11, 12, 13, 20, 40, 100])
    return "".join(rng.choice("abcdeXYZ_ é中\U0001F600") for _ in range(n))

def rand_series(name, n, depth=0):
    kind = rng.choice(["i8","i32","i64","u8","u64","f32","f64","str","bool","date","datetime","datetime_tz","duration","time","cat","enum","dec","list","struct","array","null","binary","i128"] if depth < 2 else ["i32","f64","str","bool"])
    nullp = rng.choice([0.0, 0.1, 0.5, 1.0])
    def nul(v):
        return None if rng.random() < nullp else v
    if kind == "i8": data=[nul(rng.randint(-128,127)) for _ in range(n)]; s=pl.Series(name,data,pl.Int8)
    elif kind == "i32": data=[nul(rng.randint(-2**31,2**31-1)) for _ in range(n)]; s=pl.Series(name,data,pl.Int32)
    elif kind == "i64": data=[nul(rng.choice([0,1,-1,2**63-1,-2**63,rng.randint(-1000,1000)])) for _ in range(n)]; s=pl.Series(name,data,pl.Int64)
    elif kind == "i128": data=[nul(rng.choice([0,1,-1,2**100,-2**100,rng.randint(-1000,1000)])) for _ in range(n)]; s=pl.Series(name,data,pl.Int128)
    elif kind == "u8": data=[nul(rng.randint(0,255)) for _ in range(n)]; s=pl.Series(name,data,pl.UInt8)
    elif kind == "u64": data=[nul(rng.choice([0,1,2**64-1,rng.randint(0,1000)])) for _ in range(n)]; s=pl.Series(name,data,pl.UInt64)
    elif kind == "f32": data=[nul(rng.choice([0.0,-0.0,1.5,float('nan'),float('inf'),-float('inf'),rng.uniform(-1e6,1e6)])) for _ in range(n)]; s=pl.Series(name,data,pl.Float32)
    elif kind == "f64": data=[nul(rng.choice([0.0,-0.0,1.5,float('nan'),float('inf'),-float('inf'),rng.uniform(-1e300,1e300)])) for _ in range(n)]; s=pl.Series(name,data,pl.Float64)
    elif kind == "str": data=[nul(rand_str()) for _ in range(n)]; s=pl.Series(name,data,pl.String)
    elif kind == "binary": data=[nul(rand_str().encode()) for _ in range(n)]; s=pl.Series(name,data,pl.Binary)
    elif kind == "bool": data=[nul(rng.random()<0.5) for _ in range(n)]; s=pl.Series(name,data,pl.Boolean)
    elif kind == "date": data=[nul(datetime.date(1970,1,1)+datetime.timedelta(days=rng.randint(-100000,100000))) for _ in range(n)]; s=pl.Series(name,data,pl.Date)
    elif kind == "datetime": data=[nul(datetime.datetime(1970,1,1)+datetime.timedelta(seconds=rng.randint(-10**10,10**10))) for _ in range(n)]; s=pl.Series(name,data,pl.Datetime(rng.choice(["ms","us","ns"])))
    elif kind == "datetime_tz": data=[nul(datetime.datetime(1970,1,1)+datetime.timedelta(seconds=rng.randint(-10**9,10**9))) for _ in range(n)]; s=pl.Series(name,data,pl.Datetime("us")).dt.replace_time_zone(rng.choice(["UTC","Europe/Amsterdam","America/New_York"]))
    elif kind == "duration": data=[nul(datetime.timedelta(seconds=rng.randint(-10**9,10**9))) for _ in range(n)]; s=pl.Series(name,data,pl.Duration(rng.choice(["ms","us","ns"])))
    elif kind == "time": data=[nul(datetime.time(rng.randint(0,23),rng.randint(0,59),rng.randint(0,59))) for _ in range(n)]; s=pl.Series(name,data,pl.Time)
    elif kind == "cat": data=[nul(rng.choice(["a","b","c","dddd","","é"])) for _ in range(n)]; s=pl.Series(name,data,pl.Categorical(ordering=rng.choice(["physical","lexical"])))
    elif kind == "enum": data=[nul(rng.choice(["a","b","c"])) for _ in range(n)]; s=pl.Series(name,data,pl.Enum(["a","b","c","d"]))
    elif kind == "dec": data=[nul(rng.randint(-10**12,10**12)) for _ in range(n)]; s=pl.Series(name,data,pl.Int64).cast(pl.Decimal(rng.choice([18,38]),rng.choice([0,2,5])))
    elif kind == "list":
        inner = rand_series("i", 0, depth+1).dtype
        data=[nul([v for v in rand_series("i", rng.choice([0,1,2,5,20]), depth+1).to_list()]) for _ in range(n)]
        s=pl.Series(name,data,pl.List(inner))
    elif kind == "array":
        w = rng.choice([1,2,3])
        inner = rand_series("i", 0, depth+1).dtype
        data=[nul(rand_series("i", w, depth+1).to_list()) for _ in range(n)]
        s=pl.Series(name,data,pl.Array(inner, w))
    elif kind == "struct":
        nf = rng.choice([1,2,3])
        fields=[rand_series(f"f{j}", n, depth+1) for j in range(nf)]
        s=pl.struct(fields) if False else pl.DataFrame(fields).to_struct(name)
        if nullp>0:
            mask = pl.Series([rng.random()<nullp for _ in range(n)])
            s = pl.select(pl.when(mask).then(None).otherwise(s).alias(name)).to_series()
    elif kind == "null": s=pl.Series(name,[None]*n,pl.Null)
    # random chunking
    if rng.random()<0.4 and n>1:
        k = rng.randint(1,min(n,5)); cuts=sorted(rng.sample(range(1,n),k-1)) if k>1 else []
        parts=[]; prev=0
        for c in cuts+[n]:
            parts.append(s.slice(prev,c-prev)); prev=c
        s = pl.concat(parts, rechunk=False)
    # random slice (creates offset)
    if rng.random()<0.3 and n>2:
        a=rng.randint(0,n//2); s = pl.concat([pl.Series(name,[None]*a,s.dtype) if rng.random()<0.5 else s.head(a), s], rechunk=False).slice(a, n)
    return s

def rand_df(n=None, ncols=None):
    n = rng.choice([0,1,2,3,7,17,64,65,129,1000,1025]) if n is None else n
    ncols = rng.randint(1,5) if ncols is None else ncols
    cols=[rand_series(f"c{i}", n) for i in range(ncols)]
    return pl.DataFrame(cols)

def collect(lf):
    eng = rng.choice(["in-memory","streaming"])
    return lf.collect(engine=eng)

def numeric_cols(df): return [c for c,t in df.schema.items() if t.is_numeric()]
def sortable_cols(df): return [c for c,t in df.schema.items() if not t.is_nested() and t != pl.Null and t != pl.Object]

def op(df):
    choice = rng.choice(OPS)
    log(f"op={choice.__name__} schema={df.schema} h={df.height} chunks={df.n_chunks('all')}")
    return choice(df)

def op_groupby(df):
    keys = rng.sample(df.columns, rng.randint(1,len(df.columns)))
    aggs = []
    for c in df.columns:
        if c in keys: continue
        aggs += [pl.col(c).first().alias(c+"_first"), pl.col(c).last().alias(c+"_last"), pl.col(c).count().alias(c+"_n"), pl.col(c).n_unique().alias(c+"_nu"), pl.col(c).alias(c+"_list")]
        if df.schema[c].is_numeric(): aggs += [pl.col(c).sum().alias(c+"_s"), pl.col(c).mean().alias(c+"_m"), pl.col(c).min().alias(c+"_min"), pl.col(c).max().alias(c+"_max"), pl.col(c).std().alias(c+"_std"), pl.col(c).quantile(0.3).alias(c+"_q")]
        elif df.schema[c] in (pl.String,): aggs += [pl.col(c).min().alias(c+"_min"), pl.col(c).max().alias(c+"_max"), pl.col(c).str.len_chars().sum().alias(c+"_len")]
    if not aggs: aggs=[pl.len()]
    maintain = rng.random()<0.5
    return collect(df.lazy().group_by(keys, maintain_order=maintain).agg(aggs))

def op_sort(df):
    cols = sortable_cols(df)
    if not cols: return df
    keys = rng.sample(cols, rng.randint(1,len(cols)))
    return collect(df.lazy().sort(keys, descending=[rng.random()<0.5 for _ in keys], nulls_last=[rng.random()<0.5 for _ in keys], multithreaded=rng.random()<0.5))

def op_unique(df):
    cols = [c for c,t in df.schema.items() if t != pl.Object]
    if not cols: return df
    subset = rng.sample(cols, rng.randint(1,len(cols)))
    return collect(df.lazy().unique(subset=subset, keep=rng.choice(["any","first","last","none"]), maintain_order=rng.random()<0.5))

def op_join(df):
    other = rand_df(n=rng.choice([0,1,5,100]), ncols=1)
    left = rng.choice(df.columns)
    lt = df.schema[left]
    if lt.is_nested() or lt == pl.Null: return df
    o = other.select(pl.col("c0").cast(lt, strict=False).alias("k"), pl.lit(1).alias("v")) if rng.random()<0.5 else df.select(pl.col(left).alias("k"), pl.lit(1).alias("v")).head(rng.randint(0,10))
    how = rng.choice(["inner","left","full","semi","anti","right"])
    return collect(df.lazy().join(o.lazy(), left_on=left, right_on="k", how=how, nulls_equal=rng.random()<0.5, coalesce=rng.choice([None,True,False]) if how in ("inner","left","full","right") else None))

def op_explode(df):
    cols = [c for c,t in df.schema.items() if t.is_nested() and not isinstance(t, pl.Struct)]
    if not cols: return df
    return collect(df.lazy().explode(rng.choice(cols)))

def op_window(df):
    part = rng.choice(df.columns)
    if df.schema[part].is_nested(): return df
    exprs=[]
    for c in df.columns:
        t = df.schema[c]
        if t.is_numeric(): exprs += [pl.col(c).sum().over(part).alias(c+"_ws"), pl.col(c).rank().over(part).alias(c+"_rk"), pl.col(c).cum_sum().over(part, mapping_strategy=rng.choice(["group_to_rows","explode","join"])).alias(c+"_cs")]
        exprs += [pl.col(c).first().over(part).alias(c+"_wf"), pl.col(c).shift(rng.randint(-3,3)).over(part).alias(c+"_sh")]
    return collect(df.lazy().select(exprs))

def op_string(df):
    cols = [c for c,t in df.schema.items() if t == pl.String]
    if not cols: return df
    c = rng.choice(cols)
    e = pl.col(c)
    exprs = [e.str.len_bytes(), e.str.len_chars(), e.str.slice(rng.randint(-5,5), rng.choice([None,0,1,3])), e.str.to_uppercase(), e.str.reverse(), e.str.split(rng.choice(["a","","é"])), e.str.replace_all(rng.choice(["a","",".*"]), "ZZ"), e.str.contains(rng.choice(["a","^$","(a"]), strict=False), e.str.extract_all("a+"), e.str.head(rng.randint(-5,5)), e.str.tail(rng.randint(-5,5)), e.str.strip_chars(), e.str.pad_start(rng.randint(0,30)), e.str.zfill(rng.randint(0,30)), e.str.to_integer(strict=False), e.str.json_decode(infer_schema_length=10) if False else e.str.count_matches("a"), e.str.find("a"), e.str.strip_prefix("a"), e.str.encode("hex"), e.str.escape_regex(), e.str.normalize("NFKC"), e.str.replace_many(["a","b"],["xx","y"]), e.str.contains_any(["a","b"]), e.str.extract_groups("(a)(b)?"), e.str.splitn(" ", 3), e.str.split_exact("a", 2), e.str.join("-"), e.str.to_titlecase()]
    k = rng.randint(1,len(exprs))
    return collect(df.lazy().select([x.alias(f"s{i}") for i,x in enumerate(rng.sample(exprs,k))]))

def op_list(df):
    cols = [c for c,t in df.schema.items() if isinstance(t, pl.List)]
    if not cols: return df
    c = rng.choice(cols); e = pl.col(c)
    inner = df.schema[c].inner
    exprs = [e.list.len(), e.list.first(), e.list.last(), e.list.get(rng.randint(-3,3), null_on_oob=True), e.list.slice(rng.randint(-3,3), rng.choice([None,0,1,2])), e.list.reverse(), e.list.unique(), e.list.head(2), e.list.tail(2), e.list.gather([0,-1], null_on_oob=True), e.list.contains(pl.lit(None)), e.list.explode(), e.list.sample(fraction=0.5, with_replacement=True, seed=1), e.list.shift(1), e.list.drop_nulls(), e.list.concat([e]), e.list.eval(pl.element().first()), e.list.eval(pl.element().rank()), e.list.n_unique(), e.list.set_union(e), e.list.set_difference(e.list.reverse()), e.list.to_struct(), e.list.diff() if inner.is_numeric() else e.list.len(), e.list.sum() if inner.is_numeric() else e.list.min(), e.list.mean() if inner.is_numeric() else e.list.max(), e.list.sort(descending=True, nulls_last=True), e.list.arg_max() if inner.is_numeric() else e.list.arg_min(), e.list.join(",") if inner==pl.String else e.list.len(), e.list.gather_every(2, rng.randint(0,2)), e.list.to_array(2) if False else e.list.count_matches(pl.lit(None)), e.list.filter(pl.element().is_not_null()), e.list.agg(pl.element().first()) if hasattr(e.list,'agg') else e.list.first()]
    k = rng.randint(1,len(exprs))
    return collect(df.lazy().select([x.alias(f"l{i}") for i,x in enumerate(rng.sample(exprs,k))]))

def op_array(df):
    cols = [c for c,t in df.schema.items() if isinstance(t, pl.Array)]
    if not cols: return df
    c = rng.choice(cols); e = pl.col(c); inner=df.schema[c].inner
    exprs = [e.arr.len() if hasattr(e.arr,'len') else e.arr.first(), e.arr.first(), e.arr.last(), e.arr.get(rng.randint(-3,3), null_on_oob=True), e.arr.reverse(), e.arr.unique(), e.arr.explode(), e.arr.to_list(), e.arr.contains(pl.lit(None)), e.arr.n_unique(), e.arr.shift(1), e.arr.to_struct(), e.arr.sort(), e.arr.any() if inner==pl.Boolean else e.arr.max(), e.arr.sum() if inner.is_numeric() else e.arr.min(), e.arr.eval(pl.element().rank()) if hasattr(e.arr,'eval') else e.arr.first(), e.arr.agg(pl.element().first()) if hasattr(e.arr,'agg') else e.arr.first(), e.arr.slice(0,1) if hasattr(e.arr,'slice') else e.arr.first()]
    k = rng.randint(1,len(exprs))
    return collect(df.lazy().select([x.alias(f"a{i}") for i,x in enumerate(rng.sample(exprs,k))]))

def op_struct(df):
    cols = [c for c,t in df.schema.items() if isinstance(t, pl.Struct)]
    if not cols: return df
    c = rng.choice(cols); e = pl.col(c)
    fields = df.schema[c].fields
    exprs = [e.struct.unnest(), e.struct.field(fields[0].name), e.struct.rename_fields(["z"+f.name for f in fields]), e.struct.json_encode(), e.struct.with_fields(pl.lit(1).alias("new")), e.struct.field("*") if rng.random()<0.5 else e.struct.field(fields[-1].name), e.is_null(), e.rank() if False else e.is_not_null(), e.hash(), e.n_unique(), e.first(), e.reverse(), e.shift(1), e.gather([0]) if df.height>0 else e]
    k = rng.randint(1,len(exprs))
    return collect(df.lazy().select([x.alias(f"st{i}") if not isinstance(x, pl.Expr) or True else x for i,x in enumerate(rng.sample(exprs,k))]))

def op_cast(df):
    exprs=[]
    targets=[pl.Int8,pl.Int32,pl.Int64,pl.UInt8,pl.UInt64,pl.Float32,pl.Float64,pl.String,pl.Boolean,pl.Categorical,pl.Date,pl.Datetime("ms"),pl.Duration("ns"),pl.Time,pl.Decimal(38,2),pl.Int128,pl.Binary,pl.List(pl.Int64),pl.List(pl.String),pl.Enum(["a","b"])]
    for c in df.columns:
        exprs.append(pl.col(c).cast(rng.choice(targets), strict=False, wrap_numerical=rng.random()<0.5).alias(c))
    return collect(df.lazy().select(exprs))

def op_gather(df):
    n=df.height
    idx = [rng.randint(-n, n-1) for _ in range(rng.randint(0,20))] if n>0 else []
    exprs=[pl.col(c).gather(idx).alias(c) for c in df.columns]
    if rng.random()<0.5:
        return collect(df.lazy().select(exprs))
    else:
        return df[idx] if n>0 else df

def op_filter(df):
    c = rng.choice(df.columns); t=df.schema[c]
    if t.is_numeric(): pred = pl.col(c) > rng.randint(-100,100)
    elif t == pl.String: pred = pl.col(c).str.len_bytes() > rng.randint(0,5)
    elif t == pl.Boolean: pred = pl.col(c)
    else: pred = pl.col(c).is_not_null() if rng.random()<0.5 else pl.col(c).is_null()
    return collect(df.lazy().filter(pred))

def op_slice(df):
    n=df.height
    return df.slice(rng.randint(-n-2,n+2), rng.choice([None,0,1,2,n,n+5]))

def op_concat(df):
    other = rand_df(n=rng.choice([0,1,5]), ncols=len(df.columns))
    how = rng.choice(["vertical_relaxed","diagonal_relaxed","horizontal"])
    try:
        return pl.concat([df, other.rename(dict(zip(other.columns, df.columns))) if how!="horizontal" else other.rename({c:c+"_o" for c in other.columns})], how=how, rechunk=rng.random()<0.5)
    except pl.exceptions.SchemaError: return df
    except pl.exceptions.ShapeError: return df

def op_extend(df):
    d2 = df.clone()
    d2.vstack(df, in_place=True)
    if rng.random()<0.5: d2.rechunk()
    return d2

def op_pivot(df):
    cols = [c for c,t in df.schema.items() if not t.is_nested() and t!=pl.Null]
    if len(cols)<2: return df
    on, idx = rng.sample(cols, 2)
    vals = [c for c in df.columns if c not in (on, idx)]
    if not vals: vals=[idx]
    try:
        return df.pivot(on=on, index=idx, values=vals[0], aggregate_function=rng.choice(["first","len","min","max","sum","mean"]))
    except pl.exceptions.ComputeError: return df
    except pl.exceptions.InvalidOperationError: return df

def op_unpivot(df):
    return df.unpivot(on=rng.sample(df.columns, rng.randint(1,len(df.columns))), index=None)

def op_roundtrip(df):
    import io
    fmt = rng.choice(["parquet","ipc","json","csv","ndjson","ipc_stream","parquet_stream"])
    buf = io.BytesIO()
    try:
        if fmt=="parquet": df.write_parquet(buf, compression=rng.choice(["zstd","lz4","snappy","uncompressed"]), row_group_size=rng.choice([None,1,3,64]), data_page_size=rng.choice([None,1,1024])); buf.seek(0); out = pl.read_parquet(buf) if rng.random()<0.5 else pl.scan_parquet(buf).collect(engine="streaming")
        elif fmt=="parquet_stream":
            df.write_parquet(buf, compression="zstd", row_group_size=rng.choice([None,1,3]), use_pyarrow=False); buf.seek(0)
            out = pl.scan_parquet(buf).filter(pl.col(df.columns[0]).is_not_null()).select(df.columns[::-1]).collect(engine="streaming")
        elif fmt=="ipc": df.write_ipc(buf, compression=rng.choice([None,"zstd","lz4"])); buf.seek(0); out = pl.read_ipc(buf)
        elif fmt=="ipc_stream": df.write_ipc_stream(buf); buf.seek(0); out = pl.read_ipc_stream(buf)
        elif fmt=="json": df.write_json(buf); buf.seek(0); out = pl.read_json(buf)
        elif fmt=="csv": df.write_csv(buf); buf.seek(0); out = pl.read_csv(buf, try_parse_dates=True)
        elif fmt=="ndjson": df.write_ndjson(buf); buf.seek(0); out = pl.read_ndjson(buf)
    except (pl.exceptions.PolarsError, TypeError, ValueError, NotImplementedError) as ex:
        return df
    return out

def op_arith(df):
    nc = numeric_cols(df)
    if not nc: return df
    a = pl.col(rng.choice(nc)); b = pl.col(rng.choice(nc))
    exprs=[a+b, a-b, a*b, a/b, a//b, a%b, a**2, -a, a.abs(), a.log(), a.sqrt(), a.cum_sum(), a.cum_max(), a.diff(), a.pct_change(), a.rolling_sum(rng.randint(1,5)), a.rolling_mean(rng.randint(1,5), min_samples=1), a.rolling_quantile(0.5, window_size=3), a.ewm_mean(alpha=0.5), a.fill_null(strategy=rng.choice(["forward","backward","min","max","mean","zero","one"])), a.interpolate(), a.clip(0,10), a.round(2), a.rank(method=rng.choice(["average","min","max","dense","ordinal","random"])), a.search_sorted(b), a.hist(bin_count=3), a.qcut([0.5]), a.cut([0]), a.mode(), a.value_counts(), a.unique_counts(), a.arg_sort(), a.arg_unique(), a.peak_max(), a.top_k(3), a.bottom_k(3), a.shift(-2, fill_value=1), a.is_in(b), a.is_between(b, b), (a==b).any(), a.approx_n_unique(), a.skew(), a.kurtosis(), a.entropy(), a.product(), a.sum(), a.mean(), a.median(), a.std(), a.var(), a.rolling_std(3), a.rolling_min(3), a.rolling_max(3), a.rolling_median(3), a.rolling_map(lambda s: s.sum(), 3), a.replace_strict({0:1}, default=None), a.replace({0:1}), a.reinterpret(signed=False) if df.schema[nc[0]] in (pl.Int64,pl.UInt64) else a, a.cum_count(), a.n_unique(), a.null_count(), a.reverse(), a.gather_every(2,1), a.sample(fraction=1.5, with_replacement=True, seed=0), a.explode(), a.implode(), a.repeat_by(2), a.rle(), a.rle_id(), a.min_by(b) if hasattr(a,'min_by') else a.min(), a.max_by(b) if hasattr(a,'max_by') else a.max(), a.arg_true() if False else a.sign(), a.bitwise_and(b) if df.schema[nc[0]].is_integer() and df.schema[nc[-1]]==df.schema[nc[0]] else a, a.cos(), a.exp(), a.sinh(), a.degrees(), a.is_nan() if df.schema[nc[0]].is_float() else a.is_finite() if df.schema[nc[0]].is_float() else a]
    k = rng.randint(1,min(8,len(exprs)))
    return collect(df.lazy().select([x.alias(f"ar{i}") for i,x in enumerate(rng.sample(exprs,k))]))

def op_temporal(df):
    cols=[c for c,t in df.schema.items() if t.is_temporal()]
    if not cols: return df
    c=rng.choice(cols); e=pl.col(c); t=df.schema[c]
    if t == pl.Time: exprs=[e.dt.hour(), e.dt.minute(), e.dt.second(), e.dt.nanosecond(), e.dt.to_string("%H:%M"), e.dt.truncate("1h") if False else e.dt.hour(), e.max(), e.min(), e.mean(), e.median(), e.cast(pl.Int64)]
    elif isinstance(t, pl.Duration): exprs=[e.dt.total_days(), e.dt.total_nanoseconds(), e.dt.total_seconds(), e.abs(), e.sum(), e.mean(), e.median(), e.std(), e.cum_sum(), e.dt.to_string(), e.cast(pl.Int64), e.dt.cast_time_unit("ms"), e.diff(), e.fill_null(strategy="mean"), e.rolling_mean(2), e.quantile(0.5), e.max(), e.min(), e.mode(), e.rank(), e/2, e*2, e//2]
    else: exprs=[e.dt.year(), e.dt.month(), e.dt.day(), e.dt.weekday(), e.dt.iso_year(), e.dt.week(), e.dt.ordinal_day(), e.dt.truncate(rng.choice(["1d","1mo","1y","1h","15m","1w","1q","1ns","0ns"])), e.dt.round(rng.choice(["1d","1mo","1h","1w"])), e.dt.offset_by(rng.choice(["1d","1mo","-1y","3h","1q","1y6mo","0d","1wk"] if hasattr(e,'dt') else ["1d"])), e.dt.to_string(rng.choice(["%Y-%m-%d","%c","%s","%f","%j","%G-%V","iso","%+"] )), e.dt.epoch(rng.choice(["ns","us","ms","s","d"])), e.dt.timestamp("ns"), e.dt.month_start(), e.dt.month_end(), e.dt.quarter(), e.dt.is_leap_year(), e.dt.replace(year=2000) if hasattr(e.dt,'replace') else e.dt.year(), e.dt.add_business_days(rng.randint(-10,10)) if hasattr(e.dt,'add_business_days') else e.dt.year(), e.dt.total_days() if False else e.dt.strftime("%A"), e.dt.date() if t!=pl.Date else e.dt.year(), e.dt.time() if t!=pl.Date else e.dt.year(), e.dt.replace_time_zone(rng.choice(["UTC",None,"Asia/Kolkata"])) if t!=pl.Date else e, e.dt.convert_time_zone("Asia/Kathmandu") if (t!=pl.Date and t.time_zone) else e, e.dt.dst_offset() if t!=pl.Date else e, e.dt.base_utc_offset() if t!=pl.Date else e, e.max(), e.min(), e.mean(), e.median(), e.diff(), e.cast(pl.Int64), e.dt.cast_time_unit("ms") if t!=pl.Date else e, e.rolling_max(2), e.sort(), e.rank(), e.dt.century() if hasattr(e.dt,'century') else e, e.dt.days_in_month() if hasattr(e.dt,'days_in_month') else e, e.dt.is_business_day() if hasattr(e.dt,'is_business_day') else e]
    k = rng.randint(1,min(6,len(exprs)))
    try:
        return collect(df.lazy().select([x.alias(f"t{i}") for i,x in enumerate(rng.sample(exprs,k))]))
    except pl.exceptions.PolarsError: return df

def op_rolling_groupby(df):
    cols=[c for c,t in df.schema.items() if t in (pl.Date, pl.Int64, pl.Int32) or isinstance(t, pl.Datetime)]
    if not cols: return df
    c = rng.choice(cols)
    aggs=[pl.col(x).first().alias(x+"_f") for x in df.columns if x!=c] + [pl.len()]
    nc = numeric_cols(df)
    aggs += [pl.col(x).sum().alias(x+"_s") for x in nc if x!=c]
    period = rng.choice(["2i","1i","10i"]) if df.schema[c].is_integer() else rng.choice(["2d","1mo","3h","1w","1y"])
    try:
        d = df.sort(c)
        if rng.random()<0.5:
            return collect(d.lazy().rolling(index_column=c, period=period, closed=rng.choice(["left","right","both","none"])).agg(aggs))
        else:
            return collect(d.lazy().group_by_dynamic(index_column=c, every=period, period=rng.choice([None, period]), closed=rng.choice(["left","right","both","none"]), label=rng.choice(["left","right","datapoint"]), include_boundaries=rng.random()<0.5).agg(aggs))
    except pl.exceptions.PolarsError: return df

def op_asof(df):
    cols=[c for c,t in df.schema.items() if t.is_numeric() or t.is_temporal()]
    cols=[c for c in cols if df.schema[c] not in (pl.Time,) and not isinstance(df.schema[c], pl.Decimal)]
    if not cols: return df
    c=rng.choice(cols)
    try:
        d=df.drop_nulls(c).sort(c)
        other = d.select(pl.col(c).alias("k"), pl.lit(2).alias("v2")).head(rng.randint(0,5)).sort("k")
        return collect(d.lazy().join_asof(other.lazy(), left_on=c, right_on="k", strategy=rng.choice(["backward","forward","nearest"]), tolerance=rng.choice([None, 1, "1d", 5])))
    except pl.exceptions.PolarsError: return df
    except TypeError: return df

def op_when(df):
    c = rng.choice(df.columns)
    exprs=[pl.when(pl.col(c).is_null()).then(pl.col(c)).otherwise(pl.col(c).shift(1)).alias("w1"), pl.when(pl.col(c).is_not_null()).then(pl.lit(None)).otherwise(pl.col(c)).alias("w2"), pl.coalesce(pl.col(c), pl.col(c).reverse()).alias("w3"), pl.col(c).fill_null(pl.col(c).first()).alias("w4"), pl.col(c).zip_with(pl.col(c).is_null(), pl.col(c).shift(-1)).alias("w5") if hasattr(pl.Expr,'zip_with') else pl.col(c).alias("w5"), pl.when(pl.col(c).is_null()).then(pl.lit(1)).alias("w6"), pl.col(c).eq_missing(pl.col(c).shift(1)).alias("w7"), pl.col(c).ne_missing(pl.col(c).reverse()).alias("w8"), pl.col(c).is_first_distinct().alias("w9"), pl.col(c).is_last_distinct().alias("w10"), pl.col(c).is_duplicated().alias("w11"), pl.col(c).is_unique().alias("w12"), pl.col(c).unique(maintain_order=True).alias("w13"), pl.col(c).n_unique().alias("w14"), pl.col(c).hash().alias("w15"), pl.col(c).arg_unique().alias("w16"), pl.col(c).drop_nulls().alias("w17"), pl.col(c).null_count().alias("w18"), pl.col(c).slice(rng.randint(-5,5), rng.randint(0,5)).alias("w19"), pl.col(c).head(3).alias("w20"), pl.col(c).tail(3).alias("w21"), pl.col(c).first().alias("w22"), pl.col(c).last().alias("w23"), pl.col(c).shift(rng.randint(-5,5)).alias("w24"), pl.col(c).extend_constant(None, 3).alias("w25"), pl.col(c).reverse().alias("w26"), pl.col(c).sort(nulls_last=True).alias("w27"), pl.col(c).arg_sort(descending=True).alias("w28"), pl.col(c).filter(pl.col(c).is_not_null()).alias("w29"), pl.col(c).gather_every(3).alias("w30"), pl.col(c).explode().alias("w31") if df.schema[c].is_nested() and not isinstance(df.schema[c], pl.Struct) else pl.col(c).implode().alias("w31"), pl.col(c).to_physical().alias("w32"), pl.col(c).cast(pl.String, strict=False).alias("w33") if not df.schema[c].is_nested() else pl.col(c).is_null().alias("w33"), pl.col(c).count().alias("w34"), pl.col(c).len().alias("w35"), pl.col(c).implode().alias("w36"), pl.col(c).min().alias("w37") if not df.schema[c].is_nested() or isinstance(df.schema[c], (pl.List, pl.Array)) else pl.col(c).is_null().alias("w37"), pl.col(c).max().alias("w38") if not df.schema[c].is_nested() or isinstance(df.schema[c], (pl.List, pl.Array)) else pl.col(c).is_null().alias("w38"), pl.col(c).sort_by(pl.col(c).hash()).alias("w39"), pl.col(c).arg_max().alias("w40") if not df.schema[c].is_nested() else pl.col(c).alias("w40"), pl.col(c).arg_min().alias("w41") if not df.schema[c].is_nested() else pl.col(c).alias("w41"), pl.col(c).rank().alias("w42") if not df.schema[c].is_nested() and df.schema[c]!=pl.Null else pl.col(c).alias("w42"), pl.col(c).mode().alias("w43") if not df.schema[c].is_nested() and df.schema[c] != pl.Null else pl.col(c).alias("w43"), pl.col(c).value_counts().alias("w44") if not isinstance(df.schema[c], pl.Struct) else pl.col(c).alias("w44"), pl.col(c).is_in(pl.col(c).reverse().implode()).alias("w45") if not isinstance(df.schema[c], pl.Struct) else pl.col(c).alias("w45"), pl.col(c).repeat_by(2).alias("w46") if not df.schema[c].is_nested() else pl.col(c).alias("w46"), pl.col(c).bottom_k(2).alias("w47") if not df.schema[c].is_nested() and df.schema[c]!=pl.Null else pl.col(c).alias("w47"), pl.col(c).top_k_by(pl.col(c).hash(), 2).alias("w48") if hasattr(pl.Expr, 'top_k_by') else pl.col(c).alias("w48")]
    k = rng.randint(1,min(8,len(exprs)))
    try:
        return collect(df.lazy().select(rng.sample(exprs,k)))
    except pl.exceptions.PolarsError: return df

def op_misc(df):
    choice = rng.choice(["transpose","to_numpy","to_arrow","from_arrow","describe","hash_rows","iter_rows","to_dicts","partition_by","with_row_index","null_count","estimated_size","equals","rechunk","shrink","to_pandas","from_pandas","frame_equal","sample","fold","unnest","serialize","clear","glimpse","str","explode_all","to_dummies","fill_nan","interpolate","product","sum","mean","min","max","std","var","median","quantile","n_unique","is_duplicated","is_unique","hstack","insert","replace_column","to_series","get_column","row","item","apply","map_rows","corr","cov","top_k"])
    try:
        if choice=="transpose": return df.transpose(include_header=True)
        if choice=="to_numpy": df.to_numpy(); return df
        if choice=="to_arrow": return pl.from_arrow(df.to_arrow())
        if choice=="from_arrow": return pl.from_arrow(df.to_arrow().combine_chunks() if rng.random()<0.5 else df.to_arrow())
        if choice=="describe": df.describe(); return df
        if choice=="hash_rows": df.hash_rows(); return df
        if choice=="iter_rows": list(df.iter_rows()); return df
        if choice=="to_dicts": return pl.DataFrame(df.to_dicts(), schema=df.schema) if df.height else df
        if choice=="partition_by": c=rng.choice(df.columns); return pl.concat(df.partition_by(c)) if not df.schema[c].is_nested() and df.height else df
        if choice=="with_row_index": return df.with_row_index(offset=rng.randint(0,5))
        if choice=="null_count": df.null_count(); return df
        if choice=="estimated_size": df.estimated_size(); return df
        if choice=="equals": df.equals(df.clone()); return df
        if choice=="rechunk": return df.rechunk()
        if choice=="shrink": return df.select(pl.all().shrink_dtype())
        if choice=="to_pandas": df.to_pandas(); return df
        if choice=="from_pandas": return pl.from_pandas(df.to_pandas())
        if choice=="sample": return df.sample(fraction=rng.choice([0.5,2.0]), with_replacement=True, seed=rng.randint(0,100))
        if choice=="fold": return df.select(pl.fold(pl.lit(0), lambda acc,x: acc + x.is_null().cast(pl.Int64), pl.all()).alias("f"))
        if choice=="unnest": cols=[c for c,t in df.schema.items() if isinstance(t, pl.Struct)]; return df.unnest(cols[0]) if cols else df
        if choice=="serialize": return pl.DataFrame.deserialize(__import__("io").BytesIO(df.serialize()))
        if choice=="clear": return df.clear(rng.randint(0,3))
        if choice=="glimpse": df.glimpse(return_as_string=True); return df
        if choice=="str": str(df); repr(df); return df
        if choice=="explode_all": cols=[c for c,t in df.schema.items() if isinstance(t, (pl.List, pl.Array))]; return df.explode(cols) if cols else df
        if choice=="to_dummies": cols=[c for c,t in df.schema.items() if not t.is_nested()]; return df.to_dummies(cols[:2]) if cols else df
        if choice=="fill_nan": return df.fill_nan(0)
        if choice=="interpolate": return df.interpolate()
        if choice in ("product","sum","mean","min","max","std","var","median"): return getattr(df, choice)()
        if choice=="quantile": return df.quantile(0.4, interpolation=rng.choice(["nearest","higher","lower","midpoint","linear"]))
        if choice=="n_unique": df.n_unique(); return df
        if choice=="is_duplicated": df.is_duplicated(); return df
        if choice=="is_unique": df.is_unique(); return df
        if choice=="hstack": return df.hstack(df.rename({c:c+"_x" for c in df.columns}))
        if choice=="insert": d=df.clone(); d.insert_column(rng.randint(0,len(df.columns)), pl.Series("ins", range(df.height))); return d
        if choice=="replace_column": d=df.clone(); d.replace_column(0, pl.Series(df.columns[0], range(df.height))); return d
        if choice=="to_series": df.to_series(rng.randint(-len(df.columns), len(df.columns)-1)); return df
        if choice=="get_column": df.get_column(rng.choice(df.columns)); return df
        if choice=="row": df.row(rng.randint(-df.height, df.height-1)) if df.height else None; return df
        if choice=="item": df.item() if df.shape==(1,1) else None; return df
        if choice=="apply": return df.select(pl.col(df.columns[0]).map_elements(lambda x: x, return_dtype=df.schema[df.columns[0]]))
        if choice=="map_rows": return df.map_rows(lambda r: (len(r),)) if df.height else df
        if choice=="corr": nc=numeric_cols(df); return df.select(nc).corr() if len(nc)>=2 else df
        if choice=="cov": nc=numeric_cols(df); return df.select(pl.cov(nc[0], nc[1])) if len(nc)>=2 else df
        if choice=="top_k": cols=sortable_cols(df); return df.top_k(3, by=cols[0]) if cols else df
    except (pl.exceptions.PolarsError, TypeError, ValueError, NotImplementedError, IndexError, ZeroDivisionError, OverflowError, pl.exceptions.PanicException) as ex:
        log(f"  handled exc {type(ex).__name__}: {str(ex)[:200]}")
        return df
    return df

def op_series_ops(df):
    s = df.to_series(rng.randint(0,len(df.columns)-1))
    try:
        ops = [lambda: s.to_list(), lambda: s.to_numpy(), lambda: s.to_arrow(), lambda: pl.from_arrow(s.to_arrow()), lambda: s.to_frame().to_arrow(), lambda: s.hash(), lambda: s.chunk_lengths(), lambda: s.rechunk(), lambda: s.slice(1, 2).rechunk(), lambda: s.append(s.slice(0,1)), lambda: s.extend(s), lambda: s.new_from_index(0, 5) if len(s) else s, lambda: s.set_at_idx if False else s.scatter([0], [None]) if len(s) else s, lambda: s.gather(np.array([0, len(s)-1], dtype=np.uint32)) if len(s) else s, lambda: s.shuffle(seed=1), lambda: s.equals(s), lambda: s.is_sorted(), lambda: s.search_sorted(s.head(1)) if not s.dtype.is_nested() else None, lambda: s.to_pandas(), lambda: s.__repr__(), lambda: s[[0,-1]] if len(s) else s, lambda: s[1:-1:2], lambda: s.to_init_repr(), lambda: s.describe(), lambda: s.item() if len(s)==1 else None, lambda: s.n_chunks(), lambda: s.estimated_size(), lambda: s.value_counts() if not isinstance(s.dtype, pl.Struct) else None, lambda: s.unique() if not isinstance(s.dtype, pl.Struct) else None, lambda: s.to_dummies() if not s.dtype.is_nested() else None, lambda: s.reshape((-1,1)) if not s.dtype.is_nested() and len(s) else None, lambda: s.reshape((1,-1)).explode() if not s.dtype.is_nested() and len(s) else None, lambda: s.reinterpret() if s.dtype in (pl.Int64, pl.UInt64) else None, lambda: s.set(pl.Series([True]*len(s)), None) if not s.dtype.is_nested() else None, lambda: s.zip_with(pl.Series([True,False]*(len(s)//2+1)).head(len(s)), s.reverse()), lambda: s.clone().set_sorted().max(), lambda: s.sort().set_sorted().search_sorted(s.max()) if s.dtype.is_numeric() else None, lambda: s.arg_sort(), lambda: s.rle(), lambda: s.cumulative_eval(pl.element().first()), lambda: s.to_physical(), lambda: s.cast(pl.String) if not s.dtype.is_nested() else s.cast(pl.List(pl.String), strict=False), lambda: s.get_chunks(), lambda: pl.concat(s.get_chunks(), rechunk=False) if s.n_chunks()>0 else s, lambda: s.head(rng.randint(-3,3)), lambda: s.tail(rng.randint(-3,3)), lambda: s.limit(1), lambda: s.__array__() if not s.dtype.is_nested() else None, lambda: pl.Series(s.to_numpy()) if not s.dtype.is_nested() else None]
        k = rng.randint(1,5)
        for f in rng.sample(ops, k): f()
    except (pl.exceptions.PolarsError, TypeError, ValueError, NotImplementedError, IndexError, OverflowError, pl.exceptions.PanicException, AttributeError) as ex:
        log(f"  handled exc {type(ex).__name__}: {str(ex)[:200]}")
    return df

OPS=[op_groupby, op_sort, op_unique, op_join, op_explode, op_window, op_string, op_list, op_array, op_struct, op_cast, op_gather, op_filter, op_slice, op_concat, op_extend, op_pivot, op_unpivot, op_roundtrip, op_arith, op_temporal, op_rolling_groupby, op_asof, op_when, op_misc, op_series_ops]

df = rand_df()
log(f"start df schema={df.schema} h={df.height}")
for i in range(nops):
    if df.width == 0 or rng.random()<0.15 or df.height > 20000:
        df = rand_df()
        log(f"new df schema={df.schema} h={df.height}")
    try:
        out = op(df)
        if isinstance(out, pl.DataFrame) and out.width>0:
            # touch the result to exercise validity
            out.null_count(); out.head(3).to_dicts(); out.hash_rows() if all(not isinstance(t, pl.Object) for t in out.dtypes) else None
            if rng.random()<0.5: df = out
    except pl.exceptions.PanicException as ex:
        log(f"  PANIC: {str(ex)[:500]}")
    except (pl.exceptions.PolarsError, TypeError, ValueError, NotImplementedError, IndexError, OverflowError, ZeroDivisionError, AttributeError) as ex:
        log(f"  handled exc {type(ex).__name__}: {str(ex)[:200]}")
log("done")
