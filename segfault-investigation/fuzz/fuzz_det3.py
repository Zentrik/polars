"""Streaming-vs-in-memory correctness detector, noise-controlled. Builds random
frames whose GROUP KEYS are only exact-comparable types (int/str/bool/date/datetime,
no float/NaN/signed-zero) and random lazy pipelines, then compares the streaming
engine against the in-memory engine with per-column tolerance (exact for int/str/
bool/temporal, atol for float). Flags height mismatches and non-float data
divergences -- both indicate a real streaming-engine bug (often the same root as a
memory bug). Dumps a self-contained repro on each finding.
Usage: python fuzz_det3.py SEED NITER LOGFILE [REPRODIR]
"""
import sys, os, random, warnings, datetime as dt
warnings.simplefilter("ignore")
import polars as pl, numpy as np

seed=int(sys.argv[1]); niter=int(sys.argv[2]); logf=open(sys.argv[3],"a",buffering=1)
reprodir=sys.argv[4] if len(sys.argv)>4 else None
if reprodir: os.makedirs(reprodir, exist_ok=True)
rng=random.Random(seed)
def log(m): logf.write(f"[seed={seed}] {m}\n")

KEYTYPES=["i32","i64","u32","str","bool","date","datetime","cat"]
VALTYPES=["i32","i64","f64","str","bool"]
def col(name, n, types):
    t=rng.choice(types); nullp=rng.choice([0.0,0.1,0.4])
    nul=lambda v: None if rng.random()<nullp else v
    lo=rng.choice([2,3,5,20])  # low cardinality for keys
    if t=="i32": return pl.Series(name,[nul(rng.randint(0,lo)) for _ in range(n)],pl.Int32)
    if t=="i64": return pl.Series(name,[nul(rng.randint(-lo,lo)) for _ in range(n)],pl.Int64)
    if t=="u32": return pl.Series(name,[nul(rng.randint(0,lo)) for _ in range(n)],pl.UInt32)
    if t=="f64": return pl.Series(name,[nul(rng.choice([0.0,1.0,-1.0,2.5,rng.uniform(-5,5)])) for _ in range(n)],pl.Float64)
    if t=="str": return pl.Series(name,[nul(rng.choice(["a","bb","ccc","","x"*13,"é中"])) for _ in range(n)],pl.String)
    if t=="bool": return pl.Series(name,[nul(rng.random()<0.5) for _ in range(n)],pl.Boolean)
    if t=="date": return pl.Series(name,[nul(dt.date(2000,1,1)+dt.timedelta(days=rng.randint(0,lo))) for _ in range(n)],pl.Date)
    if t=="datetime": return pl.Series(name,[nul(dt.datetime(2000,1,1)+dt.timedelta(hours=rng.randint(0,lo))) for _ in range(n)],pl.Datetime("us"))
    if t=="cat": return pl.Series(name,[nul(rng.choice(["a","bb","ccc","x"*13])) for _ in range(n)],pl.Categorical)

def rand_df():
    n=rng.choice([0,1,5,17,64,257,1000,5000])
    nk=rng.randint(1,2); nv=rng.randint(1,3)
    cols=[col(f"k{i}",n,KEYTYPES) for i in range(nk)]+[col(f"v{i}",n,VALTYPES) for i in range(nv)]
    # random chunking
    df=pl.DataFrame(cols)
    if rng.random()<0.5 and n>2:
        k=rng.randint(2,4); cuts=sorted(rng.sample(range(1,n),min(k-1,n-1)))
        parts=[]; prev=0
        for c in cuts+[n]: parts.append(df.slice(prev,c-prev)); prev=c
        df=pl.concat(parts, rechunk=rng.random()<0.3)
    return df

def build(df, rec):
    lf=df.lazy()
    keycols=[c for c in df.columns if c.startswith("k")]
    valcols=[c for c in df.columns if c.startswith("v")]
    # ONLY order-insensitive ops: the resulting MULTISET of rows must be identical
    # across runs and engines. No head/slice/first/last/keep-first (those depend on
    # unspecified row order after joins/groupbys and are not bugs when they vary).
    for _ in range(rng.randint(1,4)):
        cols=lf.collect_schema().names()
        sch=lf.collect_schema()
        kc=[c for c in cols if c.startswith("k")] or cols
        kind=rng.choice(["group_by","unique","filter","join","over","group_by","join"])
        try:
            if kind=="group_by":
                keys=rng.sample(kc,rng.randint(1,min(2,len(kc))))
                others=[c for c in cols if c not in keys]
                aggs=[pl.len().alias("len")]
                for c in others[:4]:
                    t=sch[c]
                    if t.is_numeric(): aggs.append(rng.choice([pl.col(c).sum(),pl.col(c).min(),pl.col(c).max(),pl.col(c).n_unique(),pl.col(c).count()]).alias(c+"_a"))
                    elif not t.is_nested(): aggs.append(rng.choice([pl.col(c).min(),pl.col(c).max(),pl.col(c).n_unique(),pl.col(c).count()]).alias(c+"_a"))
                lf=lf.group_by(keys,maintain_order=True).agg(aggs); rec.append(("group_by",keys))
            elif kind=="unique":
                # full-row unique only -> multiset-deterministic
                lf=lf.unique(maintain_order=True); rec.append(("unique_full",))
            elif kind=="filter":
                c=rng.choice(cols); lf=lf.filter(pl.col(c).is_not_null()); rec.append(("filter",c))
            elif kind=="join":
                jon=rng.choice(kc); how=rng.choice(["inner","left","semi","anti","full"])
                other=df.select(rng.sample(df.columns,min(2,len(df.columns)))).lazy()
                oc=other.collect_schema().names()
                if jon in oc: lf=lf.join(other,on=jon,how=how,nulls_equal=rng.random()<0.5,coalesce=True); rec.append(("join",jon,how))
            elif kind=="over":
                part=rng.choice(kc); vc=[c for c in cols if c.startswith("v") and sch[c].is_numeric()][:2]
                if vc: lf=lf.with_columns([rng.choice([pl.col(c).sum(),pl.col(c).max(),pl.col(c).min()]).over(part).alias(c+"_o") for c in vc]); rec.append(("over",part))
        except Exception: break
    return lf

def compare(a, b):
    """True if equal within float tolerance and exact elsewhere. Order-independent."""
    if a.height!=b.height: return False, f"height {a.height} vs {b.height}"
    if a.schema!=b.schema: return False, f"schema {a.schema} vs {b.schema}"
    if a.height==0: return True, ""
    sortcols=[c for c,t in a.schema.items() if not t.is_nested()]
    try:
        a=a.sort(sortcols,maintain_order=True,nulls_last=True); b=b.sort(sortcols,maintain_order=True,nulls_last=True)
    except Exception: pass
    for c,t in a.schema.items():
        ca, cb = a[c], b[c]
        if t.is_float():
            d=(ca.fill_null(0).fill_nan(0)-cb.fill_null(0).fill_nan(0)).abs().max()
            if d is not None and d>1e-6: return False, f"col {c} float diff {d}"
            if (ca.is_null()!=cb.is_null()).any(): return False, f"col {c} null pattern"
        else:
            try:
                if not ca.equals(cb): return False, f"col {c} ({t}) differs"
            except Exception:
                if ca.to_list()!=cb.to_list(): return False, f"col {c} ({t}) list differs"
    return True, ""

for it in range(niter):
    df=rand_df(); rec=[]
    try:
        lf=build(df,rec)
    except Exception: continue
    if os.environ.get("FUZZ_DUMP_ALL") and reprodir:
        try: open(os.path.join(reprodir,f"plan_s{seed}_i{it}.plan"),"wb").write(lf.serialize())
        except Exception: pass
    try:
        mem=lf.collect(engine="in-memory")
        s1=lf.collect(engine="streaming"); s2=lf.collect(engine="streaming")
    except pl.exceptions.PanicException as e:
        which="streaming"
        try: lf.collect(engine="in-memory"); 
        except pl.exceptions.PanicException: which="both"
        except Exception: which="streaming-only"
        log(f"iter {it} PANIC({which}): {str(e)[:400]} query={rec} schema={df.schema}")
        if reprodir:
            try:
                open(os.path.join(reprodir,f"PANIC_s{seed}_i{it}.plan"),"wb").write(lf.serialize())
                df.write_ipc(os.path.join(reprodir,f"PANIC_s{seed}_i{it}.ipc")); open(os.path.join(reprodir,f"PANIC_s{seed}_i{it}.q"),"w").write(repr(rec))
            except Exception as se: log(f"   serialize failed: {se}")
        continue
    except Exception: continue
    ok_ss,why_ss=compare(s1,s2)
    ok_sm,why_sm=compare(s1,mem)
    if not ok_ss or not ok_sm:
        tag = "NONDET_STREAM" if not ok_ss else "STREAM_VS_MEM"
        log(f"iter {it} !!! {tag}: {why_ss if not ok_ss else why_sm} query={rec} schema={df.schema} h_mem={mem.height} h_s={s1.height}")
        if reprodir:
            try: df.write_ipc(os.path.join(reprodir,f"{tag}_s{seed}_i{it}.ipc")); open(os.path.join(reprodir,f"{tag}_s{seed}_i{it}.q"),"w").write(repr(rec))
            except Exception: pass
log("done")
