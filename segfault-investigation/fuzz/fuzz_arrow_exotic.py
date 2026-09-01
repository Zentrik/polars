"""Exotic Arrow FFI import fuzzer: map, union (sparse/dense), run-end-encoded,
dictionary-of-nested, deeply-nested (list/struct/fsl mixes), extreme offsets/slices.
These are the corners of the C-data import surface. Import into polars via several
entry points and exercise. Any crash kills the process.
Usage: python fuzz_arrow_exotic.py SEED NITER LOGFILE [REPRODIR]"""
import sys, os, random, warnings
warnings.simplefilter("ignore")
import polars as pl, pyarrow as pa
seed=int(sys.argv[1]); niter=int(sys.argv[2]); logf=open(sys.argv[3],"a",buffering=1)
rng=random.Random(seed)
def log(m): logf.write(f"[seed={seed}] {m}\n"); logf.flush()
def strs(n): 
    pool=["","a","é","x"*9,"y"*13,"z"*40]
    return [None if rng.random()<0.2 else rng.choice(pool) for _ in range(n)]
def leaf(n):
    t=rng.choice(["i64","f64","str","lstr","strview","bin","bool"])
    nul=lambda v: None if rng.random()<0.2 else v
    if t=="i64": return pa.array([nul(rng.randint(-10**9,10**9)) for _ in range(n)], pa.int64())
    if t=="f64": return pa.array([nul(rng.uniform(-1,1)) for _ in range(n)], pa.float64())
    if t=="str": return pa.array(strs(n), pa.string())
    if t=="lstr": return pa.array(strs(n), pa.large_string())
    if t=="strview": return pa.array(strs(n), pa.string_view())
    if t=="bin": return pa.array([None if v is None else v.encode() for v in strs(n)], pa.binary())
    if t=="bool": return pa.array([nul(rng.random()<.5) for _ in range(n)], pa.bool_())
def nested(n, d=0):
    if d>=3 or rng.random()<0.4: return leaf(max(n,1))
    k=rng.choice(["list","llist","fsl","struct","map","dict","ree"])
    try:
        if k=="list":
            ch=nested(n*2,d+1); offs=[0]
            for _ in range(n): offs.append(min(len(ch),offs[-1]+rng.choice([0,1,2])))
            return pa.ListArray.from_arrays(pa.array(offs,pa.int32()),ch)
        if k=="llist":
            ch=nested(n*2,d+1); offs=[0]
            for _ in range(n): offs.append(min(len(ch),offs[-1]+rng.choice([0,1,2])))
            return pa.LargeListArray.from_arrays(pa.array(offs,pa.int64()),ch)
        if k=="fsl":
            w=rng.choice([1,2,3]); return pa.FixedSizeListArray.from_arrays(nested(n*w,d+1),w)
        if k=="struct":
            fs=[(f"f{j}",nested(n,d+1)) for j in range(rng.choice([1,2]))]
            return pa.StructArray.from_arrays([a for _,a in fs],names=[f for f,_ in fs])
        if k=="map":
            keys=pa.array([f"k{i%3}" for i in range(n*2)]); items=nested(n*2,d+1); offs=[0]
            for _ in range(n): offs.append(min(len(keys),offs[-1]+rng.choice([0,1,2])))
            return pa.MapArray.from_arrays(pa.array(offs,pa.int32()),keys,items)
        if k=="dict":
            vals=leaf(rng.choice([2,4])); idx=pa.array([rng.randint(0,len(vals)-1) if len(vals) else 0 for _ in range(n)],pa.int32())
            return pa.DictionaryArray.from_arrays(idx,vals)
        if k=="ree":
            vals=leaf(rng.choice([1,2,3])); ends=[]
            c=0
            for i in range(len(vals)): c+=rng.choice([1,2,3]); ends.append(c)
            re=pa.array(ends,pa.int32())
            return pa.RunEndEncodedArray.from_arrays(re, vals)
    except Exception:
        return leaf(max(n,1))
def slicechunk(a):
    if rng.random()<0.6 and len(a)>1:
        o=rng.randint(0,len(a)-1); a=a.slice(o,rng.randint(0,len(a)-o))
    if rng.random()<0.4 and len(a)>=2:
        m=len(a)//2; a=pa.chunked_array([a.slice(0,m),a.slice(m)]) if rng.random()<0.5 else a
    return a
for it in range(niter):
    n=rng.choice([0,1,3,17,100])
    try: arr=slicechunk(nested(n))
    except Exception: continue
    entry=rng.choice(["series","from_arrow","df","batch"])
    try:
        if entry=="series": s=pl.Series("x",arr)
        elif entry=="from_arrow": o=pl.from_arrow(arr); s=o if isinstance(o,pl.Series) else o.to_series()
        elif entry=="df": s=pl.from_arrow(pa.table({"x":arr}))["x"]
        elif entry=="batch":
            a2=arr.combine_chunks() if isinstance(arr,pa.ChunkedArray) else arr
            s=pl.from_arrow(pa.record_batch({"x":a2}))["x"]
    except pl.exceptions.PanicException as e:
        log(f"iter {it} IMPORT PANIC entry={entry} type={arr.type}: {str(e)[:180]}"); continue
    except Exception: continue
    for op in rng.sample([lambda:s.to_list(), lambda:s.to_arrow(), lambda:s.rechunk().to_list(),
                          lambda:s.slice(1,3).to_list(), lambda:pl.Series("y",s.to_arrow()).to_list(),
                          lambda:s.to_frame().to_arrow(), lambda:s.head(3).to_list()], 4):
        try: op()
        except pl.exceptions.PanicException as e: log(f"iter {it} OP PANIC entry={entry} type={arr.type}: {str(e)[:180]}"); break
        except Exception: pass
log("done")
