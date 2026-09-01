"""Arrow FFI import fuzzer. Builds a huge variety of *valid* pyarrow array layouts
(sliced, offset, chunked, nested, dictionary, REE, all types) and imports each into
polars through several entry points, then exercises the data. The Arrow C-data /
FFI boundary is where real memory-unsafety lives (cf. #28626 nested-string segfault).
Any crash kills the process; mismatches vs pyarrow's own to_pylist are logged.
Usage: python fuzz_arrow_import.py SEED NITER LOGFILE
"""
import sys, os, random, warnings
warnings.simplefilter("ignore")
import polars as pl, pyarrow as pa, numpy as np

seed=int(sys.argv[1]); niter=int(sys.argv[2]); logf=open(sys.argv[3],"a",buffering=1)
rng=random.Random(seed)
def log(m): logf.write(f"[seed={seed}] {m}\n"); logf.flush()
REPRO=sys.argv[4] if len(sys.argv)>4 else None
if REPRO: os.makedirs(REPRO, exist_ok=True)
_dc=[0]
def dump_arr(tag, arr, entry, it):
    if not REPRO: return
    _dc[0]+=1
    try:
        import pyarrow as _pa
        a=arr.combine_chunks() if isinstance(arr,_pa.ChunkedArray) else arr
        t=_pa.table({"x":a})
        import pyarrow.ipc as _ipc
        with _pa.OSFile(os.path.join(REPRO,f"{tag}_s{seed}_i{it}_{_dc[0]}.arrow"),"wb") as f:
            w=_ipc.new_file(f,t.schema); w.write_table(t); w.close()
        open(os.path.join(REPRO,f"{tag}_s{seed}_i{it}_{_dc[0]}.meta"),"w").write(f"entry={entry} type={arr.type}\n")
    except Exception as _e: log(f"   dump failed {_e}")

def rand_strs(n):
    pool=["","a","é","中","😀","x"*9,"y"*12,"z"*13,"w"*16,"q"*40,"m"*200]
    return [None if rng.random()<0.2 else rng.choice(pool) for _ in range(n)]

def leaf_array(n, depth=0):
    t=rng.choice(["i8","i16","i32","i64","u8","u32","u64","f32","f64","bool","str","lstr","strview","bin","lbin","binview","date32","ts","time64","dur","dec128","dec256","null","i128"])
    nul=lambda v: None if rng.random()<0.25 else v
    if t=="i8": a=pa.array([nul(rng.randint(-128,127)) for _ in range(n)], pa.int8())
    elif t=="i16": a=pa.array([nul(rng.randint(-2**15,2**15-1)) for _ in range(n)], pa.int16())
    elif t=="i32": a=pa.array([nul(rng.randint(-2**31,2**31-1)) for _ in range(n)], pa.int32())
    elif t=="i64": a=pa.array([nul(rng.randint(-2**63,2**63-1)) for _ in range(n)], pa.int64())
    elif t=="u8": a=pa.array([nul(rng.randint(0,255)) for _ in range(n)], pa.uint8())
    elif t=="u32": a=pa.array([nul(rng.randint(0,2**32-1)) for _ in range(n)], pa.uint32())
    elif t=="u64": a=pa.array([nul(rng.randint(0,2**64-1)) for _ in range(n)], pa.uint64())
    elif t=="f32": a=pa.array([nul(rng.choice([0.0,-0.0,1.5,float('nan'),float('inf')])) for _ in range(n)], pa.float32())
    elif t=="f64": a=pa.array([nul(rng.uniform(-1e6,1e6)) for _ in range(n)], pa.float64())
    elif t=="bool": a=pa.array([nul(rng.random()<0.5) for _ in range(n)], pa.bool_())
    elif t=="str": a=pa.array(rand_strs(n), pa.string())
    elif t=="lstr": a=pa.array(rand_strs(n), pa.large_string())
    elif t=="strview": a=pa.array(rand_strs(n), pa.string_view())
    elif t=="bin": a=pa.array([None if v is None else (v.encode() if isinstance(v,str) else v) for v in rand_strs(n)], pa.binary())
    elif t=="lbin": a=pa.array([None if v is None else v.encode() for v in rand_strs(n)], pa.large_binary())
    elif t=="binview": a=pa.array([None if v is None else v.encode() for v in rand_strs(n)], pa.binary_view())
    elif t=="date32": a=pa.array([nul(rng.randint(-100000,100000)) for _ in range(n)], pa.date32())
    elif t=="ts": a=pa.array([nul(rng.randint(-10**15,10**15)) for _ in range(n)], pa.timestamp(rng.choice(["s","ms","us","ns"]), rng.choice([None,"UTC","Europe/Amsterdam"])))
    elif t=="time64": a=pa.array([nul(rng.randint(0,86_399_000_000)) for _ in range(n)], pa.time64("us"))
    elif t=="dur": a=pa.array([nul(rng.randint(-10**12,10**12)) for _ in range(n)], pa.duration(rng.choice(["s","ms","us","ns"])))
    elif t=="dec128": a=pa.array([nul(rng.randint(-10**15,10**15)) for _ in range(n)], pa.decimal128(rng.choice([19,28,38]), rng.choice([0,2])))
    elif t=="dec256": a=pa.array([nul(rng.randint(-10**30,10**30)) for _ in range(n)], pa.decimal256(rng.choice([40,60]), rng.choice([0,5])))
    elif t=="i128":
        a=pa.array([nul(rng.randint(-10**15,10**15)) for _ in range(n)], pa.decimal128(38,0))
    elif t=="null": a=pa.nulls(n)
    # dictionary-encode sometimes
    if rng.random()<0.25 and t in ("str","lstr","i32","i64","u32"):
        try: a=a.dictionary_encode()
        except Exception: pass
    return a

def nested_array(n, depth=0):
    if depth>=2 or rng.random()<0.5:
        return leaf_array(n, depth)
    kind=rng.choice(["list","llist","fsl","struct","listview"])
    child_n = n*rng.choice([0,1,2,3])
    child = nested_array(max(child_n,1), depth+1)
    if kind=="list":
        offs=[0]; 
        for _ in range(n): offs.append(min(len(child), offs[-1]+rng.choice([0,1,2,3])))
        return pa.ListArray.from_arrays(pa.array(offs, pa.int32()), child)
    if kind=="llist":
        offs=[0]
        for _ in range(n): offs.append(min(len(child), offs[-1]+rng.choice([0,1,2])))
        return pa.LargeListArray.from_arrays(pa.array(offs, pa.int64()), child)
    if kind=="fsl":
        w=rng.choice([1,2,3]); child2=nested_array(n*w, depth+1)
        return pa.FixedSizeListArray.from_arrays(child2, w)
    if kind=="struct":
        nf=rng.choice([1,2,3]); fields=[(f"f{j}", nested_array(n, depth+1)) for j in range(nf)]
        names=[f for f,_ in fields]; arrs=[a for _,a in fields]
        mask=pa.array([rng.random()<0.2 for _ in range(n)]) if rng.random()<0.5 else None
        try: return pa.StructArray.from_arrays(arrs, names=names, mask=mask)
        except Exception: return pa.StructArray.from_arrays(arrs, names=names)
    if kind=="listview":
        try:
            offs=[rng.randint(0,max(0,len(child)-1)) for _ in range(n)]
            szs=[rng.choice([0,1,2]) for _ in range(n)]
            return pa.ListViewArray.from_arrays(pa.array(offs,pa.int32()), pa.array(szs,pa.int32()), child)
        except Exception:
            return leaf_array(n, depth)

def maybe_slice_chunk(a):
    # apply random slice (creates offset) and/or chunking
    if rng.random()<0.6 and len(a)>1:
        off=rng.randint(0,len(a)-1); ln=rng.randint(0,len(a)-off); a=a.slice(off,ln)
    if rng.random()<0.4 and len(a)>=2:
        k=rng.randint(2,4); cuts=sorted(rng.sample(range(1,len(a)),min(k-1,len(a)-1)))
        parts=[]; prev=0
        for c in cuts+[len(a)]: parts.append(a.slice(prev,c-prev)); prev=c
        return pa.chunked_array(parts) if rng.random()<0.5 else a
    return a

def exercise(s):
    ops=[lambda: s.to_list(), lambda: (s.to_arrow(), None)[1], lambda: pl.Series("x", s.to_arrow()).to_list(),
         lambda: s.to_arrow(compat_level=pl.CompatLevel.newest()), lambda: s.slice(1,3).to_list(),
         lambda: s.rechunk().to_list(), lambda: s.head(5).to_list(), lambda: (s.to_frame().write_ipc(os.devnull) if False else s.to_frame().to_arrow())]
    if not s.dtype.is_nested():
        ops += [lambda: s.sort().to_list(), lambda: s.unique().len(), lambda: s.filter(s.is_not_null()).len(),
                lambda: s.hash().sum(), lambda: s.to_frame().group_by("x").len().height if s.name=="x" else None,
                lambda: s.gather([0, len(s)-1]).to_list() if len(s)>0 else None]
    for f in rng.sample(ops, min(len(ops), rng.randint(2,6))):
        try: f()
        except (pl.exceptions.PolarsError, TypeError, ValueError, OverflowError, pa.lib.ArrowException) as e: pass

for it in range(niter):
    n=rng.choice([0,1,2,5,17,64,100,1000])
    try:
        arr=nested_array(n)
        arr=maybe_slice_chunk(arr)
    except Exception as e:
        continue
    # multiple import entry points
    entry=rng.choice(["series","from_arrow","df_table","from_arrow_batch","series_chunked"])
    try:
        if entry=="series": s=pl.Series("x", arr)
        elif entry=="from_arrow":
            o=pl.from_arrow(arr); s=o if isinstance(o, pl.Series) else o.to_series()
        elif entry=="df_table":
            t=pa.table({"x": arr}); s=pl.from_arrow(t)["x"]
        elif entry=="from_arrow_batch":
            if isinstance(arr, pa.ChunkedArray): arr2=arr.combine_chunks()
            else: arr2=arr
            b=pa.record_batch({"x": arr2}); s=pl.from_arrow(b)["x"]
        elif entry=="series_chunked":
            ca=arr if isinstance(arr, pa.ChunkedArray) else pa.chunked_array([arr])
            s=pl.Series("x", ca)
    except pl.exceptions.PanicException as e:
        log(f"iter {it} IMPORT PANIC entry={entry} type={arr.type}: {str(e)[:150]}"); dump_arr("IMPORTPANIC", arr, entry, it); continue
    except (pl.exceptions.PolarsError, TypeError, ValueError, pa.lib.ArrowException, NotImplementedError) as e:
        continue
    # correctness: round-trip vs pyarrow's own view
    try:
        got=s.to_list()
        exp=(arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr).to_pylist()
        if got!=exp:
            log(f"iter {it} DATA MISMATCH entry={entry} type={arr.type}: got[:4]={str(got[:4])[:80]} exp[:4]={str(exp[:4])[:80]}")
    except pl.exceptions.PanicException as e:
        log(f"iter {it} TOLIST PANIC entry={entry} type={arr.type}: {str(e)[:150]}"); dump_arr("TOLISTPANIC", arr, entry, it); continue
    except Exception: pass
    try:
        exercise(s)
    except pl.exceptions.PanicException as e:
        log(f"iter {it} EXERCISE PANIC entry={entry} type={arr.type}: {str(e)[:150]}"); dump_arr("EXERCISEPANIC", arr, entry, it)
log("done")
