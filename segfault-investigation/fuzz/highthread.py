"""#27842 territory: many concurrent streaming queries using the async executor
(over/cum_sum/join) with a very large polars thread pool, to trip the executor's
work-stealing deque. Runs until SECONDS elapse. A crash kills the process.
Usage: python highthread.py SEED PYTHREADS SECONDS LOGFILE"""
import sys, os, time, threading, random, warnings
warnings.simplefilter("ignore")
import polars as pl, numpy as np
seed=int(sys.argv[1]); pyt=int(sys.argv[2]); secs=float(sys.argv[3]); logf=open(sys.argv[4],"a",buffering=1)
STOP=time.time()+secs
def work(tid):
    rng=random.Random(seed*131+tid)
    while time.time()<STOP:
        n=rng.choice([1000,20000,200000])
        df=pl.DataFrame({"g":np.random.randint(0,rng.choice([4,64,1000]),n),"h":np.random.randint(0,8,n),"v":np.random.randn(n)})
        try:
            k=rng.choice(["over","cumsum","gb","join","sort"])
            if k=="over": df.lazy().with_columns(pl.col("v").sum().over("g")).collect(engine="streaming")
            elif k=="cumsum": df.lazy().with_columns(pl.col("v").cum_sum().over(["g","h"])).collect(engine="streaming")
            elif k=="gb": df.lazy().group_by("g","h").agg(pl.col("v").sum(),pl.len()).collect(engine="streaming")
            elif k=="join": df.lazy().join(df.lazy().select("g","v"),on="g",how="inner").select(pl.len()).collect(engine="streaming")
            elif k=="sort": df.lazy().sort("g","v").collect(engine="streaming")
        except pl.exceptions.PanicException as e: logf.write(f"[s{seed}t{tid}] PANIC: {str(e)[:200]}\n")
        except Exception: pass
ts=[threading.Thread(target=work,args=(i,)) for i in range(pyt)]
for t in ts: t.start()
for t in ts: t.join()
logf.write(f"[s{seed}] done\n")
