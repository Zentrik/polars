"""Run fuzz3 op loops in N Python threads concurrently (shared process, shared thread pools).
Usage: python fuzz_threads.py SEED NTHREADS NOPS LOGPREFIX
"""
import sys, threading, warnings
warnings.simplefilter("ignore")
seed, nthreads, nops, prefix = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
src = open(__file__.replace("fuzz_threads.py", "fuzz3.py")).read()
def worker(i):
    ns = {"__name__": "fuzz3_%d" % i}
    exec(compile(src, "fuzz3.py", "exec"), ns)
    ns["main"](seed * 1000 + i, nops, f"{prefix}_t{i}.txt")
ts = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
for t in ts: t.start()
for t in ts: t.join()
print("all threads done")
