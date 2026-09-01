"""Multi-threaded stress of global-state / Python-callback paths: categoricals & enums, Object dtype,
map_elements/map_batches callbacks, arrow/pandas/numpy conversions, IO plugins, streaming sinks.
Each thread verifies results; any segfault kills the process (detected by the runner).
Usage: python fuzz_misc_threads.py SEED NTHREADS NITER LOGPREFIX
"""
import sys, random, threading, warnings, io, os, tempfile, gc
warnings.simplefilter("ignore")
import polars as pl, numpy as np, pyarrow as pa

seed, nthreads, niter, prefix = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
WORDS = [f"w{i}" for i in range(200)] + ["", "é", "中文", "😀" * 5, "x" * 13, "y" * 12, "z" * 100]

def worker(tid):
    rng = random.Random(seed * 1000 + tid)
    logf = open(f"{prefix}_t{tid}.txt", "a", buffering=1)
    def log(m): logf.write(f"[t{tid}] {m}\n")
    def mism(name, got, exp):
        if got != exp: log(f"  !!! MISMATCH {name}: got={str(got)[:120]} exp={str(exp)[:120]}")
    for it in range(niter):
        op = rng.choice(["cat_ops", "enum_ops", "cat_join", "cat_concat", "object_ops", "map_elements", "map_batches", "conversions", "io_plugin", "sink", "string_cache", "cat_streaming", "numpy_ufunc", "struct_cat"])
        log(f"iter {it} op={op}")
        try:
            n = rng.choice([1, 10, 500, 5000])
            words = [rng.choice(WORDS) for _ in range(n)] + [None] * (n // 10)
            rng.shuffle(words)
            if op == "cat_ops":
                s = pl.Series("c", words, pl.Categorical)
                mism("cat to_list", s.to_list(), words)
                mism("cat cast str", s.cast(pl.String).to_list(), words)
                mism("cat unique", sorted(s.unique().cast(pl.String).to_list(), key=lambda v: (v is None, v or "")), sorted(set(words), key=lambda v: (v is None, v or "")))
                mism("cat sort", s.sort(nulls_last=True).cast(pl.String).to_list(), sorted([w for w in words if w is not None]) + [None] * words.count(None))
                mism("cat eq", (s == pl.Series(words, dtype=pl.Categorical)).to_list() if False else (s == rng.choice(WORDS)).null_count(), words.count(None))
                mism("cat n_unique", s.n_unique(), len(set(words)))
                mism("cat value_counts", s.value_counts().height, len(set(words)))
                mism("cat gather", s.gather([0, n - 1] if n > 1 else [0]).cast(pl.String).to_list(), [words[0], words[n - 1]] if n > 1 else [words[0]])
                mism("cat hash", s.hash().len(), len(words))
                mism("cat to_physical", s.to_physical().len(), len(words))
                mism("cat arrow rt", pl.Series("c", s.to_arrow()).cast(pl.String).to_list(), words)
                mism("cat slice", s.slice(1, 3).cast(pl.String).to_list(), words[1:4])
                mism("cat is_in", s.is_in(["w1", "w2"]).sum(), sum(1 for w in words if w in ("w1", "w2")))
                mism("cat group_by", s.to_frame().group_by("c").len().height, len(set(words)))
            elif op == "enum_ops":
                cats = list(WORDS)
                s = pl.Series("e", words, pl.Enum(cats))
                mism("enum to_list", s.to_list(), words)
                mism("enum sort", s.sort(nulls_last=True).cast(pl.String).to_list(), sorted([w for w in words if w is not None], key=cats.index) + [None] * words.count(None))
                mism("enum cmp", (s > "w5").null_count(), words.count(None))
                mism("enum cat cast", s.cast(pl.Categorical).cast(pl.String).to_list(), words)
                mism("enum min", s.min(), min((w for w in words if w is not None), key=cats.index, default=None))
                mism("enum arrow rt", pl.Series("e", s.to_arrow()).cast(pl.String).to_list(), words)
                mism("enum unique", s.unique().len(), len(set(words)))
            elif op == "cat_join":
                a = pl.DataFrame({"k": pl.Series(words, dtype=pl.Categorical), "v": range(len(words))})
                other_words = [rng.choice(WORDS) for _ in range(rng.choice([1, 50, 300]))]
                b = pl.DataFrame({"k": pl.Series(other_words, dtype=pl.Categorical), "w": range(len(other_words))})
                for how in ["inner", "left", "semi", "anti", "full"]:
                    got = a.join(b, on="k", how=how, nulls_equal=False).height
                    bset = {}
                    for w in other_words: bset[w] = bset.get(w, 0) + 1
                    if how == "inner": exp = sum(bset.get(w, 0) for w in words if w is not None)
                    elif how == "left": exp = sum(max(1, bset.get(w, 0)) if w is not None else 1 for w in words)
                    elif how == "semi": exp = sum(1 for w in words if w is not None and w in bset)
                    elif how == "anti": exp = sum(1 for w in words if w is None or w not in bset)
                    else: exp = sum(max(1, bset.get(w, 0)) if w is not None else 1 for w in words) + sum(c for w, c in bset.items() if w not in set(words))
                    mism(f"cat join {how}", got, exp)
                mism("cat join streaming", a.lazy().join(b.lazy(), on="k", how="inner").collect(engine="streaming").height, sum(sum(1 for o in other_words if o == w) for w in words if w is not None))
            elif op == "cat_concat":
                parts = [pl.Series("c", [rng.choice(WORDS) for _ in range(rng.randint(0, 50))], pl.Categorical) for _ in range(rng.randint(1, 6))]
                exp = sum([p.cast(pl.String).to_list() for p in parts], [])
                s = pl.concat(parts, rechunk=rng.random() < 0.5)
                mism("cat concat", s.cast(pl.String).to_list(), exp)
                mism("cat concat unique", s.n_unique(), len(set(exp)))
                df = pl.concat([p.to_frame() for p in parts], how="vertical")
                mism("cat df concat", df["c"].cast(pl.String).to_list(), exp)
                mism("cat df concat streaming", pl.concat([p.to_frame().lazy() for p in parts]).collect(engine="streaming")["c"].cast(pl.String).to_list(), exp)
            elif op == "object_ops":
                objs = [object() if rng.random() < 0.8 else None for _ in range(min(n, 500))]
                s = pl.Series("o", objs, dtype=pl.Object)
                got = s.to_list()
                mism("object identity", [o is g for o, g in zip(objs, got)], [True] * len(objs))
                df = pl.DataFrame({"o": s, "i": range(len(objs))})
                mism("object filter", len(df.filter(pl.col("i") % 2 == 0)["o"].to_list()), (len(objs) + 1) // 2)
                mism("object slice", df.slice(1, 5)["o"].to_list(), objs[1:6])
                mism("object gather", df[[0, len(objs) - 1]]["o"].to_list() if len(objs) > 1 else df[[0]]["o"].to_list(), [objs[0], objs[-1]] if len(objs) > 1 else [objs[0]])
                mism("object concat", pl.concat([df, df])["o"].len(), 2 * len(objs))
                mism("object sort by", df.sort("i", descending=True)["o"].to_list(), objs[::-1])
                mism("object rechunk", pl.concat([df, df], rechunk=True)["o"].to_list(), objs + objs)
                mism("object map", df.select(pl.col("o").map_elements(lambda x: 1, return_dtype=pl.Int64))["o"].sum(), sum(1 for o in objs if o is not None))
                str(df); df.head(3).to_dicts()
                del s, df, got; gc.collect()
            elif op == "map_elements":
                df = pl.DataFrame({"a": list(range(n)), "s": [rng.choice(WORDS) or "q" for _ in range(n)]})
                mism("map_elements int", df.select(pl.col("a").map_elements(lambda x: x * 2, return_dtype=pl.Int64))["a"].to_list(), [x * 2 for x in range(n)])
                mism("map_elements str", df.select(pl.col("s").map_elements(lambda x: x + "!", return_dtype=pl.String))["s"].to_list(), [w + "!" for w in df["s"].to_list()])
                mism("map_batches groupby", df.group_by(pl.col("a") % 3).agg(pl.col("a").map_batches(lambda s: s.sum(), return_dtype=pl.Int64).alias("x")).sort("a")["x"].to_list(), [sum(x for x in range(n) if x % 3 == r) for r in range(min(3, n))])
                mism("map_batches over", df.select(pl.col("a").map_batches(lambda s: s.max(), return_dtype=pl.Int64).over(pl.col("a") % 2))["a"].to_list(), [max(x for x in range(n) if x % 2 == v % 2) for v in range(n)])
                mism("map_elements streaming", df.lazy().select(pl.col("a").map_elements(lambda x: x + 1, return_dtype=pl.Int64)).collect(engine="streaming")["a"].to_list(), [x + 1 for x in range(n)])
                mism("map_elements struct", df.select(pl.struct("a", "s").map_elements(lambda d: d["a"], return_dtype=pl.Int64))["a"].to_list(), list(range(n)))
                if rng.random() < 0.3:
                    try: df.select(pl.col("a").map_elements(lambda x: 1 / 0, return_dtype=pl.Float64))
                    except Exception: pass
            elif op == "map_batches":
                df = pl.DataFrame({"a": list(range(n))})
                mism("map_batches", df.select(pl.col("a").map_batches(lambda s: s * 3))["a"].to_list(), [x * 3 for x in range(n)])
                mism("map_batches np", df.select(pl.col("a").map_batches(lambda s: np.sqrt(s.to_numpy()), return_dtype=pl.Float64))["a"].len(), n)
                mism("map_batches streaming", df.lazy().select(pl.col("a").map_batches(lambda s: s + 1, return_dtype=pl.Int64)).collect(engine="streaming")["a"].to_list(), [x + 1 for x in range(n)])
                mism("lf map_batches", df.lazy().map_batches(lambda d: d.with_columns(b=pl.col("a") * 2), schema={"a": pl.Int64, "b": pl.Int64}).collect(engine=rng.choice(["streaming", "in-memory"]))["b"].to_list(), [x * 2 for x in range(n)])
                mism("groupby map_groups", df.with_columns(g=pl.col("a") % 2).group_by("g").map_groups(lambda d: d.head(1)).height, min(2, n))
            elif op == "conversions":
                df = pl.DataFrame({"a": list(range(n)), "s": [rng.choice(WORDS) for _ in range(n)], "c": pl.Series([rng.choice(WORDS) for _ in range(n)], dtype=pl.Categorical), "l": [[rng.choice(WORDS)] for _ in range(n)], "st": [{"x": i, "y": rng.choice(WORDS)} for i in range(n)]})
                sl = df.slice(rng.randint(0, max(0, n - 1)), rng.choice([None, 3]))
                t = sl.to_arrow(compat_level=rng.choice([pl.CompatLevel.newest(), pl.CompatLevel.oldest()]))
                t.validate(full=True)
                mism("to_arrow rt", pl.from_arrow(t).select(pl.col("c").cast(pl.String)).rows(), sl.select(pl.col("c").cast(pl.String)).rows())
                mism("from_arrow rt full", pl.from_arrow(t).with_columns(pl.col("c").cast(pl.String)).equals(sl.with_columns(pl.col("c").cast(pl.String))), True)
                pdf = sl.to_pandas(use_pyarrow_extension_array=rng.random() < 0.5)
                mism("pandas rt", pl.from_pandas(pdf)["a"].to_list(), sl["a"].to_list())
                mism("numpy rt", pl.Series(sl["a"].to_numpy()).to_list(), sl["a"].to_list())
                mism("pa.table", pa.table(sl).num_rows, sl.height)
                mism("pa.array", pa.array(sl["s"]).to_pylist(), sl["s"].to_list())
                r = pa.RecordBatchReader.from_stream(sl); mism("stream rt", r.read_all().num_rows, sl.height)
                mism("pickle rt", pl.DataFrame.deserialize(io.BytesIO(sl.serialize()))["a"].to_list(), sl["a"].to_list())
                mism("dicts rt", pl.DataFrame(sl.to_dicts(), schema=sl.schema)["a"].to_list() if sl.height else [], sl["a"].to_list())
            elif op == "io_plugin":
                from polars.io.plugins import register_io_source
                def src(with_columns, predicate, n_rows, batch_size):
                    for i in range(3):
                        d = pl.DataFrame({"a": list(range(i * 10, i * 10 + 10)), "s": [rng.choice(WORDS) or "" for _ in range(10)]})
                        if with_columns: d = d.select(with_columns)
                        if predicate is not None: d = d.filter(predicate)
                        yield d
                lf = register_io_source(src, schema={"a": pl.Int64, "s": pl.String})
                mism("io_plugin", lf.select("a").collect(engine=rng.choice(["streaming", "in-memory"]))["a"].to_list(), list(range(30)))
                mism("io_plugin filter", lf.filter(pl.col("a") > 5).collect(engine=rng.choice(["streaming", "in-memory"]))["a"].to_list(), list(range(6, 30)))
                import pyarrow.dataset as pads
                tbl = pa.table({"a": list(range(n)), "s": [rng.choice(WORDS) or "" for _ in range(n)]})
                mism("pyarrow_dataset", pl.scan_pyarrow_dataset(pads.dataset(tbl)).filter(pl.col("a") % 2 == 0).collect(engine=rng.choice(["streaming", "in-memory"]))["a"].to_list(), list(range(0, n, 2)))
            elif op == "sink":
                d = tempfile.mkdtemp()
                lf = pl.LazyFrame({"a": list(range(n)), "s": [rng.choice(WORDS) for _ in range(n)], "c": pl.Series([rng.choice(WORDS) for _ in range(n)], dtype=pl.Categorical)})
                fmt = rng.choice(["parquet", "ipc", "csv", "ndjson"])
                path = os.path.join(d, "out." + fmt)
                getattr(lf, "sink_" + fmt)(path, engine="streaming")
                back = {"parquet": pl.read_parquet, "ipc": pl.read_ipc, "csv": pl.read_csv, "ndjson": pl.read_ndjson}[fmt](path)
                mism("sink " + fmt, back["a"].to_list(), list(range(n)))
                if fmt == "parquet" and hasattr(pl, "PartitionByKey"):
                    lf.sink_parquet(pl.PartitionByKey(d + "/part", by="c"), engine="streaming", mkdir=True)
                    mism("sink partitioned", pl.scan_parquet(d + "/part/**/*.parquet").collect().height, n)
                import shutil; shutil.rmtree(d, ignore_errors=True)
            elif op == "string_cache":
                with pl.StringCache():
                    a = pl.Series("a", words, pl.Categorical); b = pl.Series("b", [rng.choice(WORDS) for _ in range(n)], pl.Categorical)
                    mism("string_cache append", pl.concat([a, b]).cast(pl.String).to_list(), words + b.cast(pl.String).to_list())
                    mism("string_cache eq", (a == b.head(len(words)) if len(words) <= n else a.head(n) == b).null_count() >= 0, True)
                pl.enable_string_cache(); s = pl.Series(words, dtype=pl.Categorical); pl.disable_string_cache()
                mism("global string cache", s.cast(pl.String).to_list(), words)
            elif op == "cat_streaming":
                df = pl.DataFrame({"c": pl.Series(words, dtype=pl.Categorical), "v": range(len(words))})
                q = df.lazy().group_by("c").agg(pl.col("v").sum(), pl.len()).sort("c", nulls_last=True)
                got = q.collect(engine="streaming"); exp = q.collect(engine="in-memory")
                mism("cat streaming groupby", got.with_columns(pl.col("c").cast(pl.String)).rows(), exp.with_columns(pl.col("c").cast(pl.String)).rows())
                mism("cat streaming unique", df.lazy().unique("c").collect(engine="streaming").height, len(set(words)))
                mism("cat streaming sort", df.lazy().sort("c", nulls_last=True).collect(engine="streaming")["c"].cast(pl.String).to_list(), sorted([w for w in words if w is not None]) + [None] * words.count(None))
                mism("cat streaming filter", df.lazy().filter(pl.col("c") == "w1").collect(engine="streaming").height, words.count("w1"))
                mism("cat streaming window", df.lazy().select(pl.col("v").sum().over("c")).collect(engine="streaming").height, len(words))
            elif op == "numpy_ufunc":
                s = pl.Series("a", [float(i) for i in range(n)])
                mism("ufunc sqrt", np.sqrt(s).to_list(), [float(np.sqrt(i)) for i in range(n)])
                mism("ufunc add", np.add(s, s).to_list(), [2.0 * i for i in range(n)])
                mism("ufunc int", np.abs(pl.Series([-1, 2, -3])).to_list(), [1, 2, 3])
                arr = s.to_numpy(); mism("to_numpy", arr.sum(), sum(range(n)))
                try:
                    def bad(x): raise RuntimeError("boom")
                    pl.Series("a", [1.0, 2.0]).map_elements(bad, return_dtype=pl.Float64)
                except Exception: pass
            elif op == "struct_cat":
                s = pl.Series("s", [{"c": w, "i": i} for i, w in enumerate(words)], dtype=pl.Struct({"c": pl.Categorical, "i": pl.Int64}))
                mism("struct cat field", s.struct.field("c").cast(pl.String).to_list(), words)
                mism("struct cat unique", s.n_unique(), len(set(zip(words, range(len(words))))))
                mism("struct cat sort", s.sort().struct.field("i").len(), len(words))
                mism("struct cat arrow rt", pl.Series("s", s.to_arrow()).struct.field("c").cast(pl.String).to_list(), words)
                mism("struct cat json", s.struct.json_encode().len(), len(words))
                mism("struct cat hash", s.hash().len(), len(words))
                df = s.to_frame(); mism("struct cat groupby", df.group_by("s").len().height, len(words))
        except pl.exceptions.PanicException as ex:
            log(f"  PANIC: {str(ex)[:300]}")
        except Exception as ex:
            log(f"  exc {type(ex).__name__}: {str(ex)[:200]}")
    log("done")

ts = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
for t in ts: t.start()
for t in ts: t.join()
print("all threads done")
