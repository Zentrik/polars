"""Run each known-bug reproducer in a subprocess; report exit status per case.
Usage: python run_known.py [case ...]
"""
import subprocess, sys, os, signal, textwrap, time

CASES = {}
def case(name, timeout=120):
    def deco(f):
        CASES[name] = (f, timeout); return f
    return deco

# ---- stack overflows (manifest as SIGSEGV) ----
CASES["concat_align_26788"] = (textwrap.dedent('''
    import polars as pl
    dfs = [pl.DataFrame({"realization": [0], f"c{i}": [1.0]}) for i in range(1500)]
    out = pl.concat(dfs, how="align")
    print(out.shape)
'''), 300)

CASES["merge_sorted_chain_26960"] = (textwrap.dedent('''
    import functools, numpy as np, polars as pl
    np.random.seed(0)
    lfs = []
    for i in range(2000):
        ts = np.sort(np.random.randint(0, 86400_000_000_000, 1000, dtype=np.int64))
        lfs.append(pl.LazyFrame({"ts": ts, "key": np.random.choice([f"SYM_{j}" for j in range(50)], 1000), "val": np.random.randn(1000)}))
    merged = functools.reduce(lambda a, b: a.merge_sorted(b, key="ts"), lfs)
    print(merged.sort("ts", "key").collect().shape)
'''), 600)

CASES["struct_sqrt_28563"] = (textwrap.dedent('''
    import polars as pl
    print(pl.Series("a", [{"x": 1}]).sqrt())
'''), 120)
CASES["struct_pct_change_28563"] = (textwrap.dedent('''
    import polars as pl
    print(pl.Series("a", [{"x": 1}]).pct_change())
'''), 120)
CASES["struct_ewm_28563"] = (textwrap.dedent('''
    import polars as pl
    print(pl.Series("a", [{"x": 1}]).ewm_std(alpha=0.5))
'''), 120)
CASES["struct_cbrt_28563"] = (textwrap.dedent('''
    import polars as pl
    print(pl.Series("a", [{"x": 1}]).cbrt())
'''), 120)

CASES["join_where_left_29026"] = (textwrap.dedent('''
    import polars as pl
    n = 100_000
    a = pl.LazyFrame({"id_x": range(n), "val_x": pl.Series(range(n)).shuffle(seed=1)})
    b = pl.LazyFrame({"id_y": range(n), "val_y": pl.Series(range(n)).shuffle(seed=2)})
    c = a.join_where(b, pl.col("id_x") == pl.col("id_y"), pl.col("val_x") <= pl.col("val_y"), how="left")
    print(c.head(10).collect().shape)
'''), 600)

CASES["deep_when_then_15211"] = (textwrap.dedent('''
    import polars as pl
    e = pl.lit(0)
    for i in range(3000):
        e = pl.when(pl.col("a") == i).then(i).otherwise(e)
    print(pl.DataFrame({"a": [1, 2, 3]}).select(e).shape)
'''), 300)

CASES["deep_binary_expr"] = (textwrap.dedent('''
    import polars as pl
    e = pl.col("a")
    for i in range(20000):
        e = e + pl.lit(1)
    print(pl.DataFrame({"a": [1, 2, 3]}).lazy().select(e).collect().shape)
'''), 300)

# ---- Arrow FFI / interop ----
CASES["nested_string_arrow_ffi_28626"] = (textwrap.dedent('''
    import polars as pl
    s = pl.Series("c", [["a"], ["b"]])
    print(pl.Series("c", s.to_arrow()).to_list())
    s = pl.Series("c", [["x" * 13]])
    print(pl.Series("c", s.to_arrow()).to_list())
    s = pl.Series("c", [["x" * 13, "y" * 9, "z"*100] * 50] * 20)
    r = pl.Series("c", s.to_arrow()).to_list()
    assert r == s.to_list(), "DATA MISMATCH"
'''), 120)

CASES["sliced_struct_to_arrow_27450"] = (textwrap.dedent('''
    import polars as pl
    s = pl.Series("s", [{"a": 1}, None, {"a": 2}, {"a": 3}])
    t = s.slice(1).to_arrow()
    print(t)
    import pyarrow as pa
    pa.array(t).validate(full=True)
    print(s.slice(1).to_frame().to_pandas())
'''), 120)

CASES["sliced_array_to_arrow_28602"] = (textwrap.dedent('''
    import polars as pl, pyarrow as pa
    s = pl.Series("a", [[1, 2], [3, 4], [5, 6], [7, 8]], dtype=pl.Array(pl.Int64, 2))
    t = s.slice(2).to_arrow()
    print(t)
    t.validate(full=True)
    assert t.to_pylist() == [[5, 6], [7, 8]], "DATA MISMATCH: " + str(t.to_pylist())
    s2 = pl.Series("b", [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]], dtype=pl.Array(pl.Array(pl.Int64,2),2))
    t2 = s2.slice(1).to_arrow(); t2.validate(full=True)
    assert t2.to_pylist() == [[[5,6],[7,8]],[[9,10],[11,12]]], "NESTED DATA MISMATCH: " + str(t2.to_pylist())
'''), 120)

CASES["sliced_binview_c_interop_28623"] = (textwrap.dedent('''
    import polars as pl, pyarrow as pa
    s = pl.Series("a", ["x" * 20, "y" * 30, "z" * 40, "w" * 50])
    sl = s.slice(2)
    t = sl.to_arrow(compat_level=pl.CompatLevel.newest())
    t.validate(full=True)
    print(t.to_pylist())
    assert t.to_pylist() == ["z" * 40, "w" * 50], "DATA MISMATCH"
    # via C stream interface
    arr = pa.array(sl)
    arr.validate(full=True)
    assert arr.to_pylist() == ["z" * 40, "w" * 50], "C-INTEROP MISMATCH " + str(arr.to_pylist())
'''), 120)

CASES["large_string_view_offsets_27783"] = (textwrap.dedent('''
    import polars as pl, pyarrow as pa
    n = 2_200_000
    arrow_arr = pa.array(["x" * 1024] * n, type=pa.large_string())
    df = pl.DataFrame({"col": pl.Series("col", arrow_arr)})
    arrow_table = df.to_arrow(compat_level=pl.CompatLevel.newest())
    arrow_col = arrow_table.column("col").combine_chunks()
    arrow_col.validate(full=True)
    print("ok")
'''), 600)

# ---- numpy ----
CASES["numpy_ufunc_uaf_28188"] = (textwrap.dedent('''
    import polars as pl, numpy as np, traceback, sys, gc
    s = pl.Series("a", [1.0, 2.0, 3.0] * 1000)
    def bad(x):
        raise RuntimeError("boom")
    uf = np.frompyfunc(bad, 1, 1)
    for _ in range(50):
        try:
            s.__array_ufunc__(uf, "__call__", s)
        except Exception:
            tb = sys.exc_info()[2]
            frames = []
            while tb is not None:
                frames.append(tb.tb_frame); tb = tb.tb_next
            gc.collect()
            for f in frames:
                for k, v in list(f.f_locals.items()):
                    if isinstance(v, np.ndarray):
                        x = v.sum(); v[:] = 7.0
        gc.collect()
    print("ok")
'''), 120)

CASES["unaligned_numpy_28120"] = (textwrap.dedent('''
    import polars as pl, numpy as np
    buf = np.zeros(8 * 100 + 1, dtype=np.uint8)
    arr = buf[1:].view(np.int64)
    s = pl.Series(arr)
    print(s.len())
'''), 120)

# ---- misc panics / hangs ----
CASES["hash_rows_zero_width_26154"] = (textwrap.dedent('''
    import polars as pl
    print(pl.DataFrame().hash_rows())
    print(pl.DataFrame({"a": [1,2]}).select().hash_rows())
'''), 60)

CASES["concat_arr_categorical_26146"] = (textwrap.dedent('''
    import polars as pl
    df = pl.DataFrame({"a": ["x", "y"], "b": ["y", "z"]}, schema={"a": pl.Categorical, "b": pl.Categorical})
    print(df.select(pl.concat_arr("a", "b")))
    df = pl.DataFrame({"a": ["x", "y"], "b": ["y", "z"]}, schema={"a": pl.Enum(["x","y","z"]), "b": pl.Enum(["x","y","z"])})
    print(df.select(pl.concat_arr("a", "b")))
'''), 60)

CASES["shift_head_26099"] = (textwrap.dedent('''
    import polars as pl
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]})
    print(lf.select(pl.col("a").shift(1).head(2)).collect())
    print(lf.select(pl.col("a").shift(-1).head(2)).collect())
    print(lf.select(pl.col("a").head(2).shift(1)).collect())
'''), 60)

CASES["transpose_mixed_list_27038"] = (textwrap.dedent('''
    import polars as pl
    df = pl.DataFrame({"a": [[1, 2]], "b": [3]})
    print(df.transpose())
'''), 60)

CASES["over_empty_order_by_27088"] = (textwrap.dedent('''
    import polars as pl
    df = pl.DataFrame({"a": [1, 2], "g": [1, 1]})
    print(df.select(pl.col("a").cum_sum().over("g", order_by=[])))
'''), 60)

CASES["dt_replace_multichunk_28437"] = (textwrap.dedent('''
    import polars as pl, datetime as dt
    s = pl.concat([pl.Series([dt.date(2020,1,1)]), pl.Series([dt.date(2021,5,5)])], rechunk=False)
    print(s.dt.replace(year=2000))
    s2 = pl.concat([pl.Series([dt.datetime(2020,1,1)]), pl.Series([dt.datetime(2021,5,5)])], rechunk=False)
    print(s2.dt.replace(year=2000, month=pl.Series([1,2])))
'''), 60)

CASES["interpolate_nearest_overflow_27205"] = (textwrap.dedent('''
    import polars as pl
    print(pl.Series([1, None, 2**62, None, -2**62], dtype=pl.Int64).interpolate("nearest"))
    print(pl.Series([-2**63, None, 2**63-1], dtype=pl.Int64).interpolate("nearest"))
    print(pl.Series([0, None, 2**64-1], dtype=pl.UInt64).interpolate("nearest"))
'''), 60)

CASES["enum_struct_slice_26643"] = (textwrap.dedent('''
    import polars as pl
    s = pl.Series("s", [{"a": "x"}, {"a": "y"}, None], dtype=pl.Struct({"a": pl.Enum(["x", "y"])}))
    print(s.slice(1, 2))
    print(s.slice(1, 2).to_list())
    df = pl.DataFrame({"s": s})
    print(df.slice(1).with_columns(pl.col("s").struct.field("a")))
'''), 60)

CASES["groups_negative_slice_26442"] = (textwrap.dedent('''
    import polars as pl
    df = pl.DataFrame({"g": [1, 1, 1, 2, 2], "v": [1, 2, 3, 4, 5]})
    print(df.group_by("g").agg(pl.col("v").slice(-2, 1)))
    print(df.group_by("g").agg(pl.col("v").slice(-10, 3)))
    print(df.group_by("g").agg(pl.col("v").slice(1, -1)))
    print(df.select(pl.col("v").slice(-2, 1).over("g")))
'''), 60)

CASES["ipc_zero_row_compressed_27551"] = (textwrap.dedent('''
    import polars as pl, io
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Boolean), "b": pl.Series([], dtype=pl.Int64), "c": pl.Series([], dtype=pl.String)})
    for comp in ["zstd", "lz4"]:
        buf = io.BytesIO(); df.write_ipc(buf, compression=comp); buf.seek(0)
        print(pl.read_ipc(buf))
        buf.seek(0); print(pl.scan_ipc(buf).collect(engine="streaming"))
        buf.seek(0); print(pl.scan_ipc(buf).slice(0, 1).collect())
'''), 60)

CASES["rolling_positive_offset_26724"] = (textwrap.dedent('''
    import polars as pl
    df = pl.DataFrame({"i": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "v": range(10)})
    for off in ["1i", "2i", "5i", "20i"]:
        for closed in ["left","right","both","none"]:
            r = df.rolling("i", period="3i", offset=off, closed=closed).agg(pl.col("v").sum(), pl.col("v").alias("l"))
    print(r)
    print(df.select(pl.col("v").rolling_sum_by("i", "3i").alias("a")))
    dfd = pl.DataFrame({"d": pl.date_range(pl.date(2020,1,1), pl.date(2020,1,10), eager=True), "v": range(10)})
    for off in ["1d", "3d", "30d"]:
        r = dfd.rolling("d", period="2d", offset=off).agg(pl.col("v").sum(), pl.col("v").alias("l"))
    print(r)
'''), 60)

CASES["streaming_groupby_fallback_26801"] = (textwrap.dedent('''
    import polars as pl
    df = pl.DataFrame({"g": [1, 1, 2, 2, 3], "v": [1.0, 2.0, 3.0, 4.0, 5.0], "s": ["a", "b", "c", "d", "e"]})
    q = df.lazy().group_by("g").agg(pl.col("v").map_elements(lambda s: s.sum(), return_dtype=pl.Float64), pl.col("s").str.join("-"), pl.col("v").quantile(0.5), pl.col("v").mode(), pl.col("v").skew(), pl.col("v").entropy(), pl.col("s").unique(), pl.col("v").rank())
    print(q.collect(engine="streaming"))
'''), 120)

CASES["list_agg_nulls_26868"] = (textwrap.dedent('''
    import polars as pl
    s = pl.Series("a", [[1, 2], None, [3], []])
    df = s.to_frame()
    print(df.select(pl.col("a").list.eval(pl.element().sum())))
    print(df.select(pl.col("a").list.eval(pl.lit(1))))
    if hasattr(pl.Expr.list, "agg"):
        print(df.select(pl.col("a").list.agg(pl.element().sum())))
        print(df.select(pl.col("a").list.agg(pl.element().first())))
'''), 60)

CASES["struct_rechunk_27446"] = (textwrap.dedent('''
    import polars as pl
    a = pl.Series("s", [{"x": 1, "y": "a"}, None])
    b = pl.Series("s", [{"x": 2, "y": "b"}, {"x": 3, "y": None}, None])
    s = pl.concat([a, b], rechunk=False)
    r = s.rechunk()
    print(r.to_list())
    assert r.to_list() == s.to_list(), "MISMATCH"
    print(r.is_null().to_list(), s.is_null().to_list())
    assert r.is_null().to_list() == s.is_null().to_list(), "VALIDITY MISMATCH"
    df = pl.DataFrame({"s": s})
    print(df.rechunk().select(pl.col("s").struct.field("x")))
'''), 60)

CASES["temporal_extract_nulls_28054"] = (textwrap.dedent('''
    import polars as pl, datetime as dt
    s = pl.Series([dt.datetime(2020,1,1), None, dt.datetime(2021,1,1)])
    print(s.dt.year(), s.dt.month(), s.dt.day(), s.dt.hour(), s.dt.ordinal_day(), s.dt.weekday(), s.dt.quarter())
    s2 = pl.Series([dt.date(2020,1,1), None]).dt.replace_time_zone
    s3 = pl.Series([dt.datetime(2020,1,1), None]).dt.replace_time_zone("Asia/Tokyo")
    print(s3.dt.year(), s3.dt.hour(), s3.dt.dst_offset())
'''), 60)

CASES["decimal_sum_overflow_28688"] = (textwrap.dedent('''
    import polars as pl
    s = pl.Series([10**37, 10**37, 10**37, 10**37, 10**37, 10**37, 10**37, 10**37, 10**37, 10**37, 10**37], dtype=pl.Int128).cast(pl.Decimal(38, 0))
    print(s.sum())
    print(s.to_frame().group_by(pl.lit(1)).agg(pl.col(s.name).sum()))
'''), 60)

CASES["parquet_dict_rle_nullable_26411"] = (textwrap.dedent('''
    import polars as pl, pyarrow as pa, pyarrow.parquet as pq, io, random
    random.seed(0)
    n = 100_000
    vals = [random.choice(["a", "b", "c", None]) for _ in range(n)]
    ints = [random.choice([1, 2, 3, None]) for _ in range(n)]
    t = pa.table({"s": pa.array(vals, pa.string()).dictionary_encode(), "i": pa.array(ints, pa.int32()).dictionary_encode(), "plain": vals})
    buf = io.BytesIO()
    pq.write_table(t, buf, use_dictionary=True, data_page_size=1024, write_statistics=True, compression="snappy", version="2.6", data_page_version="2.0")
    buf.seek(0)
    df = pl.read_parquet(buf)
    assert df["s"].to_list() == vals, "STRING MISMATCH"
    assert df["i"].to_list() == ints, "INT MISMATCH"
    buf.seek(0)
    df2 = pl.scan_parquet(buf).filter(pl.col("plain") == "a").collect()
    assert df2["s"].to_list() == [v for v in vals if v == "a"], "FILTERED MISMATCH"
    print("ok", df.shape, df2.shape)
'''), 300)

CASES["parquet_optional_mask_28547"] = (textwrap.dedent('''
    import polars as pl, pyarrow as pa, pyarrow.parquet as pq, io, random
    random.seed(1)
    n = 200_000
    for dtype in [pa.int64(), pa.float32(), pa.string(), pa.bool_()]:
        for _ in range(3):
            vals = [None if random.random() < random.choice([0.01, 0.5, 0.99]) else (random.randint(0, 5) if dtype != pa.string() else random.choice(["aa", "bbbb", "c" * 20])) for _ in range(n)]
            if dtype == pa.bool_(): vals = [None if v is None else bool(v % 2) for v in vals]
            if dtype == pa.float32(): vals = [None if v is None else float(v) for v in vals]
            t = pa.table({"x": pa.array(vals, dtype), "f": list(range(n))})
            for kw in [dict(use_dictionary=False), dict(use_dictionary=True), dict(use_dictionary=False, data_page_version="2.0"), dict(use_dictionary=False, use_byte_stream_split=True) if dtype in (pa.int64(), pa.float32()) else dict(use_dictionary=True, data_page_version="2.0")]:
                buf = io.BytesIO(); pq.write_table(t, buf, data_page_size=4096, **kw); buf.seek(0)
                df = pl.read_parquet(buf)
                assert df["x"].to_list() == vals, f"MISMATCH full {dtype} {kw}"
                buf.seek(0)
                df = pl.scan_parquet(buf).filter(pl.col("f") % 7 == 3).collect()
                assert df["x"].to_list() == vals[3::7], f"MISMATCH filtered {dtype} {kw}"
                buf.seek(0)
                df = pl.scan_parquet(buf).slice(12345, 5000).collect()
                assert df["x"].to_list() == vals[12345:12345+5000], f"MISMATCH sliced {dtype} {kw}"
    print("ok")
'''), 600)

CASES["parquet_fastparquet_style_28656"] = (textwrap.dedent('''
    import polars as pl, pyarrow as pa, pyarrow.parquet as pq, io
    n = 50_000
    t = pa.table({"x": pa.array([None if i % 3 == 0 else i for i in range(n)], pa.int64()), "y": pa.array([None if i % 5 == 0 else "s%d" % i for i in range(n)])})
    for v in ["1.0", "2.4", "2.6"]:
        for enc in [dict(use_dictionary=False), dict(use_dictionary=True)]:
            buf = io.BytesIO(); pq.write_table(t, buf, version=v, write_statistics=False, data_page_size=512, **enc); buf.seek(0)
            df = pl.read_parquet(buf)
            assert df["x"].to_list() == t["x"].to_pylist(), f"X MISMATCH {v} {enc}"
            assert df["y"].to_list() == t["y"].to_pylist(), f"Y MISMATCH {v} {enc}"
    print("ok")
'''), 300)

CASES["large_list_arrow_28632"] = (textwrap.dedent('''
    import polars as pl, pyarrow as pa
    arr = pa.array([[["a", "bb"], None, ["ccc" * 5]], None, [[]]], type=pa.large_list(pa.large_list(pa.large_string())))
    s = pl.Series("x", arr)
    print(s.to_list())
    assert s.to_list() == arr.to_pylist(), "MISMATCH " + str(s.to_list())
    arr2 = pa.array([["x" * 13, "y" * 9], ["z"]], type=pa.large_list(pa.large_string()))
    s2 = pl.Series("y", arr2)
    assert s2.to_list() == arr2.to_pylist(), "MISMATCH2 " + str(s2.to_list())
    arr3 = pa.array([[b"x" * 13, b"y" * 9], [b"z"]], type=pa.large_list(pa.large_binary()))
    s3 = pl.Series("z", arr3)
    assert s3.to_list() == arr3.to_pylist(), "MISMATCH3 " + str(s3.to_list())
    print("ok")
'''), 60)

def run(name, code, timeout, py):
    t0 = time.time()
    try:
        p = subprocess.run([py, "-W", "ignore", "-c", code], capture_output=True, text=True, timeout=timeout)
        rc = p.returncode
        if rc < 0:
            status = f"SIGNAL {signal.Signals(-rc).name}"
        elif rc == 0:
            status = "ok"
        else:
            last = [l for l in p.stderr.strip().splitlines() if l.strip()]
            status = f"exit {rc}: {last[-1][:160] if last else ''}"
    except subprocess.TimeoutExpired:
        status = f"TIMEOUT>{timeout}s"
    return status, time.time() - t0

if __name__ == "__main__":
    py = sys.executable
    names = sys.argv[1:] or list(CASES)
    for name in names:
        code, timeout = CASES[name]
        status, dt = run(name, code, timeout, py)
        print(f"{name:40s} {status}  ({dt:.1f}s)", flush=True)
