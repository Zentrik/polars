"""String/binary view fuzzer: build string columns with many chunks/slices/gathers, run view-heavy
kernels, and verify results against a pure-Python reference. Crashes or mismatches indicate bugs.
Usage: python fuzz_binview.py SEED NITER LOGFILE
"""
import sys, random, warnings
warnings.simplefilter("ignore")
import polars as pl
seed = int(sys.argv[1]); niter = int(sys.argv[2]); logf = open(sys.argv[3], "a", buffering=1)
rng = random.Random(seed)
def log(m): logf.write(f"[seed={seed}] {m}\n")
ALPH = "abcXYZ_ é中😀"
def rstr():
    n = rng.choice([0, 1, 4, 11, 12, 13, 14, 16, 17, 31, 32, 33, 100, 1000, 5000])
    return "".join(rng.choice(ALPH) for _ in range(n))

def build(n):
    """Return (series, python list) with weird chunk layouts."""
    nullp = rng.choice([0.0, 0.1, 0.9])
    vals = [None if rng.random() < nullp else rstr() for _ in range(n)]
    s = pl.Series("s", vals, pl.String)
    mode = rng.choice(["plain", "chunked", "sliced", "gathered", "concat_sliced", "filtered", "from_arrow_large", "from_arrow_view", "cast_binary_back"])
    if mode == "chunked" and n > 1:
        cuts = sorted(rng.sample(range(1, n), min(n - 1, rng.randint(1, 6))))
        parts, prev = [], 0
        for c in cuts + [n]: parts.append(s.slice(prev, c - prev)); prev = c
        s = pl.concat(parts, rechunk=False)
    elif mode == "sliced" and n > 2:
        pad = [rstr() for _ in range(rng.randint(1, 5))]
        s = pl.concat([pl.Series("s", pad), s, pl.Series("s", pad)], rechunk=rng.random() < 0.5).slice(len(pad), n)
    elif mode == "gathered" and n > 0:
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        s = s.gather(idx); vals = [vals[i] for i in idx]
    elif mode == "concat_sliced" and n > 4:
        a = s.slice(0, n // 2); b = s.slice(n // 2)
        s = pl.concat([a.slice(1), b.slice(0, max(0, b.len() - 1))], rechunk=False)
        vals = vals[1:n // 2] + vals[n // 2:n - 1]
    elif mode == "filtered":
        mask = [rng.random() < 0.7 for _ in range(n)]
        s = s.filter(pl.Series(mask)); vals = [v for v, m in zip(vals, mask) if m]
    elif mode == "from_arrow_large":
        import pyarrow as pa
        s = pl.Series("s", pa.array(vals, pa.large_string()))
    elif mode == "from_arrow_view":
        import pyarrow as pa
        s = pl.Series("s", pa.array(vals, pa.string_view()))
    elif mode == "cast_binary_back":
        s = s.cast(pl.Binary).cast(pl.String)
    return s, vals

def check(name, got, exp):
    if got != exp:
        log(f"  !!! MISMATCH {name}: got={str(got)[:150]} exp={str(exp)[:150]}")
        return False
    return True

for it in range(niter):
    n = rng.choice([0, 1, 2, 5, 33, 100, 1000, 3000])
    try:
        s, vals = build(n)
        log(f"iter {it} n={len(vals)} chunks={s.n_chunks()} mode-check")
        df = s.to_frame()
        ops = rng.sample([
            ("to_list", lambda: s.to_list(), lambda: vals),
            ("len_bytes", lambda: s.str.len_bytes().to_list(), lambda: [None if v is None else len(v.encode()) for v in vals]),
            ("len_chars", lambda: s.str.len_chars().to_list(), lambda: [None if v is None else len(v) for v in vals]),
            ("upper", lambda: s.str.to_uppercase().to_list(), lambda: [None if v is None else v.upper() for v in vals]),
            ("concat_str", lambda: df.select(pl.concat_str([pl.col("s"), pl.col("s")], separator="-")).to_series().to_list(), lambda: [None if v is None else v + "-" + v for v in vals]),
            ("str_join", lambda: [s.str.join("|").item()], lambda: ["|".join(v for v in vals if v is not None)]),
            ("str_join_nulls", lambda: [s.str.join("|", ignore_nulls=False).item()], lambda: [None if any(v is None for v in vals) else "|".join(vals)]),
            ("slice", lambda: s.str.slice(1, 3).to_list(), lambda: [None if v is None else v[1:4] for v in vals]),
            ("head", lambda: s.str.head(2).to_list(), lambda: [None if v is None else v[:2] for v in vals]),
            ("tail", lambda: s.str.tail(2).to_list(), lambda: [None if v is None else v[-2:] if v else "" for v in vals]),
            ("reverse", lambda: s.str.reverse().to_list(), lambda: [None if v is None else v[::-1] for v in vals]),
            ("replace_all", lambda: s.str.replace_all("a", "ZZ", literal=True).to_list(), lambda: [None if v is None else v.replace("a", "ZZ") for v in vals]),
            ("replace_first", lambda: s.str.replace("a", "", literal=True).to_list(), lambda: [None if v is None else v.replace("a", "", 1) for v in vals]),
            ("contains", lambda: s.str.contains("é", literal=True).to_list(), lambda: [None if v is None else ("é" in v) for v in vals]),
            ("starts_with", lambda: s.str.starts_with("ab").to_list(), lambda: [None if v is None else v.startswith("ab") for v in vals]),
            ("strip", lambda: s.str.strip_chars(" ").to_list(), lambda: [None if v is None else v.strip(" ") for v in vals]),
            ("split", lambda: s.str.split(" ").to_list(), lambda: [None if v is None else v.split(" ") for v in vals]),
            ("sort", lambda: s.sort(nulls_last=True).to_list(), lambda: sorted([v for v in vals if v is not None]) + [None] * sum(v is None for v in vals)),
            ("unique_sorted", lambda: s.unique().sort(nulls_last=True).to_list(), lambda: sorted(set(v for v in vals if v is not None)) + ([None] if any(v is None for v in vals) else [])),
            ("n_unique", lambda: [s.n_unique()], lambda: [len(set(vals))]),
            ("hash_consistency", lambda: s.hash().to_list(), lambda: pl.Series("s", vals, pl.String).hash().to_list()),
            ("eq_self", lambda: (s == pl.Series("s", vals, pl.String)).to_list(), lambda: [None if v is None else True for v in vals]),
            ("explode_implode", lambda: s.implode().explode().to_list() if len(vals) else [], lambda: vals),
            ("rechunk", lambda: s.rechunk().to_list(), lambda: vals),
            ("cast_binary", lambda: s.cast(pl.Binary).to_list(), lambda: [None if v is None else v.encode() for v in vals]),
            ("cast_cat", lambda: s.cast(pl.Categorical).cast(pl.String).to_list(), lambda: vals),
            ("to_arrow_roundtrip", lambda: pl.Series("s", s.to_arrow()).to_list(), lambda: vals),
            ("to_arrow_newest_roundtrip", lambda: pl.Series("s", s.to_arrow(compat_level=pl.CompatLevel.newest())).to_list(), lambda: vals),
            ("parquet_roundtrip", lambda: (lambda b: (df.write_parquet(b), b.seek(0), pl.read_parquet(b).to_series().to_list())[2])(__import__("io").BytesIO()), lambda: vals),
            ("ipc_roundtrip", lambda: (lambda b: (df.write_ipc(b), b.seek(0), pl.read_ipc(b).to_series().to_list())[2])(__import__("io").BytesIO()), lambda: vals),
            ("group_by_len", lambda: sorted(df.group_by("s").len().rows(), key=lambda kv: (kv[0] is None, kv[0] or "")), lambda: sorted([(k, vals.count(k)) for k in set(vals)], key=lambda kv: (kv[0] is None, kv[0] or ""))),
            ("join_self", lambda: df.with_row_index().join(df.with_row_index(), on="s", how="inner", nulls_equal=False).height, lambda: sum(vals.count(v) for v in vals if v is not None)),
            ("filter_len_gt", lambda: s.filter(s.str.len_bytes() > 12).to_list(), lambda: [v for v in vals if v is not None and len(v.encode()) > 12]),
            ("gather_every", lambda: s.gather_every(2).to_list(), lambda: vals[::2]),
            ("shift", lambda: s.shift(1).to_list(), lambda: [None] + vals[:-1] if vals else []),
            ("fill_null", lambda: s.fill_null("X").to_list(), lambda: ["X" if v is None else v for v in vals]),
            ("pad_start", lambda: s.str.pad_start(15, "*").to_list(), lambda: [None if v is None else v.rjust(15, "*") for v in vals]),
            ("streaming_select", lambda: df.lazy().select(pl.col("s").str.to_lowercase().alias("s"), pl.col("s").str.len_bytes().alias("l")).collect(engine="streaming").to_series(0).to_list(), lambda: [None if v is None else v.lower() for v in vals]),
            ("streaming_sort", lambda: df.lazy().sort("s", nulls_last=True).collect(engine="streaming").to_series().to_list(), lambda: sorted([v for v in vals if v is not None]) + [None] * sum(v is None for v in vals)),
            ("streaming_unique", lambda: sorted(df.lazy().unique().collect(engine="streaming").to_series().to_list(), key=lambda v: (v is None, v or "")), lambda: sorted(set(vals), key=lambda v: (v is None, v or ""))),
            ("repr", lambda: [len(str(s)) > 0], lambda: [True]),
            ("min_max", lambda: [s.min(), s.max()], lambda: [min((v for v in vals if v is not None), default=None), max((v for v in vals if v is not None), default=None)]),
        ], 12)
        for name, f, g in ops:
            log(f"  op {name}")
            try:
                got = f()
            except pl.exceptions.PanicException as ex:
                log(f"  PANIC in {name}: {str(ex)[:300]}"); continue
            except Exception as ex:
                log(f"  exc in {name}: {type(ex).__name__}: {str(ex)[:120]}"); continue
            try:
                exp = g()
            except Exception as ex:
                log(f"  ref exc in {name}: {ex}"); continue
            check(name, got, exp)
    except pl.exceptions.PanicException as ex:
        log(f"  PANIC: {str(ex)[:300]}")
    except Exception as ex:
        log(f"  exc {type(ex).__name__}: {str(ex)[:200]}")
log("done")
