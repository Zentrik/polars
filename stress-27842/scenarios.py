#!/usr/bin/env python3
"""Workload library for stress-testing the polars streaming engine / async executor.

Each scenario is executed in a fresh subprocess by harness.py (one scenario per
process) so that crashes are isolated and env vars like POLARS_MAX_THREADS are
picked up at import time. Scenarios are seeded and print progress lines so a
crash can be located from captured stdout.

Targeted at pola-rs/polars#27842 (deque capacity underflow in the async
executor's work stealing, observed as segfaults on wrapping builds).
"""

import argparse
import faulthandler
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

N_SMALL_FILES = 96
SMALL_FILE_ROWS = 50_000
N_BIG_FILES = 8
BIG_FILE_ROWS = 1_000_000


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def rows_for_config():
    """Scale row counts to the configured morsel size so each query produces a
    bounded number of morsels (~120k at tiny morsels) instead of taking forever."""
    morsel = env_int("POLARS_IDEAL_MORSEL_SIZE", 100_000)
    mult = float(os.environ.get("STRESS_ROWS_MULT", "1.0"))
    # ~20k morsels per query: tiny-morsel configs are scheduling-bound at
    # only a few hundred rows/sec, so more would just hit the timeout.
    base = 20_000 * max(morsel, 1)
    return int(min(8_000_000, max(10_000, base)) * mult)


def make_lf(pl, n, seed=0):
    """Deterministic pseudo-random LazyFrame with int/float/string columns and
    several key cardinalities, generated without numpy."""
    lf = pl.select(pl.int_range(0, n, dtype=pl.Int64).alias("id")).lazy()
    return lf.with_columns(
        ((pl.col("id") * 2654435761 + 97 * seed) % 2147483647).alias("h")
    ).with_columns(
        (pl.col("h") % max(n // 4, 1)).alias("k_hi"),
        (pl.col("h") % 97).alias("k_lo"),
        (pl.col("h") % 10007).alias("k_mid"),
        ((pl.col("h") % 1000003).cast(pl.Float64) / 997.0).alias("v"),
        (pl.col("h") % 9973).cast(pl.String).alias("s"),
    )


def data_paths():
    d = Path(os.environ["STRESS_DATA_DIR"])
    return SimpleNamespace(
        many_glob=str(d / "many_small" / "*.parquet"),
        many_files=sorted((d / "many_small").glob("*.parquet")),
        big_glob=str(d / "big" / "*.parquet"),
    )


def gen_data(pl, rng, ctx):
    d = Path(os.environ["STRESS_DATA_DIR"])
    marker = d / ".complete"
    if marker.exists():
        print("[gen_data] already present")
        return
    (d / "many_small").mkdir(parents=True, exist_ok=True)
    (d / "big").mkdir(parents=True, exist_ok=True)
    for i in range(N_SMALL_FILES):
        make_lf(pl, SMALL_FILE_ROWS, seed=i).collect().write_parquet(
            d / "many_small" / f"f{i:03d}.parquet", row_group_size=16_384
        )
    print(f"[gen_data] wrote {N_SMALL_FILES} small files", flush=True)
    for i in range(N_BIG_FILES):
        make_lf(pl, BIG_FILE_ROWS, seed=1000 + i).collect().write_parquet(
            d / "big" / f"b{i}.parquet", row_group_size=65_536
        )
    print(f"[gen_data] wrote {N_BIG_FILES} big files", flush=True)
    marker.touch()


def collect_streaming(lf, pl):
    return lf.collect(engine="streaming")


def bounded_big_scan(pl, ctx):
    """Scan of the big dataset, bounded to ctx.rows so tiny-morsel configs
    don't turn every scan into millions of morsels."""
    lf = pl.scan_parquet(ctx.paths.big_glob)
    if ctx.rows < N_BIG_FILES * BIG_FILE_ROWS:
        lf = lf.head(ctx.rows)
    return lf


def checksum(df):
    return (df.height, df.width)


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------

def scen_seq_pipeline(pl, rng, ctx):
    """Sequential big internally-parallel queries: the production shape."""
    for q in range(4):
        shape = rng.randrange(5)
        if shape == 0:
            lf = (
                bounded_big_scan(pl, ctx)
                .filter(pl.col("v") > 100.0)
                .with_columns((pl.col("v") * 3.5 - pl.col("k_lo")).alias("w"))
                .group_by(pl.col("k_hi") % 100_000)
                .agg(
                    pl.sum("v"), pl.mean("w"), pl.min("s"),
                    pl.max("id"), pl.n_unique("k_lo"),
                )
                .sort("v", descending=True)
                .head(1_000)
            )
        elif shape == 1:
            dim = (
                pl.scan_parquet(ctx.paths.many_glob)
                .group_by("k_lo")
                .agg(pl.mean("v").alias("dim_v"), pl.len().alias("dim_n"))
            )
            lf = (
                bounded_big_scan(pl, ctx)
                .join(dim, on="k_lo", how="left")
                .group_by("k_mid")
                .agg(pl.sum("dim_v"), pl.max("dim_n"), pl.mean("v"))
            )
        elif shape == 2:
            lf = (
                make_lf(pl, ctx.rows, seed=rng.randrange(1000))
                .with_columns(
                    pl.col("v").sum().over("k_lo").alias("v_grp"),
                    pl.col("v").rank().over(pl.col("k_hi") % 1000).alias("r"),
                    pl.col("v").cum_sum().alias("cs"),
                )
                .filter(pl.col("r") < 5)
            )
        elif shape == 3:
            lf = bounded_big_scan(pl, ctx).sort("v").head(50)
        else:
            qs = [
                make_lf(pl, max(ctx.rows // 4, 10_000), seed=rng.randrange(1000))
                .group_by("k_lo")
                .agg(pl.sum("v"), pl.n_unique("s"))
                for _ in range(4)
            ]
            try:
                res = pl.collect_all(qs, engine="streaming")
            except TypeError:
                res = pl.collect_all(qs)
            print(f"[seq] q{q} collect_all -> {[checksum(r) for r in res]}", flush=True)
            continue
        df = collect_streaming(lf, pl)
        print(f"[seq] q{q} shape{shape} -> {checksum(df)}", flush=True)


def scen_groupby_storm(pl, rng, ctx):
    lf = make_lf(pl, ctx.rows, seed=rng.randrange(1000))
    for i in range(3):
        hi = (
            lf.group_by("k_hi")
            .agg(pl.sum("v"), pl.mean("v").alias("m"), pl.min("s"), pl.max("id"))
            .sort("v", descending=True)
            .head(100)
        )
        lo = lf.group_by("k_lo").agg(
            pl.sum("v"), pl.n_unique("k_hi"), pl.n_unique("s"), pl.len()
        )
        mid = lf.group_by("k_mid", "k_lo").agg(pl.mean("v"), pl.max("h"))
        for name, q in (("hi", hi), ("lo", lo), ("mid", mid)):
            df = collect_streaming(q, pl)
            print(f"[gb] r{i} {name} -> {checksum(df)}", flush=True)


def scen_join_storm(pl, rng, ctx):
    n = ctx.rows
    a = make_lf(pl, n, seed=rng.randrange(1000))
    b = make_lf(pl, max(n // 2, 10_000), seed=rng.randrange(1000) + 7)
    joins = [
        ("inner_hi", a.join(b, on="k_hi", how="inner")),
        ("left_lo", a.join(b.group_by("k_lo").agg(pl.mean("v").alias("bv")), on="k_lo", how="left")),
        ("inner_str", a.join(b.unique(subset="s"), on="s", how="inner")),
    ]
    for name, j in joins:
        df = collect_streaming(
            j.group_by("k_lo").agg(pl.len(), pl.sum("v")), pl
        )
        print(f"[join] {name} -> {checksum(df)}", flush=True)


def scen_multiscan_sink(pl, rng, ctx):
    out = Path.cwd() / "sink_out.parquet"
    lf = (
        pl.scan_parquet(ctx.paths.many_glob)
        .filter(pl.col("k_lo") < 90)
        .with_columns((pl.col("v") * 2.0).alias("v2"))
    )
    lf.sink_parquet(out)
    print("[sink] sink_parquet done", flush=True)
    df = collect_streaming(
        pl.scan_parquet(out).group_by("k_lo").agg(pl.sum("v2"), pl.len()), pl
    )
    print(f"[sink] rescan -> {checksum(df)}", flush=True)
    df2 = collect_streaming(
        pl.scan_parquet([str(p) for p in ctx.paths.many_files])
        .group_by("k_mid")
        .agg(pl.mean("v"), pl.n_unique("s")),
        pl,
    )
    print(f"[sink] explicit-list scan -> {checksum(df2)}", flush=True)


def scen_sort_window(pl, rng, ctx):
    lf = make_lf(pl, ctx.rows, seed=rng.randrange(1000))
    df = collect_streaming(lf.sort("k_lo", "v", descending=[False, True]), pl)
    print(f"[sortw] full sort -> {checksum(df)}", flush=True)
    df = collect_streaming(
        lf.with_columns(
            pl.col("v").sum().over("k_lo").alias("a"),
            pl.col("v").rank().over(pl.col("k_hi") % 500).alias("b"),
            pl.col("id").cum_sum().over("k_lo").alias("c"),
        ).filter(pl.col("b") <= 3),
        pl,
    )
    print(f"[sortw] window -> {checksum(df)}", flush=True)
    df = collect_streaming(lf.top_k(100, by="v"), pl)
    print(f"[sortw] top_k -> {checksum(df)}", flush=True)


def scen_concat_many(pl, rng, ctx):
    piece = max(ctx.rows // 128, 2_000)
    parts = [make_lf(pl, piece, seed=i) for i in range(128)]
    df = collect_streaming(
        pl.concat(parts).group_by("k_lo").agg(pl.sum("v"), pl.len()), pl
    )
    print(f"[concat] in-mem 128 -> {checksum(df)}", flush=True)
    scans = [pl.scan_parquet(str(p)) for p in ctx.paths.many_files]
    df = collect_streaming(
        pl.concat(scans).group_by("k_mid").agg(pl.mean("v")), pl
    )
    print(f"[concat] scans {len(scans)} -> {checksum(df)}", flush=True)


def scen_rapid_tiny(pl, rng, ctx):
    """Many tiny queries: per-query task_scope create/teardown churn."""
    for i in range(300):
        n = 1_000 + rng.randrange(3_000)
        lf = make_lf(pl, n, seed=i)
        shape = i % 4
        if shape == 0:
            q = lf.group_by("k_lo").agg(pl.sum("v"))
        elif shape == 1:
            q = lf.join(lf.select("k_lo", "v").unique(subset="k_lo"), on="k_lo")
        elif shape == 2:
            q = lf.sort("v").head(7)
        else:
            q = lf.select(pl.col("v").sum().over("k_lo"))
        collect_streaming(q, pl)
        if shape == 3:
            f = ctx.paths.many_files[i % len(ctx.paths.many_files)]
            collect_streaming(pl.scan_parquet(str(f)).head(10), pl)
        if i % 50 == 0:
            print(f"[tiny] {i}", flush=True)


def scen_early_stop(pl, rng, ctx):
    """head/limit/slice on hot pipelines: cancels producer tasks mid-steal."""
    for i in range(25):
        mode = i % 4
        base = pl.scan_parquet(ctx.paths.big_glob).with_columns(
            (pl.col("v") * 1.5).alias("w")
        )
        if mode == 0:
            q = base.filter(pl.col("k_lo") == 42).head(3)
        elif mode == 1:
            q = base.head(1)
        elif mode == 2:
            q = base.slice(1_000_000 + rng.randrange(1_000_000), 5)
        else:
            q = base.group_by("k_lo").agg(pl.sum("v")).head(5)
        df = collect_streaming(q, pl)
        if i % 5 == 0:
            print(f"[estop] {i} -> {checksum(df)}", flush=True)


def scen_cancel_async(pl, rng, ctx):
    """Cancel in-flight collects: scope-cancellation racing hot stealing."""
    import asyncio

    heavy = (
        bounded_big_scan(pl, ctx)
        .group_by("k_hi")
        .agg(pl.sum("v"), pl.n_unique("s"))
    )

    async def one():
        try:
            return await heavy.collect_async(engine="streaming")
        except TypeError:
            return await heavy.collect_async()

    async def main():
        for i in range(12):
            t = asyncio.ensure_future(one())
            await asyncio.sleep(rng.uniform(0.02, 0.4))
            t.cancel()
            try:
                await t
            except BaseException as e:
                print(f"[cancel] {i} -> {type(e).__name__}", flush=True)

    asyncio.run(main())

    for i in range(6):
        try:
            q = (
                make_lf(pl, max(ctx.rows // 4, 50_000), seed=i)
                .group_by("k_hi")
                .agg(pl.sum("v"))
                .collect(background=True)
            )
            time.sleep(rng.uniform(0.02, 0.3))
            q.cancel()
            print(f"[cancel] bg {i} cancelled", flush=True)
        except BaseException as e:
            print(f"[cancel] background collect not usable: {e!r}", flush=True)
            break


def scen_interrupt_victim(pl, rng, ctx):
    """Runs queries while the harness fires SIGINT at random moments
    (KeyboardInterrupt-driven query cancellation)."""
    attempts = 0
    done = 0
    while attempts < 40:
        attempts += 1
        try:
            lf = make_lf(pl, max(ctx.rows // 4, 50_000), seed=attempts)
            mode = attempts % 3
            if mode == 0:
                q = lf.group_by("k_hi").agg(pl.sum("v"), pl.n_unique("s"))
            elif mode == 1:
                q = lf.join(lf.select("k_hi", "v").unique(subset="k_hi"), on="k_hi")
            else:
                q = bounded_big_scan(pl, ctx).group_by("k_lo").agg(pl.mean("v"))
            collect_streaming(q, pl)
            done += 1
            if done % 5 == 0:
                print(f"[interrupt] {done}/{attempts} ok", flush=True)
        except KeyboardInterrupt:
            print(f"[interrupt] KI at attempt {attempts}", flush=True)
        except Exception as e:
            print(f"[interrupt] {type(e).__name__} at attempt {attempts}", flush=True)
    print(f"[interrupt] finished {done}/{attempts}", flush=True)


def scen_threads_concurrent(pl, rng, ctx):
    from concurrent.futures import ThreadPoolExecutor

    n = max(ctx.rows // 8, 50_000)
    errors = []

    def work(i):
        try:
            lf = make_lf(pl, n, seed=i)
            mode = i % 4
            if mode == 0:
                q = lf.group_by("k_hi").agg(pl.sum("v"))
            elif mode == 1:
                q = lf.join(make_lf(pl, n // 2, seed=i + 1), on="k_lo", how="inner").head(10_000)
            elif mode == 2:
                q = pl.scan_parquet(ctx.paths.many_glob).head(n).group_by("k_lo").agg(pl.len())
            else:
                q = lf.sort("v").head(5)
            return checksum(collect_streaming(q, pl))
        except BaseException as e:
            errors.append(repr(e))
            return None

    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(work, range(32)))
    print(f"[thr] done, {sum(r is not None for r in res)}/32 ok, errors={errors[:3]}", flush=True)


def scen_mixed_engines(pl, rng, ctx):
    """Streaming + in-memory engine at once (scans use streaming multiscan on
    both), plus tiny-query churn on the main thread."""
    from concurrent.futures import ThreadPoolExecutor

    n = max(ctx.rows // 8, 50_000)
    errors = []

    def work(i):
        try:
            engine = "streaming" if i % 2 == 0 else "in-memory"
            if i % 3 == 0:
                q = pl.scan_parquet(ctx.paths.many_glob).head(n).group_by("k_lo").agg(pl.sum("v"))
            else:
                q = make_lf(pl, n, seed=i).group_by("k_mid").agg(pl.mean("v"))
            return checksum(q.collect(engine=engine))
        except BaseException as e:
            errors.append(repr(e))
            return None

    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = ex.map(work, range(24))
        for i in range(100):
            collect_streaming(make_lf(pl, 2_000, seed=i).group_by("k_lo").agg(pl.sum("v")), pl)
        res = list(fut)
    print(f"[mix] done, {sum(r is not None for r in res)}/24 ok, errors={errors[:3]}", flush=True)


# name -> (func, weight, base_timeout_s, inject_sigint)
REGISTRY = {
    "gen_data": (gen_data, 0.0, 600, False),
    "seq_pipeline": (scen_seq_pipeline, 3.0, 240, False),
    "groupby_storm": (scen_groupby_storm, 2.0, 200, False),
    "join_storm": (scen_join_storm, 2.0, 220, False),
    "multiscan_sink": (scen_multiscan_sink, 2.0, 220, False),
    "sort_window": (scen_sort_window, 1.5, 220, False),
    "concat_many": (scen_concat_many, 1.2, 220, False),
    "rapid_tiny": (scen_rapid_tiny, 1.5, 220, False),
    "early_stop": (scen_early_stop, 2.0, 200, False),
    "cancel_async": (scen_cancel_async, 1.2, 220, False),
    "interrupt_victim": (scen_interrupt_victim, 1.5, 320, True),
    "threads_concurrent": (scen_threads_concurrent, 1.0, 260, False),
    "mixed_engines": (scen_mixed_engines, 1.0, 260, False),
}


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=sorted(REGISTRY))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import polars as pl  # after env vars are set by the harness

    rng = random.Random(args.seed)
    ctx = SimpleNamespace(rows=rows_for_config(), paths=None)
    if args.scenario != "gen_data":
        ctx.paths = data_paths()
    t0 = time.time()
    print(
        f"[start] {args.scenario} seed={args.seed} rows={ctx.rows} "
        f"polars={pl.__version__} threads={pl.thread_pool_size()}",
        flush=True,
    )
    REGISTRY[args.scenario][0](pl, rng, ctx)
    print(f"[done] {args.scenario} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
