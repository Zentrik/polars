#!/usr/bin/env python3
"""Crash-hunting harness for pola-rs/polars#27842.

Runs scenarios from scenarios.py, each in a fresh subprocess with a sampled
environment config (executor thread count, morsel size, channel buffer sizes,
allocator hardening). Classifies every exit (ok / segfault / rust panic /
abort / hang / oom-kill), captures core dumps + gdb backtraces, and appends
one JSON line per iteration to results.jsonl.

Usage:
  python3 harness.py --python /path/venv/bin/python --out RUNDIR --data DATADIR \
      [--minutes 90 | --iters N] [--seed 1] [--mult 1.0] [--pin SPEC ...]

  --pin "scenario=early_stop,threads=64,morsel=32,tight=1,hardened=1"
        repeat one exact config instead of random sampling (repeatable flag).
"""

import argparse
import json
import os
import random
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scenarios import REGISTRY  # noqa: E402  (does not import polars)

THREAD_CHOICES = [(4, 1.0), (16, 2.0), (64, 3.0), (256, 2.0)]
MORSEL_CHOICES = [(100_000, 1.5), (1_024, 2.0), (32, 3.0), (1, 2.0)]

CRASH_SIGNALS = {
    signal.SIGSEGV: "segfault",
    signal.SIGBUS: "sigbus",
    signal.SIGILL: "sigill",
    signal.SIGTRAP: "sigtrap",
}


def wchoice(rng, pairs):
    vals, weights = zip(*pairs)
    return rng.choices(vals, weights=weights, k=1)[0]


def sample_config(rng):
    scen_names = [n for n, (_, w, _, _) in REGISTRY.items() if w > 0]
    scen_weights = [REGISTRY[n][1] for n in scen_names]
    return {
        "scenario": rng.choices(scen_names, weights=scen_weights, k=1)[0],
        "threads": wchoice(rng, THREAD_CHOICES),
        "morsel": wchoice(rng, MORSEL_CHOICES),
        "tight": rng.random() < 0.4,
        "hardened": rng.random() < 0.5,
        "max_scans": rng.choice([None, None, 4, 32]),
    }


def parse_pin(spec):
    cfg = {"scenario": "seq_pipeline", "threads": 64, "morsel": 32,
           "tight": False, "hardened": False, "max_scans": None}
    for part in spec.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        v = v.strip()
        if k == "scenario":
            cfg[k] = v
        elif k in ("threads", "morsel"):
            cfg[k] = int(v)
        elif k in ("tight", "hardened"):
            cfg[k] = v not in ("0", "false", "False", "")
        elif k == "max_scans":
            cfg[k] = None if v in ("", "none", "None") else int(v)
        else:
            raise ValueError(f"unknown pin key {k!r}")
    if cfg["scenario"] not in REGISTRY:
        raise ValueError(f"unknown scenario {cfg['scenario']!r}")
    return cfg


def build_env(cfg, data_dir, mult):
    env = os.environ.copy()
    env.update(
        POLARS_MAX_THREADS=str(cfg["threads"]),
        POLARS_IDEAL_MORSEL_SIZE=str(cfg["morsel"]),
        STRESS_DATA_DIR=str(data_dir),
        STRESS_ROWS_MULT=str(mult),
        PYTHONFAULTHANDLER="1",
        PYTHONUNBUFFERED="1",
        RUST_BACKTRACE="full",
    )
    if cfg["tight"]:
        env["POLARS_DEFAULT_LINEARIZER_BUFFER_SIZE"] = "1"
        env["POLARS_DEFAULT_DISTRIBUTOR_BUFFER_SIZE"] = "1"
        env["POLARS_DEFAULT_ZIP_HEAD_BUFFER_SIZE"] = "1"
    if cfg["hardened"]:
        # glibc heap tripwires + jemalloc junk fill (whichever allocator is live)
        env["MALLOC_PERTURB_"] = "90"
        env["MALLOC_CHECK_"] = "3"
        env["MALLOC_CONF"] = "junk:true"
        env["_RJEM_MALLOC_CONF"] = "junk:true"
    if cfg.get("max_scans"):
        env["POLARS_MAX_CONCURRENT_SCANS"] = str(cfg["max_scans"])
    return env


def timeout_for(cfg):
    base = REGISTRY[cfg["scenario"]][2]
    scale = 1.0
    if cfg["threads"] >= 64:
        scale *= 1.6
    if cfg["morsel"] <= 32:
        scale *= 1.6
    return min(int(base * scale), 520)


def sigint_injector(proc, stop_evt, rng, soft_window):
    time.sleep(3.0)
    end = time.time() + soft_window
    while not stop_evt.is_set() and time.time() < end and proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return
        stop_evt.wait(rng.uniform(1.5, 5.0))


def gdb_backtrace(python, core, out_path):
    try:
        r = subprocess.run(
            ["gdb", "--batch", "-q", "-ex", "set pagination off",
             "-ex", "bt", "-ex", "thread apply all bt 14", str(python), str(core)],
            capture_output=True, text=True, timeout=180,
        )
        out_path.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        return r.stdout
    except Exception as e:  # gdb missing/timeout: keep going
        out_path.write_text(f"gdb failed: {e!r}")
        return ""


def classify(rc, stderr, timed_out):
    if timed_out:
        return "hang"
    if rc is not None and rc < 0:
        sig = -rc
        if sig in CRASH_SIGNALS:
            return CRASH_SIGNALS[sig]
        if sig == signal.SIGABRT:
            low = stderr.lower()
            if "panicked at" in stderr:
                return "rust_panic_abort"
            if any(m in low for m in ("malloc", "free(): ", "corrupt", "double free")):
                return "heap_abort"
            return "abort"
        if sig == signal.SIGKILL:
            return "killed"
        if sig == signal.SIGINT:
            # We inject SIGINT ourselves (interrupt_victim); receiving it back is
            # expected, not a crash. Real crashes there surface as SEGV/ABRT/BUS.
            return "sigint"
        return f"signal_{sig}"
    if rc == 0:
        return "ok"
    if "panicked at" in stderr:
        return "rust_panic"
    return "py_error"


def interesting(outcome):
    """Crash-like outcomes worth keeping cores for."""
    return outcome not in ("ok", "py_error", "sigint")


def keep_dir(outcome):
    return outcome not in ("ok", "sigint")


def note_from_stderr(stderr):
    for line in stderr.splitlines():
        l = line.strip()
        if ("panicked at" in l or "Segmentation" in l or "overflow" in l
                or "free()" in l or "malloc" in l.lower() and "error" in l.lower()):
            return l[:300]
    return ""


def oom_hint():
    try:
        r = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=10)
        lines = [l for l in r.stdout.splitlines() if "illed process" in l]
        return lines[-1][-200:] if lines else ""
    except Exception:
        return ""


def run_iteration(i, cfg, seed, args, run_dir, kept_cores):
    itdir = run_dir / f"it{i:06d}"
    itdir.mkdir(parents=True, exist_ok=True)
    env = build_env(cfg, args.data, args.mult)
    timeout = timeout_for(cfg)
    inject = REGISTRY[cfg["scenario"]][3]
    cmd = [args.python, str(HERE / "scenarios.py"), cfg["scenario"], "--seed", str(seed)]

    def preexec():
        resource.setrlimit(resource.RLIMIT_CORE,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=itdir, env=env, preexec_fn=preexec,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stop_evt = threading.Event()
    if inject:
        threading.Thread(target=sigint_injector,
                         args=(proc, stop_evt, random.Random(seed ^ 0xABCD), timeout * 0.6),
                         daemon=True).start()
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.send_signal(signal.SIGABRT)  # try to get a core of the hang
        try:
            out, err = proc.communicate(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
    stop_evt.set()
    dur = time.time() - t0
    rc = proc.returncode
    outcome = classify(rc, err, timed_out)
    note = note_from_stderr(err)
    if outcome == "killed":
        hint = oom_hint()
        if hint:
            outcome = "oom_kill"
            note = note or hint

    bt_head = ""
    cores = sorted(itdir.glob("core*"))
    for core in cores:
        bt_path = itdir / (core.name + ".bt.txt")
        bt = gdb_backtrace(args.python, core, bt_path)
        bt_head = bt_head or "\n".join(bt.splitlines()[:20])
        if kept_cores[0] < args.max_cores_kept and interesting(outcome) and not timed_out:
            kept_cores[0] += 1
        else:
            core.unlink(missing_ok=True)

    if keep_dir(outcome):
        (itdir / "stdout.txt").write_text(out)
        (itdir / "stderr.txt").write_text(err)
    else:
        shutil.rmtree(itdir, ignore_errors=True)

    row = {
        "i": i, "ts": round(time.time(), 1), "scenario": cfg["scenario"],
        "threads": cfg["threads"], "morsel": cfg["morsel"],
        "tight": cfg["tight"], "hardened": cfg["hardened"],
        "max_scans": cfg.get("max_scans"), "seed": seed,
        "outcome": outcome, "rc": rc, "dur": round(dur, 1),
        "note": note, "dir": itdir.name if keep_dir(outcome) else "",
    }
    with open(run_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

    tag = ""
    if "overflow" in (note + err[:2000]) or "deque" in (note + err[:2000]):
        tag = "  <<< DEQUE/OVERFLOW SIGNATURE"
    marker = "!!!" if interesting(outcome) else "   "
    print(f"{marker} #{i:04d} {cfg['scenario']:<18} thr={cfg['threads']:<3} "
          f"mor={cfg['morsel']:<6} {'T' if cfg['tight'] else '-'}{'H' if cfg['hardened'] else '-'} "
          f"seed={seed:<6} -> {outcome:<12} {dur:6.1f}s {note[:80]}{tag}", flush=True)
    if interesting(outcome) and bt_head:
        print(bt_head, flush=True)
    return outcome


def ensure_data(args, run_dir):
    args.data.mkdir(parents=True, exist_ok=True)
    if (args.data / ".complete").exists():
        return
    print("generating datasets ...", flush=True)
    env = build_env({"threads": 4, "morsel": 100_000, "tight": False,
                     "hardened": False, "max_scans": None}, args.data, 1.0)
    r = subprocess.run([args.python, str(HERE / "scenarios.py"), "gen_data"],
                       env=env, cwd=run_dir, timeout=900)
    if r.returncode != 0:
        sys.exit("data generation failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--pin", action="append", default=[])
    ap.add_argument("--max-cores-kept", type=int, default=2)
    args = ap.parse_args()

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_data(args, run_dir)

    rng = random.Random(args.seed)
    pins = [parse_pin(s) for s in args.pin]
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    max_iters = args.iters if args.iters else 10**9

    print(f"run dir: {run_dir}  python: {args.python}", flush=True)
    counts = Counter()
    kept_cores = [0]
    i = 0
    try:
        while i < max_iters and (deadline is None or time.time() < deadline):
            cfg = pins[i % len(pins)] if pins else sample_config(rng)
            seed = rng.randrange(1_000_000)
            outcome = run_iteration(i, cfg, seed, args, run_dir, kept_cores)
            counts[outcome] += 1
            i += 1
    except KeyboardInterrupt:
        print("interrupted", flush=True)

    summary = {"iters": i, "counts": dict(counts)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
