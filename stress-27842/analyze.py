#!/usr/bin/env python3
"""Summarize a harness results.jsonl: outcome tallies overall and per axis,
and list every interesting (crash-like) run with its note."""
import collections
import json
import sys
from pathlib import Path

CRASH = {"segfault", "sigbus", "sigill", "sigtrap", "abort", "heap_abort",
         "rust_panic", "rust_panic_abort", "hang", "oom_kill"}


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
    rows = [json.loads(l) for l in path.open()] if path.exists() else []
    if not rows:
        print("no results")
        return
    total = collections.Counter(r["outcome"] for r in rows)
    print(f"total iterations: {len(rows)}")
    print("outcomes:", dict(total))
    crashes = [r for r in rows if r["outcome"] in CRASH]
    print(f"\ncrash-like: {len(crashes)}")
    for r in crashes:
        print(f"  #{r['i']:04d} {r['outcome']:<14} {r['scenario']:<18} "
              f"thr={r['threads']} mor={r['morsel']} "
              f"tight={r['tight']} hard={r['hardened']} seed={r['seed']}")
        if r.get("note"):
            print(f"        note: {r['note']}")
        if r.get("dir"):
            print(f"        artifacts: {r['dir']}")

    # crash rate per axis value
    for axis in ("scenario", "threads", "morsel", "tight", "hardened"):
        by = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            b = by[r[axis]]
            b[0] += 1
            if r["outcome"] in CRASH:
                b[1] += 1
        line = ", ".join(f"{k}={v[1]}/{v[0]}" for k, v in sorted(by.items(), key=lambda x: str(x[0])))
        print(f"\n{axis} (crashes/total): {line}")


if __name__ == "__main__":
    main()
