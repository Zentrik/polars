# Polars 1.37.1+ crash / correctness investigation

Goal: find bugs and reproducers in polars 1.37.1 (the version the reporter runs,
via the conda-forge build) and newer, with an emphasis on crashes/segfaults and
on issues **not already reported** upstream.

Everything here was run against clean wheels: `polars==1.37.1` and each release up
to `polars==1.44.1` (latest at time of writing), plus the conda-forge `polars=1.37.1`
build (which segfaults identically to the wheel on the FFI cases below).

## TL;DR

Two **new, unreported, silent correctness bugs** were found and root-caused. Both
reproduce on **every** release from 1.37.1 through 1.44.1, so upgrading does not fix
them:

| Bug | What happens | Engine(s) | Reproducer |
|-----|--------------|-----------|------------|
| **A** | `scan_parquet` + filter on a `Decimal` column silently drops matching rows, and writes corrupt min/max stats that also break pyarrow/other readers | eager + in-memory + streaming | `reproducers/bug_A_decimal_parquet_statistics.py` |
| **B** | `scan_parquet().filter(...).slice(neg_offset, len)` returns too many rows | streaming only | `reproducers/bug_B_streaming_negative_slice_after_filter.py` |
| **D** | `pl.col(...).rle()` on the streaming engine panics (`assertion failed: chunks.len() == 1`) whenever the column has >= 5 chunks (natural after reading many files / concat / vstack) | streaming only | `reproducers/bug_D_streaming_rle_multichunk_panic.py` |

Bug **D** was found by building 1.44.1 from source **with debug assertions** and fuzzing
that (see the debug-build section below); it reproduces on the plain release wheel too,
since it is a hard `assert!`. It is the first crash found that is genuinely live on
1.44.1 from a normal operation.

The sporadic **segfaults** the reporter sees are best explained by already-reported,
already-fixed memory-safety bugs that are live in 1.37.1 (see "Known crashes"
below). The single most likely culprit is the **nested-Arrow FFI corruption
(#28626)** — `reproducers/bug_C_nested_binary_ffi_segfault_1371.py`. It is a hard
SIGSEGV on 1.37.1 (wheel and conda build), fixed in 1.44. See the segfault-hunt
section below for how pervasive it is on 1.37.1 versus 1.44.1.

**Actionable recommendation: upgrade off 1.37.1.** The dedicated segfault hunt
below could not produce a single crash on 1.44.1 across tens of thousands of
executions, while 1.37.1 crashes on the large majority of nested/exotic Arrow
imports.

---

## Segfault hunt: 1.37.1 vs 1.44.1 (follow-up)

The reporter noted the fault rate is *higher on 1.37.1 than on newer versions*, so
this is a timing/data-dependent memory bug that is largely (for everything found
here, entirely) fixed by 1.44. To pin it down, seven fuzzing strategies were run on
both a `1.37.1` and a `1.44.1` wheel:

| Strategy (script in `fuzz/`) | 1.37.1 | 1.44.1 |
|------------------------------|--------|--------|
| Arrow-import FFI (`fuzz_arrow_import.py`) — nested list/struct/fsl, sliced, chunked, dictionary | **766 / 880 seeds SIGSEGV** | 0 / 652 |
| Exotic Arrow FFI (`fuzz_arrow_exotic.py`) — map, run-end-encoded, dict-of-nested, deep nesting | **121 / 121 seeds SIGSEGV** | 0 / 107 |
| Streaming-vs-in-memory determinism (`fuzz_det3.py`) | streaming "not implemented" panics | 0 |
| 400-thread executor stress (`highthread.py`, `#27842`) | only a proper rt32 length error | 0 |
| 12M-row memory pressure (`fuzz_mempressure.py` / `over_stress.py`, `#29020`) | OOM-kill only | OOM-kill only |
| Random-op / string-view / parquet-differential (phase 1) | FFI segfaults | 0 crashes |

**Conclusion.** On 1.37.1, importing a nested Arrow array (a `list`/`large_list` of
`binary`/`string`/`dictionary`, a `map`, a run-end-encoded array, or anything nested
inside them) whose inner values land in the ~9–16 byte inline-vs-heap window of
Arrow's view layout is a hard SIGSEGV, and this happens for the large majority of
such shapes. Because it is data-dependent, a production pipeline sees it as a random,
sporadic crash. **1.44.1 handled every one of these gracefully** — no segfault, no
panic, no hang, across ~760 fuzz seeds plus the determinism, thread, and
memory-pressure runs.

A couple of extra 1.37.1-only defects fell out and are worth noting (all fixed in
1.44): importing a `RunEndEncodedArray` **hangs forever** on 1.37.1 (1.44 raises a
clean "not supported" error), and several streaming `group_by`/`over` chains panic
with "not implemented".

The one open segfault that still affects 1.44.1 upstream is **#29020** (streaming
`.over()` under sustained cgroup memory-reclaim pressure). It could not be triggered
in this sandbox because reproducing it needs the reporter's exact
`systemd-run -p MemoryHigh=… -p MemoryMax=…` throttling, not a plain hard limit
(a hard limit just OOM-kills). If newer-version faults persist for the reporter, run
`fuzz/fuzz_mempressure.py` and `fuzz/over_stress.py` inside *their* memory-constrained
environment (where the real reclaim pressure exists) and watch for exit 139/134;
that is the most likely remaining cause and is already tracked as #29020.

### Debug-assertions build of 1.44.1 (the right tool)

A plain release wheel has debug-assertions **off**, so latent out-of-bounds /
invalid-layout bugs corrupt silently and only rarely happen to hit unmapped memory —
which is why fuzzing the wheel found nothing new on 1.44.1. The fix is to build 1.44.1
from source with debug-assertions **on** (optimized profile, symbols/LTO off to keep
it small): the standard-library and polars `debug_assert!` / `assert!` checks then turn
that latent corruption into a loud panic at the exact point it happens. The built
library carries 5,323 `unsafe precondition` checks and 432 `slice::from_raw_parts`
alignment checks — the exact canaries from #29020 — confirming the assertions compiled
in.

Fuzzing that build surfaced **Bug D** (streaming `rle()` on a >= 5-chunk column,
`assert!(chunks.len() == 1)` in `series/builder.rs:168`, reached from the streaming
run-length-encoding node). It is a real crash on the release wheel, from a common
pattern, on the latest version — the concrete answer to "find a crash on 1.44.1". The
hunt for `debug_assert!`-only failures (silent UB in release, i.e. true
segfaults-in-waiting) continues in `fuzz/` against this build.

Reproducer: `reproducers/bug_D_streaming_rle_multichunk_panic.py`. Minimal:

```python
import polars as pl
s = pl.concat([pl.Series("p", [i % 3]) for i in range(5)], rechunk=False)  # >= 5 chunks
pl.DataFrame({"p": s}).lazy().select(pl.col("p").rle()).collect(engine="streaming")
# 1.44.1: PANIC assertion failed: chunks.len() == 1   (in-memory engine is fine)
```

### Bug C — nested-Arrow FFI SIGSEGV on 1.37.1 (reported #28626, fixed 1.44)

`reproducers/bug_C_nested_binary_ffi_segfault_1371.py`. Minimal:

```python
import pyarrow as pa, polars as pl
pl.Series("c", pa.array([[b"x" * 13]] * 200, pa.large_list(pa.binary()))).to_list()
# 1.37.1: SIGSEGV.  1.44.1: fine.
```

---

## Bug A — Decimal parquet statistics are written with the wrong sign order (NEW)

**Reproducer:** `reproducers/bug_A_decimal_parquet_statistics.py`
**Severity:** silent data loss on a filtered scan; the written file's metadata is
corrupt for *every* reader (confirmed with pyarrow).
**Status upstream:** not reported (searched issues; nearest matches #26293 / #17289
are unrelated decimal-sink encoding errors).
**Affected:** 1.37.1 … 1.44.1. Eager `read_parquet`-then-scan, in-memory, and
streaming all drop the row.

Minimal:

```python
import io, polars as pl
df = pl.DataFrame({"c": pl.Series([1, -1], dtype=pl.Int64).cast(pl.Decimal(38, 0))})
buf = io.BytesIO(); df.write_parquet(buf, data_page_size=1); buf.seek(0)
pl.scan_parquet(buf).filter(pl.col("c") == pl.lit(1, dtype=pl.Decimal(38, 0))).collect()
# -> 0 rows, should be 1
```

The file polars writes reports `min=1, max=-1` (min > max is impossible for correct
statistics), so `pyarrow.parquet.read_table(file, filters=[("c","=",Decimal(1))])`
also returns 0 rows.

**Trigger conditions (all required):**
* `Decimal` precision **>= 19** (i128-backed). Precision <= 18 (i64-backed) is fine.
* The pruned column-chunk / data-page min and max **straddle zero** (a mix of
  negative and non-negative values). Same-sign chunks are fine.
* Statistics/pruning actually engage — many data pages (small `data_page_size`) or
  a real multi-row-group file whose chunk spans zero.

**Root cause** (write side):
`crates/polars-parquet/src/arrow/write/fixed_size_binary/mod.rs::build_statistics_decimal`.
i128 `Decimal` (precision > 18) is stored as `FIXED_LEN_BYTE_ARRAY` and its min/max
are encoded as two's-complement big-endian bytes (`x.to_be_bytes()`), but the parquet
`FIXED_LEN_BYTE_ARRAY` statistics are compared with **unsigned** byte order during
pruning. When a chunk straddles zero the stored min (`-1` → `0xFF..FF`) is
byte-greater than the stored max (`1` → `0x00..01`); a pruner sees `min > max`,
treats the range as empty, and prunes the chunk that actually contains the match.
The same `build_statistics_decimal` is used for both signed `Int128` and unsigned
`UInt128`, which is the tell. This matches the observed behavior exactly: only
mixed-sign chunks, only precision >= 19, and other engines (pyarrow) are affected
because the on-disk metadata itself is wrong.

`known/decimal_boundary.py` and `known/decimal_rootcause.py` demonstrate the
precision boundary, the straddles-zero condition, and the cross-engine corruption.

---

## Bug B — streaming `scan_parquet` filter + negative slice over-returns rows (NEW)

**Reproducer:** `reproducers/bug_B_streaming_negative_slice_after_filter.py`
**Severity:** silent wrong results (extra rows past the requested length).
**Status upstream:** not reported.
**Affected:** 1.37.1 … 1.44.1, **streaming engine only** (in-memory is correct).

Minimal:

```python
import polars as pl
df = pl.DataFrame({"k": [i % 5 == 0 for i in range(100)], "i": range(100)})
df.write_parquet("f.parquet", row_group_size=7)
pl.scan_parquet("f.parquet").filter("k").select("i").slice(-10, 5).collect(engine="streaming")
# -> 6 rows [50,55,60,65,70,75]; correct answer is 5 rows [50,55,60,65,70]
```

**Trigger conditions:**
* a `scan_*` source with more than one row group (small `row_group_size` relative to
  the number of matching rows),
* a filter, and
* a `slice` with a **negative offset AND an explicit length** (`slice(-10, 5)`,
  `tail(n)`). `slice(-10)` without a length is correct; positive offsets are correct.

The negative-offset slice is mis-combined with the per-row-group filtered output, so
rows leak past the requested length. `known/negslice_min.py` sweeps offsets, lengths,
and row-group sizes.

A related smaller variant: `scan_ndjson(...).filter(...).slice(-k, n)` can return 0
rows for larger inputs; not fully minimized here.

---

## Known crashes verified live on 1.37.1 (already reported upstream)

These are memory-safety / panic bugs that are present and reproducible on 1.37.1
(the reporter's version) and are the most plausible source of the sporadic
segfaults. Each has an upstream issue and most are fixed by 1.44. Full pass/fail
matrices for both versions are in `results/known_repros_polars-1.37.1.txt` and
`results/known_repros_polars-1.44.1.txt`; the runner is `known/run_known.py`.

| Repro (script case) | Symptom on 1.37.1 | Upstream | Fixed by 1.44 |
|---------------------|-------------------|----------|---------------|
| nested-string Arrow FFI: `pl.Series(name, s.to_arrow())` with 9–16-byte inner strings | **SIGSEGV / SIGABRT**, data corruption | #28626 | yes |
| `pl.concat(dfs, how="align")` with many frames | **SIGSEGV** (stack overflow) | #26788 | yes |
| deep `merge_sorted` chain then sort | **SIGSEGV** | #26960 | yes |
| `Series([{...}]).sqrt()/cbrt()/pct_change()/ewm_*` on a Struct | **SIGSEGV / infinite recursion** | #28563 | yes (raises) |
| deep binary-expr / `join_where` left join at scale | **SIGSEGV** (rayon stack overflow) | #15211 / #29026 | partly |
| streaming `.over()` under cgroup memory pressure | **SIGSEGV** + wrong results | #29020 | **open** |
| async-executor deque underflow at ~400 threads | sporadic **SIGSEGV** | #27842 | open (accepted) |
| streaming `rolling()` with empty windows | wrong length / panic | #26732, #26234 | #26732 fixed 1.39; #26234 open |

`known/nested_arrow_crash_hunt.py` reproduces the #28626 family (the highest-value
one for a 1.37.1 conda user — it is a plain SIGSEGV on both the wheel and the
conda-forge build), and `known/rolling_min.py` bisects the rolling-window regression
across versions.

---

## Fuzzing

Harnesses under `fuzz/` (random-op frame fuzzer, parquet round-trip differential
fuzzer, string-view fuzzer with a Python oracle, and a multi-threaded
categorical/object/callback/IO-plugin fuzzer). They were run for thousands of seeds
on both 1.37.1 and 1.44.1.

* On **1.44.1** the fuzzers found **no new memory-safety segfault**. The only
  process-exits were OOM kills / allocation-failure aborts under an intentional
  address-space limit (adversarial `group_by_dynamic`/`rolling` inputs requesting
  huge allocations), plus a handful of `not yet implemented` panics
  (`Int128 → FixedLenByteArray(16)` on `pyarrow`-written parquet;
  `Writing BinaryView to JSON`; `timedelta` tolerance on a non-temporal
  `join_asof`). These are functional gaps, not crashes on well-formed input.
* The parquet differential fuzzer is what surfaced **Bug A** and **Bug B** above.
* On **1.37.1** the fuzzers additionally reproduced the known FFI segfault family.

The `polars-stream` source was also audited for data races (`.over()` fallback path,
async executor, shared-buffer primitives). No definitely-unsound code was found in
this checkout; the audit's leads (an unchecked `perfect_sort` scatter in the window
map strategy, and trust of pyarrow/IO-plugin buffer immutability) are consistent
with #29020 but could not be turned into a standalone crash here.

## Layout

```
reproducers/  standalone, self-contained scripts for the two NEW bugs
known/        scripts for the known/verified crashes + minimizers used to root-cause A and B
fuzz/         the fuzzing harnesses and runners
results/      known-bug pass/fail matrices for 1.37.1 and 1.44.1
```
