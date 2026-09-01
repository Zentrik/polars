#!/usr/bin/env python3
"""
BUG C: hard SIGSEGV importing a nested Arrow array (large_list/list of binary or
string) whose inner values are ~9-16 bytes. This is the single most likely cause of
the reporter's frequent, "sporadic" 1.37.1 segfaults: it is data-dependent (only
fires when a nested column happens to contain 9-16 byte inner strings/binaries, the
inline-vs-heap boundary of Arrow's view layout), so in a varied production pipeline
it looks random.

Affected: polars 1.37.1 (SIGSEGV). FIXED in 1.44.1 (upstream #28626 / PR #28632).
This is the concrete, actionable finding: UPGRADE from 1.37.1.

The fuzzers in ../fuzz measured this at scale: on 1.37.1, ~87% of random nested-Arrow
imports and ~100% of exotic nested types (map/dict/deep-nest) crash; on 1.44.1, zero
crashes across ~760 seeds.
"""
import pyarrow as pa
import polars as pl

print("polars", pl.__version__)

# Minimal reliable SIGSEGV on 1.37.1 (release build); harmless on 1.44.1.
arr = pa.array([[b"x" * 13]] * 200, pa.large_list(pa.binary()))
s = pl.Series("c", arr)
print(s.to_list()[:1])          # <-- SIGSEGV here on 1.37.1
print("survived -> not 1.37.1 (or already fixed):", pl.__version__)
