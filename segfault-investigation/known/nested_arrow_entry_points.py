import polars as pl, pyarrow as pa, sys, traceback, io
import pyarrow.parquet as pq, pyarrow.dataset as pads, tempfile, os
print("polars", pl.__version__, "pyarrow", pa.__version__)
vals = [["x" * 13, "y" * 9, "z"], None, [], ["a", "bb" * 20]] * 3
def check(name, f):
    try:
        got = f()
        print(f"{name:45s} {'OK' if got == vals else 'DATA MISMATCH: ' + str(got)[:120]}")
    except BaseException as e:
        print(f"{name:45s} EXC {type(e).__name__}: {str(e)[:120]}")
    sys.stdout.flush()
for lt, st in [(pa.large_list, pa.large_string), (pa.list_, pa.string), (pa.large_list, pa.string), (pa.list_, pa.large_string), (pa.large_list, pa.string_view)]:
    arr = pa.array(vals, type=lt(st()))
    tag = f"{lt.__name__}<{st.__name__}>"
    check(f"pl.Series(arr) {tag}", lambda: pl.Series("c", arr).to_list())
    check(f"pl.from_arrow(arr) {tag}", lambda: pl.from_arrow(arr).to_list())
    check(f"pl.from_arrow(table) {tag}", lambda: pl.from_arrow(pa.table({"c": arr}))["c"].to_list())
    check(f"pl.DataFrame(table) {tag}", lambda: pl.DataFrame(pa.table({"c": arr}))["c"].to_list())
    check(f"pl.from_arrow(chunked) {tag}", lambda: pl.from_arrow(pa.chunked_array([arr[:5], arr[5:]]))["c"].to_list() if False else pl.from_arrow(pa.chunked_array([arr[:5], arr[5:]])).to_list())
    check(f"pl.Series(chunked) {tag}", lambda: pl.Series("c", pa.chunked_array([arr[:5], arr[5:]])).to_list())
    check(f"pl.from_arrow(batch) {tag}", lambda: pl.from_arrow(pa.RecordBatch.from_pydict({"c": arr}))["c"].to_list())
    with tempfile.TemporaryDirectory() as d:
        pq.write_table(pa.table({"c": arr, "i": list(range(len(vals)))}), os.path.join(d, "f.parquet"))
        check(f"scan_pyarrow_dataset {tag}", lambda: pl.scan_pyarrow_dataset(pads.dataset(d, format="parquet")).collect()["c"].to_list())
        check(f"read_parquet(use_pyarrow=True) {tag}", lambda: pl.read_parquet(os.path.join(d, "f.parquet"), use_pyarrow=True)["c"].to_list())
        check(f"read_parquet native {tag}", lambda: pl.read_parquet(os.path.join(d, "f.parquet"))["c"].to_list())
# struct with strings
svals = [{"s": "x" * 13, "l": ["y" * 9]}, None, {"s": "", "l": []}] * 4
sarr = pa.array(svals, type=pa.struct([("s", pa.large_string()), ("l", pa.large_list(pa.large_string()))]))
def chk_s(name, f):
    try:
        got = f(); print(f"{name:45s} {'OK' if got == [None if v is None else v for v in svals] else 'DATA MISMATCH: ' + str(got)[:120]}")
    except BaseException as e: print(f"{name:45s} EXC {type(e).__name__}: {str(e)[:120]}")
chk_s("pl.Series(struct<large_string,large_list>)", lambda: pl.Series("c", sarr).to_list())
chk_s("pl.from_arrow(struct table)", lambda: pl.from_arrow(pa.table({"c": sarr}))["c"].to_list())
# pandas
import pandas as pd
pdf = pd.DataFrame({"c": pd.Series(vals, dtype=pd.ArrowDtype(pa.large_list(pa.large_string())))})
check("from_pandas arrow-backed large_list<large_string>", lambda: pl.from_pandas(pdf)["c"].to_list())
pdf2 = pd.DataFrame({"c": pd.Series(vals, dtype=pd.ArrowDtype(pa.list_(pa.string())))})
check("from_pandas arrow-backed list<string>", lambda: pl.from_pandas(pdf2)["c"].to_list())
# dictionary-encoded strings inside list
darr = pa.array(vals, type=pa.large_list(pa.large_string()))
# binary
bvals = [[v.encode() for v in x] if x is not None else None for x in vals]
barr = pa.array(bvals, type=pa.large_list(pa.large_binary()))
try:
    got = pl.Series("c", barr).to_list(); print(f"{'pl.Series large_list<large_binary>':45s} {'OK' if got == bvals else 'DATA MISMATCH ' + str(got)[:100]}")
except BaseException as e: print(f"{'pl.Series large_list<large_binary>':45s} EXC {type(e).__name__}: {str(e)[:120]}")
