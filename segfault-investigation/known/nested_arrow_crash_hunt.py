import polars as pl, pyarrow as pa, sys, subprocess, signal
tests = {
 "to_list_13": "s=pl.Series('c',[['x'*13]]); r=pl.Series('c', s.to_arrow()); print(r.to_list())",
 "len_bytes_13": "s=pl.Series('c',[['x'*13]]*1000); r=pl.Series('c', s.to_arrow()); print(r.list.explode().str.len_bytes().sum())",
 "hash_13": "s=pl.Series('c',[['x'*13]]*1000); r=pl.Series('c', s.to_arrow()); print(r.hash().sum())",
 "explode_upper_12": "s=pl.Series('c',[['x'*12,'y'*9]]*1000); r=pl.Series('c', s.to_arrow()); print(r.list.explode().str.to_uppercase().str.len_bytes().sum())",
 "join_9": "s=pl.Series('c',[['q'*9,'r'*16]]*10000); r=pl.Series('c', s.to_arrow()); print(r.list.join(',').str.len_bytes().sum())",
 "sort_unique": "s=pl.Series('c',[['q'*9,'r'*16,'s'*30]]*10000); r=pl.Series('c', s.to_arrow()); print(r.list.explode().unique().sort().len())",
 "write_parquet": "import io; s=pl.Series('c',[['q'*9,'r'*16,'s'*30]]*10000); r=pl.Series('c', s.to_arrow()); b=io.BytesIO(); r.to_frame().write_parquet(b); print(len(b.getvalue()))",
 "large_direct": "import pyarrow as pa; arr=pa.array([['x'*13,'y'*9,'zz'*20]]*20000, pa.large_list(pa.large_string())); r=pl.Series('c', arr); print(r.list.explode().str.len_bytes().sum(), r.list.explode().str.contains('y').sum())",
 "large_direct_group": "import pyarrow as pa; arr=pa.array([['x'*13,'y'*9,'zz'*20]]*20000, pa.large_list(pa.large_string())); r=pl.Series('c', arr); print(r.to_frame().group_by('c').len())",
 "large_direct_slice_gather": "import pyarrow as pa; arr=pa.array([['x'*13,'y'*9,'zz'*20]]*20000, pa.large_list(pa.large_string())); r=pl.Series('c', arr); print(r.list.get(1).str.slice(0,3).str.len_bytes().sum(), r.gather([5,3,19999]).to_list())",
 "sliced_array_nulls_pa_table": "import pyarrow as pa; df=pl.DataFrame({'a': pl.Series([[1,2],None,[5,6],[7,8]], dtype=pl.Array(pl.Int64,2))}).slice(1); t=pa.table(df); t.validate(full=True); print(t.to_pydict()); assert t.to_pydict()['a']==[None,[5,6],[7,8]], 'MISMATCH'",
 "sliced_array_nulls_to_arrow": "df=pl.DataFrame({'a': pl.Series([[1,2],None,[5,6],[7,8]], dtype=pl.Array(pl.Int64,2))}).slice(1); t=df.to_arrow(); t.validate(full=True); print(t.to_pydict()); assert t.to_pydict()['a']==[None,[5,6],[7,8]], 'MISMATCH'",
 "sliced_struct_nulls_pa_table": "import pyarrow as pa; df=pl.DataFrame({'a': pl.Series([{'x':1},None,{'x':5},{'x':7}])}).slice(1); t=pa.table(df); t.validate(full=True); print(t.to_pydict())",
 "sliced_str_pa_array_chunks": "import pyarrow as pa; s=pl.concat([pl.Series(['a'*20,'b'*30]), pl.Series(['c'*40,'d'*50,'e'*60])], rechunk=False).slice(1,3); a=pa.array(s); a.validate(full=True); print(a.to_pylist()); assert a.to_pylist()==['b'*30,'c'*40,'d'*50], 'MISMATCH '+str(a.to_pylist())",
 "sliced_str_pa_table_df": "import pyarrow as pa; df=pl.DataFrame({'s': ['a'*20,'b'*30,'c'*40,'d'*50,'e'*60], 'i': [1,2,3,4,5]}).slice(2,2); t=pa.table(df); t.validate(full=True); print(t.to_pydict()); assert t.to_pydict()['s']==['c'*40,'d'*50], 'MISMATCH'",
 "sliced_str_nulls_pa_table_df": "import pyarrow as pa; df=pl.DataFrame({'s': ['a'*20,None,'c'*40,'d'*50,None], 'i': [1,2,3,4,5]}).slice(1,3); t=pa.table(df); t.validate(full=True); print(t.to_pydict()); assert t.to_pydict()['s']==[None,'c'*40,'d'*50], 'MISMATCH'",
 "sliced_binview_stream_reader": "import pyarrow as pa; df=pl.DataFrame({'s': ['a'*20,None,'c'*40,'d'*50,None,'f'*70], 'i': [1,2,3,4,5,6]}).slice(3); r=pa.RecordBatchReader.from_stream(df); t=r.read_all(); t.validate(full=True); print(t.to_pydict()); assert t.to_pydict()['s']==['d'*50,None,'f'*70], 'MISMATCH'",
 "write_json_binary": "import io; df=pl.DataFrame({'b': [b'abc', b'def']}); b=io.BytesIO(); df.write_json(b); print(b.getvalue())",
 "write_ndjson_binary": "import io; df=pl.DataFrame({'b': [b'abc', b'def']}); b=io.BytesIO(); df.write_ndjson(b); print(b.getvalue())",
}
py = sys.executable
for name, code in tests.items():
    p = subprocess.run([py, "-W", "ignore", "-c", "import polars as pl\n" + code], capture_output=True, text=True, timeout=300)
    rc = p.returncode
    if rc < 0: st = f"SIGNAL {signal.Signals(-rc).name}"
    elif rc == 0: st = "ok: " + p.stdout.strip().replace("\n", " | ")[:100]
    else:
        last = [l for l in p.stderr.strip().splitlines() if l.strip()]
        st = f"exit {rc}: {last[-1][:140] if last else ''}"
    print(f"{name:32s} {st}", flush=True)
