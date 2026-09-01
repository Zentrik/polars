import polars as pl, sys
print("polars", pl.__version__)
# multi-chunk column (built from many small chunks, no rechunk)
parts = [pl.Series("p", [i]) for i in [1,1,2,2,2,3,1,1]]
s = pl.concat(parts, rechunk=False)
print("n_chunks:", s.n_chunks())
df = pl.DataFrame({"p": s})
for eng in ["in-memory", "streaming"]:
    try:
        out = df.lazy().select(pl.col("p").rle()).collect(engine=eng)
        print(f"  {eng}: ok shape={out.shape}")
    except pl.exceptions.PanicException as e:
        print(f"  {eng}: PANIC {str(e)[:90]}")
    except Exception as e:
        print(f"  {eng}: {type(e).__name__} {str(e)[:80]}")
# also rle_id and via with_columns
for expr, name in [(pl.col("p").rle_id(), "rle_id"), (pl.col("p").rle(), "rle")]:
    try:
        df.lazy().select(expr).collect(engine="streaming"); print(f"  streaming {name}: ok")
    except pl.exceptions.PanicException as e: print(f"  streaming {name}: PANIC {str(e)[:80]}")
    except Exception as e: print(f"  streaming {name}: {type(e).__name__}")
