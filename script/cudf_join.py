import cudf
import cupy as cp
import time

for p in [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
    print("-------------------- Probability ", p)
    N = 1000
    vals = cp.arange(1, N + 1)
    x = cp.repeat(vals, N)
    y = cp.tile(vals, N)

    R1 = cudf.DataFrame({
        "a": x,
        "b": y
    })
    R2 = cudf.DataFrame({
        "a": x,
        "c": y
    })
    mask1 = cp.random.random(len(R1)) < p
    mask2 = cp.random.random(len(R2)) < p

    R1 = R1[mask1]
    R2 = R2[mask2]

    print(f"R1 sampled size: {len(R1)}")
    print(f"R2 sampled size: {len(R2)}")  

    start = time.perf_counter()

    joined = R1.merge(R2, on=["a"], how="inner")
    result = joined.drop(columns=["a"]).drop_duplicates()

    end = time.perf_counter()
    elapsed = end - start
    print(f"Elapsed time: {elapsed:.6f} seconds")
    print(f"Join result shape: {len(joined)} rows")


