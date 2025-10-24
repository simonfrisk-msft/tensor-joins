import cudf
import time

path = "../data/stupid.txt"
print(f"Loading dataset from {path}...")
num = 300000
try:
    df1 = cudf.read_csv(path, sep=r" ", nrows=num, header=None, names=["A", "B"], dtype={"A": "int32", "B": "int32"})
    df2 = cudf.read_csv(path, sep=r" ", nrows=num, header=None, names=["B", "C"], dtype={"B": "int32", "C": "int32"})
except Exception as e:
    print("Failed to read file: ", e)
    exit()

print(f"Loaded {len(df1)} rows")

print("Starting join...")
start = time.perf_counter()
joined = df1.merge(df2, on=["B"], how="inner")
join_done = time.perf_counter()
result = joined.drop(columns=["B"]).drop_duplicates()
end = time.perf_counter()

print(f"Join: {len(joined)} intermediate rows in {join_done - start:.4f}s")
print(f"Projection+dedup: {len(result)} rows in {end - join_done:.4f}s")
print(f"Total time: {end - start:.4f}s")
