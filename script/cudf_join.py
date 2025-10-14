import cudf
import cupy as cp
import time

for p in [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]:
    print("-------------------- Probability ", p)
    N1 = 1000
    N2 = 1000
    N3 = 1000
    domain_A = cp.arange(N1)
    domain_B = cp.arange(N2)
    domain_C = cp.arange(N3)
    
    A_vals, B_vals_R = cp.meshgrid(domain_A, domain_B, indexing='ij')
    A_flat = A_vals.ravel()
    B_flat_R = B_vals_R.ravel()
    mask_R = cp.random.random(A_flat.shape[0]) < p
    R = cudf.DataFrame({
        'A': A_flat[mask_R],
        'B': B_flat_R[mask_R]
    })

    B_vals_S, C_vals = cp.meshgrid(domain_B, domain_C, indexing='ij')
    B_flat_S = B_vals_S.ravel()
    C_flat = C_vals.ravel()
    mask_S = cp.random.random(B_flat_S.shape[0]) < p
    S = cudf.DataFrame({
        'B': B_flat_S[mask_S],
        'C': C_flat[mask_S]
    })

    print(f"R sampled size: {len(R)}")
    print(f"S sampled size: {len(S)}")  

    start = time.perf_counter()

    joined = R.merge(S, on=["B"], how="inner")
    lap = time.perf_counter()
    result = joined.drop(columns=["B"]).drop_duplicates()

    end = time.perf_counter()
    print(f"Total time: {end - start:.6f} seconds")
    print(f"Of which projection: {end - lap:.6f} seconds")
    print(f"Join result shape: {len(result)} rows")

