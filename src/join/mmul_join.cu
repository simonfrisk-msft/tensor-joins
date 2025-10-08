#include "mmul_join.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void RelationToMatrix(Relation rel, float* matrix, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Tuple tuple = rel.data[idx];
    matrix[tuple.x + tuple.y * stride] = 1;
}

MMUL_Join::MMUL_Join(int a, int b, int c) {
    dimA = a;
    dimB = b;
    dimC = c;
}

Relation MMUL_Join::join(Relation rel1, Relation rel2) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;

    float *M1, *M2, *out, *C;

    int M1Size = dimA * dimB;
    int M2Size = dimB * dimC;
    int outSize = dimA * dimC;
    cudaMalloc(&M1, M1Size);
    cudaMalloc(&M2, M2Size);
    cudaMalloc(&out, outSize);
    cudaMalloc(&C, outSize);

    int numBlocksM1 = rel1.count / 1024 + 1;
    int numBlocksM2 = rel2.count / 1024 + 1;
    RelationToMatrix<<<numBlocksM1, 1024>>>(rel1, M1, dimB);
    RelationToMatrix<<<numBlocksM2, 1024>>>(rel2, M2, dimC);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dimC, dimA, dimB, &alpha,
                M2, dimC,
                M1, dimB,
                &beta,
                C, dimC);

    cublasDestroy(handle);

    Relation outRel;
    // Move matrix to outrel

    return outRel;
}
