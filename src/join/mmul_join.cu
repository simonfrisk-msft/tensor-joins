#include "mmul_join.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "../util.h"

int roundUp32(int num) {
    return (num + 31) / 32 * 32;
}

__global__ void PrintMatrix(int8_t* matrix, int n, int m) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < m && y < n)
        printf("(%d, %d): %d\n", x, y, matrix[y + n * x]);
}

__global__ void PrintMatrix(int32_t* matrix, int n, int m) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < m && y < n)
        printf("(%d, %d): %d\n", x, y, matrix[y + n * x]);
}

__global__ void RelationToMatrix(Relation rel, int8_t* matrix, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rel.count) {
        Tuple tuple = rel.data[idx];
        matrix[tuple.x + stride * tuple.y] = 1.0;
    }
}

__global__ void MatrixToRelation(Relation out, int32_t* matrix, int n, int m, int* counter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < m && y < n && matrix[x + m * y] > 0) {
        int idx = atomicAdd(counter, 1);
        out.data[idx] = Tuple{ x: x, y: y };
    }
}

MMUL_Join::MMUL_Join(int a, int b, int c) {
    dimA = roundUp32(a);
    dimB = roundUp32(b);
    dimC = roundUp32(c);
}

Relation MMUL_Join::join(Relation rel1, Relation rel2) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    Timer t("MMUL Join");

    int alpha = 1;
    int beta = 0;

    int8_t *M1, *M2;
    int32_t *outMatrix;

    int M1Size = dimA * dimB * sizeof(int8_t);
    int M2Size = dimB * dimC * sizeof(int8_t);
    int outSize = dimA * dimC * sizeof(int32_t);
    CUDA_CHECK(cudaMalloc(&M1, M1Size));
    CUDA_CHECK(cudaMalloc(&M2, M2Size));
    CUDA_CHECK(cudaMalloc(&outMatrix, outSize));

    int blockSizeRelToMat = 1024;
    int numBlocksM1 = (rel1.count + blockSizeRelToMat - 1) / blockSizeRelToMat;
    int numBlocksM2 = (rel2.count + blockSizeRelToMat - 1) / blockSizeRelToMat;

    t.lap("Init");

    RelationToMatrix<<<numBlocksM1, blockSizeRelToMat>>>(rel1, M1, dimA);
    RelationToMatrix<<<numBlocksM2, blockSizeRelToMat>>>(rel2, M2, dimB);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    t.lap("Rel to Matrix");

    cublasStatus_t mmul_status = cublasGemmEx(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        dimA, dimC, dimB, &alpha,
        M1, CUDA_R_8I, dimA,
        M2, CUDA_R_8I, dimB,
        &beta,
        outMatrix, CUDA_R_32I, dimC,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT);

    t.lap("Core MMUL");

    CUBLAS_CHECK(mmul_status);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(M1));
    CUDA_CHECK(cudaFree(M2));

    int *counter;
    CUDA_CHECK(cudaMalloc(&counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(counter, 0, sizeof(int)));
    Relation outRel;
    CUDA_CHECK(cudaMalloc(&outRel.data, rel1.count * rel2.count * sizeof(Tuple)));

    int blockSizeMat2Rel = 32;
    int blockCountX = (dimC + blockSizeMat2Rel - 1) / blockSizeMat2Rel;
    int blockCountY = (dimA + blockSizeMat2Rel - 1) / blockSizeMat2Rel;

    MatrixToRelation<<<dim3(blockCountX, blockCountY), dim3(blockSizeMat2Rel, blockSizeMat2Rel)>>>(outRel, outMatrix, dimA, dimC, counter);
    CUDA_CHECK(cudaDeviceSynchronize());

    t.lap("Matrix to Relation");

    cudaMemcpy(&outRel.count, counter, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(outMatrix);
    cudaFree(counter);

    t.finish();

    cublasDestroy(handle);

    return outRel;
}
