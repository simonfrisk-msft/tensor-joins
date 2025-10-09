#include "mmul_join.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

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
    Tuple tuple = rel.data[idx];
    matrix[tuple.x * stride + tuple.y] = 1.0;
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
    cublasStatus_t status = cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    int alpha = 1;
    int beta = 0;

    int8_t *M1, *M2;
    int32_t *outMatrix;

    int M1Size = dimA * dimB * sizeof(int8_t);
    int M2Size = dimB * dimC * sizeof(int8_t);
    int outSize = dimA * dimC * sizeof(int32_t);
    cudaMalloc(&M1, M1Size);
    cudaMalloc(&M2, M2Size);
    cudaMalloc(&outMatrix, outSize);

    int blockSizeRel2Mat = 1024;
    int numBlocksM1 = (rel1.count + blockSizeRel2Mat - 1) / blockSizeRel2Mat;
    int numBlocksM2 = (rel2.count + blockSizeRel2Mat - 1) / blockSizeRel2Mat;
    RelationToMatrix<<<numBlocksM1, blockSizeRel2Mat>>>(rel1, M1, dimA);
    RelationToMatrix<<<numBlocksM2, blockSizeRel2Mat>>>(rel2, M2, dimB);
    cudaDeviceSynchronize();

    Timer t1("Core Matrix Multiplication");
    cublasStatus_t status2 = cublasGemmEx(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        dimA, dimC, dimB, &alpha,
        M1, CUDA_R_8I, dimA,
        M2, CUDA_R_8I, dimB,
        &beta,
        outMatrix, CUDA_R_32I, dimC,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    t1.finish();

    cublasDestroy(handle);

    int *counter;
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));
    Relation outRel;
    cudaMalloc(&outRel.data, rel1.count*rel2.count);

    int blockSizeMat2Rel = 32;
    int blockCountX = (dimC + blockSizeMat2Rel - 1) / blockSizeMat2Rel;
    int blockCountY = (dimA + blockSizeMat2Rel - 1) / blockSizeMat2Rel;
    MatrixToRelation<<<dim3(blockCountX, blockCountY), dim3(blockSizeMat2Rel, blockSizeMat2Rel)>>>(outRel, outMatrix, dimA, dimC, counter);
    cudaDeviceSynchronize();

    cudaMemcpy(&outRel.count, counter, sizeof(int), cudaMemcpyDeviceToHost);

    return outRel;
}
