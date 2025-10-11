#include "mmul_join.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include "../util.h"

#define BLOCK 32

int roundUpToBlock(int num) {
    return ((num + BLOCK - 1) / BLOCK) * BLOCK;
}

__global__ void PrintMatrix(int8_t* matrix, int n, int m) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < m && matrix[x + y * n] > 0)
        printf("(%d, %d): %d\n", x, y, matrix[x + y * n]);
}

__global__ void PrintMatrix(int32_t* matrix, int n, int m) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < m && matrix[x + y * n] > 0)
        printf("(%d, %d): %d\n", x, y, matrix[x + y * n]);
}

__global__ void RelationToMatrix(Relation rel, int8_t* matrix, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rel.count) {
        Tuple tuple = rel.data[idx];
        matrix[tuple.x + tuple.y * stride] = 1.0;
    }
}

__global__ void CountOutputSizePerBlock(int32_t* matrix, int n, int m, int* outputSizePerBlock) {
    __shared__ int count;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        count = 0;
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int block = blockIdx.x + blockIdx.y * gridDim.x;
    if(x < n && y < m && matrix[x + y * n] > 0)
        atomicAdd(&count, 1);
    if (threadIdx.x == 0 && threadIdx.y == 0)
        outputSizePerBlock[block] = count;
}

__global__ void MatrixToRelation(Relation out, int32_t* matrix, int n, int m, int* prefixOutputSizePerBlock) {
    __shared__ int count;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        count = 0;
    __syncthreads();
    int block = blockIdx.x + blockIdx.y * gridDim.x;
    int globalOffset = prefixOutputSizePerBlock[block];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < n && y < m && matrix[x + y * n] > 0) {
        int idx = atomicAdd(&count, 1);
        out.data[globalOffset + idx] = Tuple{ x: x, y: y };
    }
}

MMUL_Join::MMUL_Join(int a, int b, int c) {
    dimA = roundUpToBlock(a);
    dimB = roundUpToBlock(b);
    dimC = roundUpToBlock(c);
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

    if(rel1.count != 0)
        RelationToMatrix<<<numBlocksM1, blockSizeRelToMat>>>(rel1, M1, dimA);
    if(rel2.count != 0)
        RelationToMatrix<<<numBlocksM2, blockSizeRelToMat>>>(rel2, M2, dimB);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    t.lap("Relation to Matrix");

    cublasStatus_t mmul_status = cublasGemmEx(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        dimA, dimC, dimB, &alpha,
        M1, CUDA_R_8I, dimA,
        M2, CUDA_R_8I, dimB,
        &beta,
        outMatrix, CUDA_R_32I, dimC,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();
    CUBLAS_CHECK(mmul_status);
    CUDA_CHECK(cudaGetLastError());

    t.lap("Matrix Multiplication");

    int blockCountX = (dimC + BLOCK - 1) / BLOCK;
    int blockCountY = (dimA + BLOCK - 1) / BLOCK;
    int blockCount = blockCountX * blockCountY;

    int* outputSizePerBlock;
    int* prefixOutputSizePerBlock;
    CUDA_CHECK(cudaMalloc(&outputSizePerBlock, blockCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&prefixOutputSizePerBlock, (blockCount+1) * sizeof(int)));

    CountOutputSizePerBlock<<<dim3(blockCountX, blockCountY), dim3(BLOCK, BLOCK)>>>(outMatrix, dimA, dimC, outputSizePerBlock);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> thrust_ptr(outputSizePerBlock);
    thrust::device_ptr<int> thrust_prefix_ptr(prefixOutputSizePerBlock);
    thrust::inclusive_scan(thrust_ptr, thrust_ptr + blockCount, thrust_prefix_ptr+1);

    Relation outRel;
    CUDA_CHECK(cudaMemcpy(&outRel.count, prefixOutputSizePerBlock + blockCount, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&outRel.data, outRel.count * sizeof(Tuple)));

    t.lap("Counting and prefix sum");

    MatrixToRelation<<<dim3(blockCountX, blockCountY), dim3(BLOCK, BLOCK)>>>(outRel, outMatrix, dimA, dimC, prefixOutputSizePerBlock);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    t.lap("Matrix to Relation");

    CUDA_CHECK(cudaFree(outMatrix));
    CUDA_CHECK(cudaFree(M1));
    CUDA_CHECK(cudaFree(M2));
    
    t.finish();
    cublasDestroy(handle);

    return outRel;
}
