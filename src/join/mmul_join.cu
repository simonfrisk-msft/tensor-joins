#include "mmul_join.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <sstream>
#include <cstdio>
#include "../util.h"

#define MM_BLOCK 32
typedef int8_t IN_MAT;
typedef int32_t OUT_MAT;

int roundUpToBlock(int num) {
    return ((num + MM_BLOCK - 1) / MM_BLOCK) * MM_BLOCK;
}

__global__ void RelationToMatrix(Relation rel, IN_MAT* matrix, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rel.count) {
        Tuple tuple = rel.data[idx];
        matrix[tuple.x + tuple.y * stride] = 1;
    }
}

__global__ void CountOutputSizePerBlock(OUT_MAT* matrix, int n, int m, int* outputSizePerBlock) {
    __shared__ int count;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        count = 0;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % n;
    int y = idx / n;
    int block = blockIdx.x;
    if(x < n && y < m && matrix[idx] > 0)
        atomicAdd(&count, 1);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        outputSizePerBlock[block] = count;
}

__global__ void MatrixToRelation(Relation out, OUT_MAT* matrix, int n, int m, int* prefixOutputSizePerBlock) {
    __shared__ int count;
    if (threadIdx.x == 0 && threadIdx.y == 0)
        count = 0;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % n;
    int y = idx / n;
    int block = blockIdx.x;
    int globalOffset = prefixOutputSizePerBlock[block];
    if(x < n && y < m && matrix[idx] > 0) {
       int outIdx = atomicAdd(&count, 1);
       out.data[globalOffset + outIdx] = Tuple{ x: x, y: y };
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

    std::stringstream name;
    name << "MMUL Join (" << rel1.count << ", " << rel2.count << ")";
    Timer t(name.str().c_str());

    int alpha = 1;
    int beta = 0;

    IN_MAT *M1, *M2;
    OUT_MAT *outMatrix;
    Relation outRel;

    int M1Size = dimA * dimB * sizeof(IN_MAT);
    int M2Size = dimB * dimC * sizeof(IN_MAT);
    int outSize = dimA * dimC * sizeof(OUT_MAT);
    CUDA_CHECK(cudaMalloc(&M1, M1Size));
    CUDA_CHECK(cudaMalloc(&M2, M2Size));
    CUDA_CHECK(cudaMalloc(&outMatrix, outSize));

    int blockSizeRelToMat = 1024;
    int numBlocksM1 = (rel1.count + blockSizeRelToMat - 1) / blockSizeRelToMat;
    int numBlocksM2 = (rel2.count + blockSizeRelToMat - 1) / blockSizeRelToMat;

    if (rel1.count <= 0 || rel2.count <= 0) {
        outRel.count = 0;
        cudaMalloc(&outRel.data, 0);
        return outRel;
    }

    RelationToMatrix<<<numBlocksM1, blockSizeRelToMat>>>(rel1, M1, dimA);
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
        CUDA_R_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaDeviceSynchronize();
    CUBLAS_CHECK(mmul_status);
    CUDA_CHECK(cudaGetLastError());

    t.lap("Matrix Multiplication");

    int relToMatrixBlock = 1024;
    int blockCountRelToMatrix = (dimA*dimC + relToMatrixBlock - 1) / relToMatrixBlock;

    int* outputSizePerBlock;
    int* prefixOutputSizePerBlock;
    CUDA_CHECK(cudaMalloc(&outputSizePerBlock, blockCountRelToMatrix * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&prefixOutputSizePerBlock, (blockCountRelToMatrix+1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(prefixOutputSizePerBlock, 0, sizeof(int))); // set first offset to 0

    CountOutputSizePerBlock<<<blockCountRelToMatrix, relToMatrixBlock>>>(outMatrix, dimA, dimC, outputSizePerBlock);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    t.lap("Output counting");

    thrust::device_ptr<int> thrust_ptr(outputSizePerBlock);
    thrust::device_ptr<int> thrust_prefix_ptr(prefixOutputSizePerBlock);
    thrust::inclusive_scan(thrust_ptr, thrust_ptr + blockCountRelToMatrix, thrust_prefix_ptr+1);
    CUDA_CHECK(cudaMemcpy(&outRel.count, prefixOutputSizePerBlock + blockCountRelToMatrix, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&outRel.data, outRel.count * sizeof(Tuple)));
    t.lap("Prefix sum");

    MatrixToRelation<<<blockCountRelToMatrix, relToMatrixBlock>>>(outRel, outMatrix, dimA, dimC, prefixOutputSizePerBlock);
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
