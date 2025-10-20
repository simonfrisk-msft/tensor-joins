#include "mmul_join.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <sstream>
#include <cstdio>
#include "../relation.h"
#include "../util.h"

#define MM_BLOCK 32

int roundUpToBlock(int num) {
    return ((num + MM_BLOCK - 1) / MM_BLOCK) * MM_BLOCK;
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

    cusparseSpMatDescr_t mat1, mat2, matOut;

    std::stringstream name;
    name << "MMUL Join (" << rel1.count << ", " << rel2.count << ")";
    Timer t(name.str().c_str());

    int alpha = 1;
    int beta = 0;

    OUT_MAT *outMatrix;
    if (rel1.count <= 0 || rel2.count <= 0) {
        Relation outRel;
        outRel.count = 0;
        cudaMalloc(&outRel.data, 0);
        return outRel;
    }
    int outSize = dimA * dimC * sizeof(OUT_MAT);
    CUDA_CHECK(cudaMalloc(&outMatrix, outSize));

    IN_MAT* M1 = rel1.toDenseMatrix(dimA, dimB);
    IN_MAT* M2 = rel2.toDenseMatrix(dimB, dimC);
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
    
    Relation outRel(outMatrix, dimA, dimC);
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
