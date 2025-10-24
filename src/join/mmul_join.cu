#include "mmul_join.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sstream>
#include <cstdio>
#include "../relation/relation.cuh"
#include "../relation/dense_matrix.cuh"
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

Relation<2> MMUL_Join::join(Relation<2> rel1, Relation<2> rel2) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    int *min, *max, *count;
    CUDA_CHECK(cudaMallocManaged(&min, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&max, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&count, sizeof(int)));
    CUDA_CHECK(cudaMemset(min, INT_MAX, sizeof(int)));
    CUDA_CHECK(cudaMemset(max, INT_MIN, sizeof(int)));
    CUDA_CHECK(cudaMemset(count, 0, sizeof(int)));
    rel1.countDomain(count, min, max);
    printf("Rel1 Domain X: count=%d min=%d max=%d\n", *count, *min, *max);

    std::stringstream name;
    name << "MMUL Join (" << rel1.count << ", " << rel2.count << ")";
    Timer t(name.str().c_str());

    int alpha = 1;
    int beta = 0;

    if (rel1.count <= 0 || rel2.count <= 0) {
        Relation<2> outRel;
        outRel.count = 0;
        cudaMalloc(&outRel.data, 0);
        return outRel;
    }

    DenseMatrix<OUT_MAT> outMatrix(dimA, dimC);
    DenseMatrix<IN_MAT> M1(rel1, dimA, dimB);
    DenseMatrix<IN_MAT> M2(rel2, dimB, dimC);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    t.lap("Relation to Matrix");

    CUBLAS_CHECK(cublasGemmEx(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        dimA, dimC, dimB, &alpha,
        M1.matrix, CUDA_R_8I, dimA,
        M2.matrix, CUDA_R_8I, dimB,
        &beta,
        outMatrix.matrix, CUDA_R_32I, dimA,
        CUDA_R_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    t.lap("Matrix Multiplication");
    
    Relation<2> outRel = outMatrix.toRelation();
    
    t.lap("Matrix to Relation");

    outMatrix.free();
    M1.free();
    M2.free();
    
    t.finish();
    cublasDestroy(handle);

    return outRel;
}
