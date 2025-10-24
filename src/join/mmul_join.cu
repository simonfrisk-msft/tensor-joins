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

Relation<2> MMUL_Join::join(Relation<2> rel1, Relation<2> rel2) {
    if (rel1.count <= 0 || rel2.count <= 0) {
        Relation<2> outRel;
        outRel.count = 0;
        cudaMalloc(&outRel.data, 0);
        return outRel;
    }

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    std::stringstream name;
    name << "MMUL Join (" << rel1.count << ", " << rel2.count << ")";
    Timer t(name.str().c_str());

    int *domX, *domY, *domY1, *domY2, *domZ;
    CUDA_CHECK(cudaMallocManaged(&domX, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&domY, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&domY1, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&domY2, sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&domZ, sizeof(int)));
    CUDA_CHECK(cudaMemset(domX, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(domY, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(domY1, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(domY2, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(domZ, 0, sizeof(int)));
    rel1.countDomain<0>(domX);
    rel1.countDomain<1>(domY1);
    rel2.countDomain<1>(domY2);
    rel2.countDomain<1>(domZ);
    CUDA_CHECK(cudaDeviceSynchronize());
    *domY = roundUpToBlock(12912); // TODO this is not just max, since things may be missing on one side.
    *domX = roundUpToBlock(12912);
    *domZ = roundUpToBlock(12912);
    printf("Domains %d %d %d\n", *domX, *domY, *domZ);

    // TODO consecutize domains

    t.lap("Consecutizing Domain");

    int alpha = 1;
    int beta = 0;

    DenseMatrix<OUT_MAT> outMatrix(*domX, *domZ);
    DenseMatrix<IN_MAT> M1(rel1, *domX, *domY);
    DenseMatrix<IN_MAT> M2(rel2, *domY, *domZ);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    t.lap("Relation to Matrix");

    CUBLAS_CHECK(cublasGemmEx(handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        *domX, *domZ, *domY, &alpha,
        M1.matrix, CUDA_R_8I, *domX,
        M2.matrix, CUDA_R_8I, *domY,
        &beta,
        outMatrix.matrix, CUDA_R_32I, *domX,
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
