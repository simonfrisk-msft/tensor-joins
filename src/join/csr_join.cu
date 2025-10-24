#include "csr_join.h"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <sstream>
#include <cstdio>
#include "../relation/relation.cuh"
#include "../relation/csr_relation.h"
#include "../util.h"

CSR_Join::CSR_Join(int a, int b, int c) {
    dimA = a;
    dimB = b;
    dimC = c;
}

Relation<2> CSR_Join::join(Relation<2> rel1, Relation<2> rel2) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));    

    std::stringstream name;
    name << "CSR Join (" << rel1.count << ", " << rel2.count << ")";
    Timer t(name.str().c_str());

    float alpha = 1;
    float beta = 0;

    cusparseSpMatDescr_t mat1, mat2, matOut;

    CSRMatrix rel1csr(rel1, dimA, dimB);
    CSRMatrix rel2csr(rel2, dimB, dimC);

    CUSPARSE_CHECK(cusparseCreateCsr(&mat1, dimA, dimB, rel1.count,
            rel1csr.rowOffsets, rel1csr.columnIndexes, rel1csr.values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(&mat2, dimB, dimC, rel2.count,
            rel2csr.rowOffsets, rel2csr.columnIndexes, rel2csr.values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(&matOut, dimA, dimC, 0,
            nullptr, nullptr, nullptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    t.lap("Initialization");

    size_t bufferSize1 = 0;
    void* dBuffer1 = nullptr;

    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat1, mat2, &beta, matOut,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1, nullptr));

    cudaMalloc(&dBuffer1, bufferSize1);

    CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat1, mat2, &beta, matOut,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1, dBuffer1));

    t.lap("Work estimation");

    size_t bufferSize2 = 0;
    void* dBuffer2 = nullptr;

    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat1, mat2, &beta, matOut,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, nullptr));

    cudaMalloc(&dBuffer2, bufferSize2);

    CUSPARSE_CHECK(cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat1, mat2, &beta, matOut,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, dBuffer2));

    t.lap("Core MMUL");

    // Output buffers
    int64_t outRows, outCols, outNumZero;
    cusparseSpMatGetSize(matOut, &outRows, &outCols, &outNumZero);
    CSRMatrix outcsr;
    outcsr.numRows = outRows;
    outcsr.numNonZeros = outNumZero;

    cudaMalloc(&outcsr.rowOffsets, sizeof(int) * (outcsr.numRows + 1));
    cudaMalloc(&outcsr.columnIndexes, sizeof(int) * outcsr.numNonZeros);
    cudaMalloc(&outcsr.values, sizeof(float) * outcsr.numNonZeros);

    CUSPARSE_CHECK(cusparseCsrSetPointers(matOut, outcsr.rowOffsets, outcsr.columnIndexes, outcsr.values));

    CUSPARSE_CHECK(cusparseSpGEMM_copy(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat1, mat2, &beta, matOut,
        CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc));

    Relation<2> outRel = outcsr.toRelation();

    t.lap("Matrix to relation");

    // Cleanup
    CUSPARSE_CHECK(cusparseDestroySpMat(mat1));
    CUSPARSE_CHECK(cusparseDestroySpMat(mat2));
    CUSPARSE_CHECK(cusparseDestroySpMat(matOut));
    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemmDesc));
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);

    t.finish();
    cusparseDestroy(handle);

    return outRel;
}
