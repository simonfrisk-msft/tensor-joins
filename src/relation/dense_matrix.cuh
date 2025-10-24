#pragma once
#include "relation.cuh"
#include "../util.h"
#include <type_traits>
#include <thrust/device_ptr.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

typedef int8_t IN_MAT;
typedef int32_t OUT_MAT;

template <typename T>
__global__ void RelationToMatrix(Tuple<2>* data, int relationSize, T* matrix, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relationSize) {
        Tuple tuple = data[idx];
        // Column major
        matrix[tuple.values[0] + tuple.values[1] * stride] = 1;
    }
}

template <typename T>
__global__ void MatrixToRelation(Tuple<2>* data, T* matrix, int n, int m, int* prefixOutputSizePerBlock) {
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
       data[globalOffset + outIdx] = Tuple<2>{{ x, y }};
    }
}

template <typename T>
__global__ void CountOutputSizePerBlock(T* matrix, int n, int m, int* outputSizePerBlock) {
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

template <typename T>
class DenseMatrix {
public:
    int domX, domY;
    T* matrix;
    DenseMatrix(Relation<2> relation, int domX, int domY);
    DenseMatrix(int domX, int domY);
    Relation<2> toRelation();
    void free();
};

// Matrix to dense relation
template <typename T>
DenseMatrix<T>::DenseMatrix(Relation<2> relation, int domX, int domY): domX(domX), domY(domY) {
    int size = domX * domY * sizeof(T);
    CUDA_CHECK(cudaMalloc(&matrix, size));
    int blockSizeRelToMat = 1024;
    int numBlocks = (relation.count + blockSizeRelToMat - 1) / blockSizeRelToMat;
    RelationToMatrix<<<numBlocks, blockSizeRelToMat>>>(relation.data, relation.count, matrix, domX);
}

template <typename T>
DenseMatrix<T>::DenseMatrix(int domX, int domY): domX(domX), domY(domY) {
    int size = domX * domY * sizeof(T);
    CUDA_CHECK(cudaMalloc(&matrix, size));
}

// Dense matrix to relation
template <typename T>
Relation<2> DenseMatrix<T>::toRelation() {
    int relToMatrixBlock = 1024;
    int blockCountRelToMatrix = (domX*domY + relToMatrixBlock - 1) / relToMatrixBlock;

    int* outputSizePerBlock;
    int* prefixOutputSizePerBlock;
    CUDA_CHECK(cudaMalloc(&outputSizePerBlock, blockCountRelToMatrix * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&prefixOutputSizePerBlock, (blockCountRelToMatrix+1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(prefixOutputSizePerBlock, 0, sizeof(int))); // set first offset to 0

    CountOutputSizePerBlock<<<blockCountRelToMatrix, relToMatrixBlock>>>(matrix, domX, domY, outputSizePerBlock);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<int> thrust_ptr(outputSizePerBlock);
    thrust::device_ptr<int> thrust_prefix_ptr(prefixOutputSizePerBlock);
    thrust::inclusive_scan(thrust_ptr, thrust_ptr + blockCountRelToMatrix, thrust_prefix_ptr+1);

    Relation<2> outRel;
    CUDA_CHECK(cudaMemcpy(&outRel.count, prefixOutputSizePerBlock + blockCountRelToMatrix, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&outRel.data, outRel.count * sizeof(Tuple<2>)));

    MatrixToRelation<<<blockCountRelToMatrix, relToMatrixBlock>>>(outRel.data, matrix, domX, domY, prefixOutputSizePerBlock);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    return outRel;
}

template <typename T>
void DenseMatrix<T>::free() {
    CUDA_CHECK(cudaFree(matrix));
}
