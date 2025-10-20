#include "relation.h"
#include "util.h"
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__global__ void print_gpu_kernel(Tuple* data, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < count) {
        Tuple tuple = data[idx];
        printf("(%d, %d)\n", tuple.x, tuple.y);
    }
}

__global__ void RelationToDenseMatrix(Tuple* data, int relationSize, IN_MAT* matrix, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relationSize) {
        Tuple tuple = data[idx];
        matrix[tuple.x + tuple.y * stride] = 1;
    }
}

__global__ void MatrixToRelation(Tuple* data, OUT_MAT* matrix, int n, int m, int* prefixOutputSizePerBlock) {
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
       data[globalOffset + outIdx] = Tuple{ x: x, y: y };
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

void Relation::sort() {
    thrust::device_ptr<Tuple> begin(data);
    thrust::device_ptr<Tuple> end(data + count);
    thrust::sort(begin, end);
}

Relation::Relation() { }

Relation::Relation(Tuple* tuples, int numberTuples) {
    data = tuples;
    count = numberTuples;
}

// Dense matrix to relation
Relation::Relation(OUT_MAT* matrix, int domX, int domY) {
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

    CUDA_CHECK(cudaMemcpy(&count, prefixOutputSizePerBlock + blockCountRelToMatrix, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMalloc(&data, count * sizeof(Tuple)));

    MatrixToRelation<<<blockCountRelToMatrix, relToMatrixBlock>>>(data, matrix, domX, domY, prefixOutputSizePerBlock);
}

// Matrix to dense relation
IN_MAT* Relation::toDenseMatrix(int domX, int domY) {
    int size = domX * domY * sizeof(IN_MAT);
    IN_MAT *matrix;
    CUDA_CHECK(cudaMalloc(&matrix, size));

    int blockSizeRelToMat = 1024;
    int numBlocks = (count + blockSizeRelToMat - 1) / blockSizeRelToMat;
    RelationToDenseMatrix<<<numBlocks, blockSizeRelToMat>>>(data, count, matrix, domX);
    return matrix;
}

void Relation::free() {
    CUDA_CHECK(cudaFree(data));
}

void Relation::print_gpu() {
    printf("------------ (Relation)\n");
    int blockSize = 256;
    int blocks = (count + blockSize - 1) / blockSize;
    print_gpu_kernel<<<blocks, blockSize>>>(data, count);
    cudaDeviceSynchronize();
    printf("------------ (%d tuples)\n", count);
}

void Relation::print_stats() {
    printf("%d tuples\n", count);
}

Relation Relation::transferToDevice() {
    Relation deviceRelation;
    CUDA_CHECK(cudaMalloc(&deviceRelation.data, count * sizeof(Tuple)));
    CUDA_CHECK(cudaMemcpy(deviceRelation.data, data, count * sizeof(Tuple), cudaMemcpyHostToDevice));
    deviceRelation.count = count;
    return deviceRelation;
}