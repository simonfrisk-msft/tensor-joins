#include "relation.h"
#include "util.h"
#include <cstdio>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

__host__ __device__ inline
bool operator==(const Tuple& a, const Tuple& b) {
    return a.x == b.x && a.y == b.y;
}

__host__ __device__ inline
bool operator<(const Tuple& a, const Tuple& b) {
    if (a.x < b.x) return true;
    if (a.x > b.x) return false;
    return a.y < b.y;
}

struct TupleX {
    __host__ __device__
    int operator()(const Tuple& t) const {
        return t.x;
    }
};

struct TupleY {
    __host__ __device__
    int operator()(const Tuple& t) const {
        return t.y;
    }
};

__global__ void print_gpu_kernel(Tuple* data, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < count) {
        Tuple tuple = data[idx];
        printf("(%d, %d)\n", tuple.x, tuple.y);
    }
}

__device__ int binary_search_round_down(int search, int* array, int length) {
    int low = 0, high = length;
    while (low < high) {
        int mid = (low + high) / 2;
        if (array[mid + 1] <= search)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

__global__ void CSRMatrixToRelation(int numRows, int relationSize, int* rowOffsets, int* columnIndexes, int* values, Relation relation) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relationSize) {
        relation.data[idx].y = columnIndexes[idx];
        relation.data[idx].x = binary_search_round_down(idx, rowOffsets, numRows);
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

void Relation::deduplicate() {
    thrust::device_ptr<Tuple> begin(data);
    thrust::device_ptr<Tuple> end(data + count);
    thrust::sort(begin, end);
    thrust::device_ptr<Tuple> new_end = thrust::unique(begin, end);
    count = new_end - begin;
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
    CUDA_CHECK(cudaMallocManaged(&outputSizePerBlock, blockCountRelToMatrix * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&prefixOutputSizePerBlock, (blockCountRelToMatrix+1) * sizeof(int)));
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
    CUDA_CHECK(cudaMallocManaged(&matrix, size));

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

CSRMatrix::CSRMatrix(Relation relation, int domX, int domY) {
    relation.sort();
    thrust::device_ptr<Tuple> thrustData(relation.data);
    auto rowIterator = thrust::make_transform_iterator(thrustData, TupleX());
    CUDA_CHECK(cudaMalloc(&rowOffsets, (domX + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemset(rowOffsets, 0, (domX + 1) * sizeof(int)));
    // Compute number of non-zeros per row
    thrust::counting_iterator<int> searchBegin(0);
    thrust::upper_bound(
        rowIterator, rowIterator + relation.count,
        searchBegin, searchBegin + domX,
        thrust::device_pointer_cast(rowOffsets) + 1
    );
    //thrust::inclusive_scan(thrust::device_pointer_cast(rowOffsets), thrust::device_pointer_cast(rowOffsets) + domX + 1, thrust::device_pointer_cast(rowOffsets));
    // Compute column indexes
    CUDA_CHECK(cudaMalloc(&columnIndexes, relation.count * sizeof(int)));
    auto columnIterator = thrust::make_transform_iterator(thrustData, TupleY());
    thrust::copy(columnIterator, columnIterator + relation.count, thrust::device_pointer_cast(columnIndexes));
    // Values are all ones
    CUDA_CHECK(cudaMalloc(&values, relation.count * sizeof(int)));
    CUDA_CHECK(cudaMemset(values, 1, relation.count * sizeof(int)));
    // Metadata
    numNonZeros = relation.count;
    numRows = domX;
}

Relation CSRMatrix::toRelation() {
    Relation out;
    CUDA_CHECK(cudaMalloc(&out.data, numNonZeros * sizeof(Tuple)));
    out.count = numNonZeros;
    int blockSize = 1024;
    int blocks = (numNonZeros + blockSize - 1) / blockSize;
    CSRMatrixToRelation<<<blocks, blockSize>>>(numRows, numNonZeros, rowOffsets, columnIndexes, values, out);
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

void CSRMatrix::print() {
    std::cout << "- Print CSR Matrix" << std::endl;
    std::cout << "Row Offsets" << std::endl;
    print_vec_gpu(rowOffsets, numRows + 1);
    std::cout << "Column Indexes" << std::endl;
    print_vec_gpu(columnIndexes, numNonZeros);
    std::cout << "- End CSR Matrix" << std::endl;
}
