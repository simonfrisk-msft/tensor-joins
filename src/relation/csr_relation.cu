#include "./csr_relation.h"
#include "./tuple.cuh"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

struct TupleX {
    __host__ __device__
    int operator()(const Tuple<2>& t) const {
        return t.values[0];
    }
};

struct TupleY {
    __host__ __device__
    int operator()(const Tuple<2>& t) const {
        return t.values[1];
    }
};

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

__global__ void CSRMatrixToRelation(int numRows, int relationSize, int* rowOffsets, int* columnIndexes, float* values, Relation<2> relation) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relationSize) {
        relation.data[idx].values[1] = columnIndexes[idx];
        relation.data[idx].values[0] = binary_search_round_down(idx, rowOffsets, numRows);
    }
}

Relation<2> CSRMatrix::toRelation() {
    Relation<2> out;
    CUDA_CHECK(cudaMalloc(&out.data, numNonZeros * sizeof(Tuple<2>)));
    out.count = numNonZeros;
    int blockSize = 1024;
    int blocks = (numNonZeros + blockSize - 1) / blockSize;
    CSRMatrixToRelation<<<blocks, blockSize>>>(numRows, numNonZeros, rowOffsets, columnIndexes, values, out);
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}

CSRMatrix::CSRMatrix() {}

CSRMatrix::CSRMatrix(Relation<2> relation, int domX, int domY): domX(domX), domY(domY) {
    relation.sort();
    thrust::device_ptr<Tuple<2>> thrustData(relation.data);
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
    // Compute column indexes
    CUDA_CHECK(cudaMalloc(&columnIndexes, relation.count * sizeof(int)));
    auto columnIterator = thrust::make_transform_iterator(thrustData, TupleY());
    thrust::copy(columnIterator, columnIterator + relation.count, thrust::device_pointer_cast(columnIndexes));
    // Values are all ones
    CUDA_CHECK(cudaMalloc(&values, relation.count * sizeof(float)));
    CUDA_CHECK(cudaMemset(values, 1, relation.count * sizeof(float)));
    // Metadata
    numNonZeros = relation.count;
    numRows = domX; // TODO
}

void CSRMatrix::print() {
    std::cout << "- Print CSR Matrix" << std::endl;
    std::cout << "Row Offsets" << std::endl;
    print_vec_gpu(rowOffsets, numRows + 1);
    std::cout << "Column Indexes" << std::endl;
    print_vec_gpu(columnIndexes, numNonZeros);
    std::cout << "- End CSR Matrix" << std::endl;
}
