#pragma once
#include "../util.h"
#include "tuple.cuh"
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

template <int n>
__global__ void print_gpu_kernel(Tuple<n>* data, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < count) {
        Tuple<2>tuple = data[idx];
        printf("(");
        for (int i = 0; i < n; i++) {
            printf("%d", tuple.values[i]);
            if (i < n - 1)
                printf(", ");
        }
        printf(")\n");
    }
}

template <int n>
class Relation {
public:
    int count;
    Tuple<n>* data;
    Relation();
    Relation(Tuple<n>* data, int count);

    void free_gpu();
    void print_gpu();
    void print_stats();
    void deduplicate();
    Relation transferToDevice();

    // Sort on all attributes
    void sort();
    // Sort on one attribute
    template <int attribute>
    void sort_on_attribute();

    // Find the min, max, and count of distinct values for some attribute
    // count, min, max should be pointers to device/managed memory
    template <int attribute>
    void countDomain(int* count);
    void consecutifyRelation();
};

template <int n>
void Relation<n>::sort() {
    thrust::device_ptr<Tuple<n>> begin(data);
    thrust::device_ptr<Tuple<n>> end(data + count);
    thrust::sort(begin, end);
}

template <int n>
template <int attribute>
void Relation<n>::sort_on_attribute() {
    thrust::device_ptr<Tuple<n>> begin(data);
    thrust::device_ptr<Tuple<n>> end(data + count);
    thrust::sort(begin, end, CompareTuple<attribute, n>());
}

template <int n>
void Relation<n>::deduplicate() {
    thrust::device_ptr<Tuple<n>> begin(data);
    thrust::device_ptr<Tuple<n>> end(data + count);
    thrust::sort(begin, end);
    thrust::device_ptr<Tuple<n>> new_end = thrust::unique(begin, end);
    count = new_end - begin;
}

template <int n>
Relation<n>::Relation() { }

template <int n>
Relation<n>::Relation(Tuple<n>* tuples, int numberTuples) {
    data = tuples;
    count = numberTuples;
}

template <int n>
void Relation<n>::free_gpu() {
    if(data != nullptr)
        CUDA_CHECK(cudaFree(data));
}

template <int n>
void Relation<n>::print_gpu() {
    printf("------------ (Relation)\n");
    int blockSize = 256;
    int blocks = (count + blockSize - 1) / blockSize;
    print_gpu_kernel<n><<<blocks, blockSize>>>(data, count);
    cudaDeviceSynchronize();
    printf("------------ (%d tuples)\n", count);
}

template <int n>
void Relation<n>::print_stats() {
    printf("%d tuples\n", count);
}

template <int n>
Relation<n> Relation<n>::transferToDevice() {
    Relation<n> deviceRelation;
    CUDA_CHECK(cudaMalloc(&deviceRelation.data, count * sizeof(Tuple<n>)));
    CUDA_CHECK(cudaMemcpy(deviceRelation.data, data, count * sizeof(Tuple<n>), cudaMemcpyHostToDevice));
    deviceRelation.count = count;
    return deviceRelation;
}

template <int n, int attribute>
__global__ void domainProperties(Tuple<n>* data, int size, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && (idx == 0 || data[idx - 1].values[attribute] != data[idx].values[attribute])) {
        atomicAdd(count, 1);
    }
}

template <int n>
template <int attribute>
void Relation<n>::countDomain(int* domSize) {
    sort_on_attribute<attribute>();
    CUDA_CHECK(cudaMemset(domSize, 0, sizeof(int)));
    int blockSize = 1024;
    int blocks = (count + blockSize - 1) / blockSize; 
    domainProperties<n, attribute><<<blocks, blockSize>>>(data, count, domSize);
}

template <int n>
void Relation<n>::consecutifyRelation() {

}

