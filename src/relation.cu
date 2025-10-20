#include "relation.h"
#include "util.h"
#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

__global__ void print_gpu_kernel(Tuple* data, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < count) {
        Tuple tuple = data[idx];
        printf("(%d, %d)\n", tuple.x, tuple.y);
    }
}

void Relation::sort() {
    thrust::device_ptr<Tuple> begin(data);
    thrust::device_ptr<Tuple> end(data + count);
    thrust::sort(begin, end);
}

Relation::Relation(Tuple* tuples, int numberTuples) {
    data = tuples;
    count = numberTuples;
}

Relation::Relation() { }

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