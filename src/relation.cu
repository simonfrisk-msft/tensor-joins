#include "relation.h"
#include <cstdio>

__global__ void print_gpu_kernel(Tuple* data, int count) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < count) {
        Tuple tuple = data[idx];
        printf("(%d, %d)\n", tuple.x, tuple.y);
    }
}

Relation::Relation(Tuple* tuples, int numberTuples) {
    data = tuples;
    count = numberTuples;
}

Relation::Relation() { }

void Relation::free() {
    cudaFree(data);
}

void Relation::print_gpu() {
    printf("------------ (%d tuples)\n", count);
    int blockSize = 256;
    int blocks = (count + blockSize - 1) / blockSize;
    print_gpu_kernel<<<blocks, blockSize>>>(data, count);
    cudaDeviceSynchronize();
    printf("------------\n");
}

void Relation::print_stats() {
    printf("%d tuples\n", count);
}

Relation Relation::transferToDevice() {
    Relation deviceRelation;
    cudaMalloc(&deviceRelation.data, count * sizeof(Tuple));
    cudaMemcpy(deviceRelation.data, data, count * sizeof(Tuple), cudaMemcpyHostToDevice);
    return deviceRelation;
}