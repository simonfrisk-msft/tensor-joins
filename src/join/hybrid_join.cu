#include "hybrid_join.h"
#include "mmul_join.h"
#include "naive_join.h"
#include "../util.h"
#include <cstdio>
#include <sstream>
#include "deduplicate.h"

// Take a relation R, and an attribute a with domain d
// Return the vector of length d, with degrees of values in a
// TODO for now hard code finding degrees in X/Y
__global__ void findDegreesX(Relation relation, int domain, int* degreeVector) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relation.count) {
        Tuple t = relation.data[idx];
        atomicAdd(&degreeVector[t.x], 1);
    }
}

// Find which tuple to put in what partition
__global__ void multiplyPartitionHLX(Relation relation, int* degreeVector, int* partition, int length, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relation.count) {
        Tuple t = relation.data[idx];
        int isLight = 1;
        if(degreeVector[t.x] > threshold)
            isLight = 0;
        partition[idx] = isLight;
    }
}

// Calculate the size of each partition
__global__ void calculatePartitionSizes(int relationSize, int* partition, int* partitionLengths) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relationSize) {
        atomicAdd(&partitionLengths[partition[idx]], 1);
    }
}

// Take a relation, put tuples into different partitions
__global__ void partition(Relation relation, Relation* partitionBuffers, int* partition) {
    int tupleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tupleIdx < relation.count) {
        Tuple t = relation.data[tupleIdx];
        int p = partition[tupleIdx];
        int bufferIdx = atomicAdd(&partitionBuffers[p].count, 1);
        // TODO this is a random access. Can fix later. Maybe first sort input on partition?
        // Actually, if just sort the relation by the partition index, can create the new relations by just taking a pointer to the correct index in the buffer
        partitionBuffers[p].data[bufferIdx] = t; 
    }
}

Hybrid_Join::Hybrid_Join(int a, int b, int c) {
    domX = a;
    domY = b;
    domZ = c;
}

Relation Hybrid_Join::join(Relation relationR, Relation relationS) {
    std::stringstream name;
    name << "Hybrid Join (" << relationR.count << ", " << relationS.count << ")";
    Timer t(name.str().c_str());
    // Allocate histograms ---
    int *degXInR;
    CUDA_CHECK(cudaMallocManaged(&degXInR, domX * sizeof(int)));
    // Compute degrees ---
    findDegreesX<<<(relationR.count + 1024 - 1) / 1024, 1024>>>(relationR, domX, degXInR);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.lap("Compute degrees");
    // Partition ---
    // Vector that for each element in R, contains an int of which partition it should bee in
    int *partitionR;
    CUDA_CHECK(cudaMallocManaged(&partitionR, relationR.count * sizeof(int)));
    // Find which tuples to put in what partition
    int threshold = 10;
    multiplyPartitionHLX<<<(relationR.count + 1024 - 1) / 1024, 1024>>>(relationR, degXInR, partitionR, relationR.count, threshold);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.lap("Decide which tuples in which partitions");
    // Calculate size of partitions
    int* partitionLengths;
    CUDA_CHECK(cudaMallocManaged(&partitionLengths, 2 * sizeof(int)));
    calculatePartitionSizes<<<(relationR.count + 1024 - 1) / 1024, 1024>>>(relationR.count, partitionR, partitionLengths);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Allocate partition buffers
    Relation* partitionBuffers;
    CUDA_CHECK(cudaMallocManaged(&partitionBuffers, 2 * sizeof(Relation)));
    CUDA_CHECK(cudaMalloc(&partitionBuffers[0].data, partitionLengths[0] * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&partitionBuffers[1].data, partitionLengths[1] * sizeof(int)));
    // Partition
    partition<<<(relationR.count + 1024 - 1) / 1024, 1024>>>(relationR, partitionBuffers, partitionR);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.lap("Partition");
    // Compute the joins ---
    Naive_Join naive_join;
    MMUL_Join mmul_join(domX, domY, domZ); // TODO DOING THIS BREAKS POINT; USE SMALLER DOM
    Relation light;
    Relation heavy;
    if(partitionBuffers[1].count > 0)
        light = naive_join.join(partitionBuffers[1], relationS);
    if(partitionBuffers[0].count > 1)
        heavy = mmul_join.join(partitionBuffers[0], relationS);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.lap("Join w. Projection");

    // TODO MORE NEEDED; UNION; DEDUPLICATE

    t.finish();

    return light; 
}
