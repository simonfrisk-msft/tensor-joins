#include "hybrid_join.h"
#include "mmul_join.h"
#include "naive_join.h"
#include "../util.h"
#include <cstdio>
#include <sstream>

// Take a relation R, and an attribute a with domain d
// Return the vector of length d, with degrees of values in a
// TODO for now hard code finding degrees in X/Y
// TODO this is really slow
__global__ void findDegreesX(Relation<2> relation, int domain, int* degreeVector) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relation.count) {
        Tuple t = relation.data[idx];
        atomicAdd(&degreeVector[t.values[0]], 1);
    }
}

// Find which tuple to put in what partition
__global__ void multiplyPartitionHLX(Relation<2> relation, int* degreeVector, int* partition, int length, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < relation.count) {
        Tuple t = relation.data[idx];
        int isLight = 1;
        if(degreeVector[t.values[0]] > threshold)
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
__global__ void partition(Relation<2> relation, Relation<2>* partitionBuffers, int* partition) {
    int tupleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tupleIdx == 0)
        printf("counts %d %d\n", partitionBuffers[0].count, partitionBuffers[1].count);
    __syncthreads();
    if(tupleIdx < relation.count) {
        if(tupleIdx == 0)
            printf("Size %d\n", relation.count);
        __syncthreads();
        Tuple t = relation.data[tupleIdx];
        int p = partition[tupleIdx];
        if(tupleIdx == 0)
            printf("A %d\n", partitionBuffers[p].count);
        __syncthreads();

        int bufferIdx = atomicAdd(&partitionBuffers[p].count, 1);
        // TODO this is a random access. Can fix later. Maybe first sort input on partition?
        // Actually, if just sort the relation by the partition index, can create the new relations by just taking a pointer to the correct index in the buffer
        __syncthreads();
        if(tupleIdx == 0)
            printf("N %d\n", partitionBuffers[p].count);

        partitionBuffers[p].data[bufferIdx] = t; 
    }
}

Hybrid_Join::Hybrid_Join(int a, int b, int c) {
    domX = a;
    domY = b;
    domZ = c;
}

Relation<2> Hybrid_Join::join(Relation<2> relationR, Relation<2> relationS) {
    std::stringstream name;
    name << "Hybrid Join (" << relationR.count << ", " << relationS.count << ")";
    Timer t(name.str().c_str());
    // Allocate histograms ---
    int *degXInR;
    CUDA_CHECK(cudaMallocManaged(&degXInR, domX * sizeof(int)));
    CUDA_CHECK(cudaMemset(degXInR, 0, domX * sizeof(int)));
    // Compute degrees ---
    relationR.sort();
    t.lap("Sorting");
    findDegreesX<<<(relationR.count + 1024 - 1) / 1024, 1024>>>(relationR, domX, degXInR);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.lap("Compute degrees");
    // Partition ---
    // Vector that for each element in R, contains an int of which partition it should bee in
    int *partitionR;
    CUDA_CHECK(cudaMallocManaged(&partitionR, relationR.count * sizeof(int)));
    CUDA_CHECK(cudaMemset(partitionR, 0, relationR.count * sizeof(int))); // TODO We should initialize things to 1
    int threshold = domX / 33 ; // 3%
    multiplyPartitionHLX<<<(relationR.count + 1024 - 1) / 1024, 1024>>>(relationR, degXInR, partitionR, relationR.count, threshold);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.lap("Decide which tuples in which partitions");
    // Calculate size of partitions
    int* partitionLengths;
    CUDA_CHECK(cudaMallocManaged(&partitionLengths, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(partitionLengths, 0, 2 * sizeof(int)));
    calculatePartitionSizes<<<(relationR.count + 1024 - 1) / 1024, 1024>>>(relationR.count, partitionR, partitionLengths);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Allocate partition buffers
    Relation<2>* partitionBuffers;
    CUDA_CHECK(cudaMallocManaged(&partitionBuffers, 2 * sizeof(Relation<2>)));
    CUDA_CHECK(cudaMemset(&partitionBuffers[0].count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(&partitionBuffers[1].count, 0, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&partitionBuffers[0].data, partitionLengths[0] * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&partitionBuffers[1].data, partitionLengths[1] * sizeof(int)));
    // Partition
    printf("Launching partition w. %d %d\n", (relationR.count + 1024 - 1) / 1024, 1024);
    printf("Sizes allocated. %d %d\n", partitionLengths[0], partitionLengths[1]);
    partition<<<(relationR.count + 1024 - 1) / 1024, 1024>>>(relationR, partitionBuffers, partitionR);
    CUDA_CHECK(cudaDeviceSynchronize());
    t.lap("Partition");
    // Compute the joins ---
    Naive_Join naive_join;
    MMUL_Join mmul_join; // TODO DOING THIS BREAKS POINT; USE SMALLER DOM
    Relation<2> light;
    Relation<2> heavy;
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
