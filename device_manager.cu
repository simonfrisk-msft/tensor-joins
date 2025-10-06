#include <stdio.h>
#include <iostream>
#include "device_manager.h"

typedef int* Matrix;

__global__ void cuda_naive_join(Relation out, Relation rel1, Relation rel2, int n1, int n2, int* counter) {
    int xR1 = blockIdx.x * blockDim.x + threadIdx.x;
    int xR2 = blockIdx.y * blockDim.y + threadIdx.y;
    if(xR1 < n1 && xR2 < n2 && rel1[xR1].x == rel2[xR2].x) {
        int loc = atomicAdd(counter, 1);
        out[loc].x = rel1[xR1].y;
        out[loc].y = rel1[xR2].y;
    }
}

__global__ void cuda_print(Relation relation, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto [x, y] = relation[idx];
    if (idx < n) {
        printf("%d: (%d, %d)\n", idx, x, y);
    }
}

__global__ void cuda_print_device_count(Relation relation, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto [x, y] = relation[idx];
    if (idx < *count) {
        printf("%d: (%d, %d)\n", idx, x, y);
    }
}

Relation DeviceManager::TransferDataToDevice(Dataset* ds) {
    Relation deviceArray;
    cudaMalloc(&deviceArray, ds->size_bytes());
    cudaMemcpy(deviceArray, ds->relation(), ds->size_bytes(), cudaMemcpyHostToDevice);
    relations.push_back(deviceArray);
    return deviceArray;
}

void DeviceManager::PrintRelation(Relation relation, int maxCount) {
    cuda_print<<<1,32>>>(relation, maxCount);
    cudaDeviceSynchronize(); 
}

void DeviceManager::NaiveJoin(Relation rel1, Relation rel2, int n1, int n2) {
    int *counter;
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    Relation output;
    cudaMalloc(&output, n1*n2);

    dim3 blockDim(32, 32);
    dim3 gridDim(n1 / blockDim.x + 1, n2 / blockDim.y + 1);
    cuda_naive_join<<<gridDim, blockDim>>>(output, rel1, rel2, n1, n2, counter);
    cuda_print_device_count<<<1,32>>>(output, counter);
    cudaFree(&output);
}

DeviceManager::~DeviceManager() {
    for (auto& relation: relations) {
        cudaFree(relation);
    }
}
