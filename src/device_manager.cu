#include <stdio.h>
#include <iostream>
#include "device_manager.h"

__global__ void cuda_print(Relation relation, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto [x, y] = relation.data[idx];
    if (idx < n) {
        printf("%d: (%d, %d)\n", idx, x, y);
    }
}

__global__ void cuda_print_device_count(Relation relation, int* count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto [x, y] = relation.data[idx];
    if (idx < *count) {
        printf("%d: (%d, %d)\n", idx, x, y);
    }
}

Relation DeviceManager::TransferDataToDevice(Dataset* ds) {
    Relation deviceArray;
    Relation source = ds->relation();
    cudaMalloc(&deviceArray.data, source.count);
    cudaMemcpy(deviceArray.data, source.data, source.count, cudaMemcpyHostToDevice);
    relations.push_back(deviceArray);
    return deviceArray;
}

void DeviceManager::PrintRelation(Relation relation, int maxCount) {
    cuda_print<<<1,32>>>(relation, maxCount);
    cudaDeviceSynchronize(); 
}

DeviceManager::~DeviceManager() {
    for (auto& relation: relations) {
        cudaFree(relation.data);
    }
}
