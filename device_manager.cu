#include "device_manager.h"
#include <stdio.h>

__global__ void cuda_echo() {
    printf("GPU Test.\n");
}

__global__ void cuda_print(Relation relation, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto [x, y] = relation[idx];
    if (idx < n) {
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

void DeviceManager::Echo() {
    cuda_echo<<<1,1>>>();
    cudaDeviceSynchronize(); 
}

void DeviceManager::PrintRelation(Tuple* relation, int maxCount) {
    cuda_print<<<1,32>>>(relation, maxCount);
    cudaDeviceSynchronize(); 
}

DeviceManager::~DeviceManager() {
    for (auto& relation: relations) {
        cudaFree(relation);
    }
}
