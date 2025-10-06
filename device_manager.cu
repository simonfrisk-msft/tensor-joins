#include "device_manager.h"
#include <stdio.h>

__global__ void cuda_echo() {
    printf("GPU Test.\n");
}

__global__ void cuda_print(std::tuple<int,int>* relation, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto [x, y] = relation[idx];
    if (idx < n) {
        printf("%d: (%d, %d)\n", idx, x, y);
    }
}

std::tuple<int,int>* DeviceManager::TransferDataToDevice(Dataset* ds) {
    std::tuple<int,int>* deviceArray;
    cudaMalloc(&deviceArray, ds->size_bytes());
    cudaMemcpy(deviceArray, ds->data.data(), ds->size_bytes(), cudaMemcpyHostToDevice);
    relations.push_back(deviceArray);
    return deviceArray;
}

void DeviceManager::Echo() {
    cuda_echo<<<1,1>>>();
    cudaDeviceSynchronize(); 
}

void DeviceManager::PrintRelation(std::tuple<int,int>* relation, int maxCount) {
    cuda_print<<<1,32>>>(relation, maxCount);
    cudaDeviceSynchronize(); 
}

~DeviceManager::DeviceManager() {
    for (auto& relation: relations) {
        cudaFree(relation);
    }
}
