#include "util.h"
#include <cstdio>

Timer::Timer(const char* timer_name) {
    name = std::string(timer_name);
    start = std::chrono::high_resolution_clock::now();
}

void Timer::lap(const char* label) {
    lapTimes.push_back(std::chrono::high_resolution_clock::now());
    lapLabels.push_back(std::string(label));
}

void Timer::finish() {
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total = end - start;

    if(lapTimes.size() == 0) {
        std::cout << "(Timer) (" << name << "): " << total.count() << " s" << std::endl;
    } else {
        std::cout << "(Timer) (" << name << ")"<< std::endl;
        for(int i = 0; i < lapLabels.size(); i++) {
            auto previous = i == 0 ? start : lapTimes[i-1];
            std::chrono::duration<double> elapsed = lapTimes[i] - previous;
            std::cout << "  (" << lapLabels[i] << "): " << elapsed.count() << " s" << std::endl;
        }
        std::cout << "  (Total): " << total.count() << " s" << std::endl;
    }
}

void print_vec_gpu(int* vec, int len) {
    int* buffer = (int*) malloc(len * sizeof(int));
    cudaMemcpy(buffer, vec, len * sizeof(int), cudaMemcpyDeviceToHost);
    print_vec_cpu(buffer, len);
    free(buffer);
}

void print_vec_cpu(int* vec, int len) {
    printf("[ ");
    for(int i = 0; i < len; i++)
        printf("%d ", vec[i]);
    printf("]\n");
}

const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "<unknown>";
    }
}
