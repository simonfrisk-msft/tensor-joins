#include "util.h"

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
            std::chrono::duration<double> elapsed = lapTimes[i] - start;
            std::cout << "  (" << lapLabels[i] << "): " << elapsed.count() << " s" << std::endl;
        }
        std::cout << "  (Total): " << total.count() << " s" << std::endl;
    }
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
