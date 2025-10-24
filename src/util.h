#pragma once
#include <chrono>
#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>

class Timer {
private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    std::vector<std::chrono::high_resolution_clock::time_point> lapTimes;
    std::vector<std::string> lapLabels;
public:
    Timer(const char* name);
    void lap(const char* name);
    void finish();
};

void print_vec_cpu(int* vec, int len);
void print_vec_gpu(int* vec, int len);

const char* cublasGetErrorString(cublasStatus_t status);

#define CUDA_CHECK(err) \
    do { \
        cudaError_t cuda_err = (err); \
        if (cuda_err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(cuda_err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(err) \
    do { \
        cublasStatus_t cublas_err = (err); \
        if (cublas_err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << cublasGetErrorString(cublas_err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUSPARSE_CHECK(func) \
    do { \
        cusparseStatus_t status = (func); \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSparse Error: " << cusparseGetErrorString(status) \
                        << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
