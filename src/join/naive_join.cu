#include "naive_join.h"

__global__ void cuda_naive_join(Relation out, Relation rel1, Relation rel2, int n1, int n2, int* counter) {
    int xR1 = blockIdx.x * blockDim.x + threadIdx.x;
    int xR2 = blockIdx.y * blockDim.y + threadIdx.y;
    if(xR1 < n1 && xR2 < n2 && rel1.data[xR1].x == rel2.data[xR2].x) {
        int loc = atomicAdd(counter, 1);
        out.data[loc].x = rel1.data[xR1].y;
        out.data[loc].y = rel1.data[xR2].y;
    }
}

Relation Naive_Join::join(Relation rel1, Relation rel2) {
    int *counter;
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    Relation output;
    cudaMalloc(&output.data, rel1.count*rel2.count);

    dim3 blockDim(32, 32);
    dim3 gridDim(rel1.count / blockDim.x + 1, rel2.count / blockDim.y + 1);
    cuda_naive_join<<<gridDim, blockDim>>>(output, rel1, rel2, rel1.count, rel2.count, counter);
    cudaFree(&output.data);
}
