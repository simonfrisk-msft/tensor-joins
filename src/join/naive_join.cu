#include "naive_join.h"
#include "../util.h"
#include <cstdio>
#include <sstream>

__global__ void cuda_count_output(Relation<2> rel1, Relation<2> rel2, int n1, int n2, int* counter) {
    int xR1 = blockIdx.x * blockDim.x + threadIdx.x;
    int xR2 = blockIdx.y * blockDim.y + threadIdx.y;
    if(xR1 < n1 && xR2 < n2 && rel1.data[xR1].values[1] == rel2.data[xR2].values[0]) {
        atomicAdd(counter, 1);
    }
}

__global__ void cuda_naive_join(Relation<2> out, Relation<2> rel1, Relation<2> rel2, int n1, int n2, int* counter) {
    int xR1 = blockIdx.x * blockDim.x + threadIdx.x;
    int xR2 = blockIdx.y * blockDim.y + threadIdx.y;
    if(xR1 < n1 && xR2 < n2 && rel1.data[xR1].values[1] == rel2.data[xR2].values[0]) {
        int loc = atomicAdd(counter, 1);
        out.data[loc].values[0] = rel1.data[xR1].values[0];
        out.data[loc].values[1] = rel2.data[xR2].values[1];
    }
}

Relation<2> Naive_Join::join(Relation<2> rel1, Relation<2> rel2) {
    std::stringstream name;
    name << "Naive Join (" << rel1.count << ", " << rel2.count << ")";
    Timer t(name.str().c_str());

    int *counter;
    cudaMallocManaged(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    Relation<2> output;

    dim3 blockDim(32, 32);
    dim3 gridDim((rel1.count + blockDim.x - 1) / blockDim.x, (rel2.count + blockDim.y - 1) / blockDim.y);
    cuda_count_output<<<gridDim, blockDim>>>(rel1, rel2, rel1.count, rel2.count, counter);
    cudaDeviceSynchronize();

    CUDA_CHECK(cudaMalloc(&output.data, (*counter) * sizeof(Tuple<2>)));
    cudaMemset(counter, 0, sizeof(int));

    cuda_naive_join<<<gridDim, blockDim>>>(output, rel1, rel2, rel1.count, rel2.count, counter);

    cudaDeviceSynchronize();
    cudaMemcpy(&output.count, counter, sizeof(int), cudaMemcpyDeviceToHost);

    t.lap("Join w. Projection");

    output.deduplicate();

    t.lap("Deduplicate");

    t.finish();

    return output;
}
