#include "naive_join.h"
#include "../util.h"
#include <cstdio>
#include <sstream>
#include "deduplicate.h"

__global__ void cuda_naive_join(Relation out, Relation rel1, Relation rel2, int n1, int n2, int* counter) {
    int xR1 = blockIdx.x * blockDim.x + threadIdx.x;
    int xR2 = blockIdx.y * blockDim.y + threadIdx.y;
    if(xR1 < n1 && xR2 < n2 && rel1.data[xR1].x == rel2.data[xR2].x) {
        int loc = atomicAdd(counter, 1);
        out.data[loc].x = rel1.data[xR1].x;
        out.data[loc].y = rel2.data[xR2].y;
    }
}

Relation Naive_Join::join(Relation rel1, Relation rel2) {
    std::stringstream name;
    name << "Naive Join (" << rel1.count << ", " << rel2.count << ")";
    Timer t(name.str().c_str());

    int *counter;
    cudaMalloc(&counter, sizeof(int));
    cudaMemset(counter, 0, sizeof(int));

    Relation output;
    CUDA_CHECK(cudaMalloc(&output.data, rel1.count*rel2.count*sizeof(Tuple)));

    dim3 blockDim(32, 32);
    dim3 gridDim((rel1.count + blockDim.x - 1) / blockDim.x, (rel2.count + blockDim.y - 1) / blockDim.y);
    cuda_naive_join<<<gridDim, blockDim>>>(output, rel1, rel2, rel1.count, rel2.count, counter);

    cudaDeviceSynchronize();
    cudaMemcpy(&output.count, counter, sizeof(int), cudaMemcpyDeviceToHost);

    t.lap("Join w. Projection");

    deduplicate(output);

    t.lap("Deduplicate");

    t.finish();

    return output;
}
