#pragma once

typedef int8_t IN_MAT;
typedef int32_t OUT_MAT;

struct Tuple {
    int x;
    int y;
};

__host__ __device__ inline
bool operator==(const Tuple& a, const Tuple& b) {
    return a.x == b.x && a.y == b.y;
}

__host__ __device__ inline
bool operator<(const Tuple& a, const Tuple& b) {
    if (a.x < b.x) return true;
    if (a.x > b.x) return false;
    return a.y < b.y;
}

class Relation {
public:
    int count;
    Tuple* data;
    Relation();
    Relation(Tuple* data, int count);
    Relation(OUT_MAT* matrix, int domX, int domY); // Convert from dense matrix to relation
    void free();
    void print_gpu();
    void print_stats();
    void sort();
    IN_MAT* toDenseMatrix(int domX, int domY); // Convert from relation to dense matrix
    Relation transferToDevice();
};
