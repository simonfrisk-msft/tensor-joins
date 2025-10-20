#pragma once

typedef int8_t IN_MAT;
typedef int32_t OUT_MAT;

struct Tuple {
    int x;
    int y;
};

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
    void deduplicate();
    IN_MAT* toDenseMatrix(int domX, int domY); // Convert from relation to dense matrix
    Relation transferToDevice();
};

class CSRMatrix {
public:
    int* values;
    int* columnIndexes;
    int* rowOffsets;
    int numRows;
    int numNonZeros;
    CSRMatrix(Relation relation, int domX, int domY);
    Relation toRelation();
    void print();
};
