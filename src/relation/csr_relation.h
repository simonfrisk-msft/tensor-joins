#pragma once
#include "./relation.cuh"

class CSRMatrix {
public:
    int domX, domY;
    float* values;
    int* columnIndexes;
    int* rowOffsets;
    int numRows;
    int numNonZeros;
    CSRMatrix(Relation<2> relation, int domX, int domY); // From relation to CSR
    CSRMatrix();
    Relation<2> toRelation(); // To CSR to relation 
    void print();
};