#pragma once

#include "./relation/tuple.cuh"
#include "./relation/relation.cuh"
#include <vector>

class Dataset {
protected:
    std::vector<Tuple<2>> data;
    int domX, domY;
public:
    Relation<2> relation();
    int getX();
    int getY();
};

class RandomDataset: public Dataset {
public:
    RandomDataset(int domX, int domY, float probability);
};

class TxtFileDataset: public Dataset {
public:
    TxtFileDataset(const char* filename, int max_rows);
};
