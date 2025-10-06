#pragma once
#include "tuple.h"
#include <vector>

class Dataset {
protected:
    std::vector<Tuple> data;
public:
    void print_summary();
    void print_data(int maxCount);
    int tuple_count();
    int size_bytes();
    Relation relation();
};

class RandomDataset: public Dataset {
public:
    RandomDataset(int domX, int domY, float probability);
};
