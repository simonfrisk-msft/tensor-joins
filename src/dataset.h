#pragma once

#include "relation.h"
#include <vector>

class Dataset {
protected:
    std::vector<Tuple> data;
    int domX, domY;
public:
    Relation relation();
    int getX();
    int getY();
};

class RandomDataset: public Dataset {
public:
    RandomDataset(int domX, int domY, float probability);
};

class TxtFileDataset: public Dataset {
public:
    TxtFileDataset(const char* filename);
};
