#pragma once

#include "relation.h"
#include <vector>

class Dataset {
protected:
    std::vector<Tuple> data;
public:
    Relation relation();
};

class RandomDataset: public Dataset {
public:
    RandomDataset(int domX, int domY, float probability);
};
