#pragma once
#include <tuple>
#include <vector>

class Dataset {
public:
    std::vector<std::tuple<int,int>> data;
    void print_summary();
    void print_data(int maxCount);
};

class RandomDataset: public Dataset {
public:
    RandomDataset(int domX, int domY, float probability);
};
