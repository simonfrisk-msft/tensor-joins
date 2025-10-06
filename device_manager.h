#pragma once
#include <vector>
#include <tuple>
#include "dataset.h"

class DeviceManager {
private:
    std::vector<std::tuple<int,int>*> relations;
public:
    std::tuple<int,int>* TransferDataToDevice(Dataset* ds);
    void Echo();
    void PrintRelation(std::tuple<int,int>* relation, int maxCount);
    ~DeviceManager();
};
