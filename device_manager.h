#pragma once
#include <vector>
#include "tuple.h"
#include "dataset.h"

class DeviceManager {
private:
    std::vector<Relation> relations;
public:
    Relation TransferDataToDevice(Dataset* ds);
    void Echo();
    void PrintRelation(Relation relation, int maxCount);
    ~DeviceManager();
};
