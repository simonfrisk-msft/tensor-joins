#pragma once
#include <vector>
#include <string>
#include "relation.h"
#include "dataset.h"

class DeviceManager {
private:
    std::vector<Relation> relations;
    std::string name;
public:
    Relation TransferDataToDevice(Dataset* ds);
    void PrintRelation(Relation relation, int maxCount);
    ~DeviceManager();
};
