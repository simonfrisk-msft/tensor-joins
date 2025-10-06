#pragma once
#include <vector>
#include <string>
#include "tuple.h"
#include "dataset.h"

class DeviceManager {
private:
    std::vector<Relation> relations;
    std::string name;
public:
    Relation TransferDataToDevice(Dataset* ds);
    void PrintRelation(Relation relation, int maxCount);
    void NaiveJoin(Relation rel1, Relation rel2, int n1, int n2);
    ~DeviceManager();
};
