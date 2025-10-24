#pragma once

#include "base_join.h"

class CSR_Join : public BaseJoin {
private:
    int dimA;
    int dimB;
    int dimC;
public:
    CSR_Join(int a, int b, int c);
    Relation join(Relation rel1, Relation rel2);
};
