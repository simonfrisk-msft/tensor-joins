#pragma once

#include "base_join.h"

class CSR_Join : public BaseJoin {
private:
    int dimA;
    int dimB;
    int dimC;
public:
    CSR_Join(int a, int b, int c);
    Relation<2> join(Relation<2> rel1, Relation<2> rel2);
};
