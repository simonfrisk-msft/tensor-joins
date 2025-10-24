#pragma once

#include "base_join.h"

class MMUL_Join : public BaseJoin {
private:
    int dimA;
    int dimB;
    int dimC;
public:
    MMUL_Join(int a, int b, int c);
    Relation<2> join(Relation<2> rel1, Relation<2> rel2);
};
