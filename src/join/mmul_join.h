#pragma once

#include "base_join.h"

class MMUL_Join : public BaseJoin {
private:
    int dimA;
    int dimB;
    int dimC;
public:
    MMUL_Join(int a, int b, int c);
    Relation join(Relation rel1, Relation rel2);
};
