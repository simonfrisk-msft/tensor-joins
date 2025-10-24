#pragma once
#include "base_join.h"

class Hybrid_Join : public BaseJoin {
private:
    int domX;
    int domY;
    int domZ;
public:
    Hybrid_Join(int a, int b, int c);
    Relation<2> join(Relation<2> rel1, Relation<2> rel2);
};
