#pragma once
#include "../relation/relation.cuh"
#include "../relation/tuple.cuh"
#include "../util.h"

class BaseJoin {
public:
    virtual Relation<2> join(Relation<2> rel1, Relation<2> rel2) = 0;
};
