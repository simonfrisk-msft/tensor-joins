#pragma once

#include "base_join.h"

class MMUL_Join : public BaseJoin {
public:
    Relation<2> join(Relation<2> rel1, Relation<2> rel2);
};
