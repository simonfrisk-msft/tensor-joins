#pragma once
#include "base_join.h"

class Naive_Join : public BaseJoin {
public:
    Relation join(Relation rel1, Relation rel2);
};
