#pragma once
#include "base_join.h"

class PHashJoin : public BaseJoin {
public:
    Relation join(Relation rel1, Relation rel2);
};
