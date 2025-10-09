#pragma once
#include "../relation.h"
#include "../util.h"


class BaseJoin {
public:
    virtual Relation join(Relation rel1, Relation rel2) = 0;
};
