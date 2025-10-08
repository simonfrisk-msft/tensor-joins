#include "relation.h"

Relation::Relation(Tuple* tuples, int numberTuples) {
    data = tuples;
    count = numberTuples;
}

Relation::Relation() { }
