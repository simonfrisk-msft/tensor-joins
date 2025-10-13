#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "../relation.h"

void deduplicate(Relation relation) {
    thrust::device_ptr<Tuple> begin(relation.data);
    thrust::device_ptr<Tuple> end(relation.data + relation.count);

    thrust::sort(begin, end);
    thrust::device_ptr<Tuple> new_end = thrust::unique(begin, end);

    relation.count = new_end - begin;
}
