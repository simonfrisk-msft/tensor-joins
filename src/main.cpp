#include <iostream>
#include "dataset.h"
#include "join/mmul_join.h"
#include "join/naive_join.h"
#include "util.h"

int main() {
    int dom = 10000;

    Timer td("Creating random dataset");
    RandomDataset hd1(dom, dom, 1.0);
    RandomDataset hd2(dom, dom, 1.0);
    td.finish();
    Timer tt("Transfer to device");
    Relation dd1 = hd1.relation().transferToDevice();
    Relation dd2 = hd2.relation().transferToDevice();
    tt.finish();

    Timer t2("MMUL Join");
    MMUL_Join mmul_join(dom, dom, dom);
    Relation mmul = mmul_join.join(dd1, dd2);
    t2.finish();
    mmul.print_stats();
    mmul.free();

    dd1.free();
    dd2.free();
}

