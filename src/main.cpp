#include <iostream>
#include "dataset.h"
#include "device_manager.h"
#include "join/mmul_join.h"
#include "join/naive_join.h"
#include "util.h"

int main() {
    int dom = 10000;

    Timer td("Creating random dataset");
    Dataset* hd1 = new RandomDataset(dom, dom, 0.5);
    Dataset* hd2 = new RandomDataset(dom, dom, 0.5);
    td.finish();

    Timer tt("Transfer to device");
    DeviceManager device;
    Relation dd1 = device.TransferDataToDevice(hd1);
    Relation dd2 = device.TransferDataToDevice(hd2);
    tt.finish();

    Timer t1("Naive Join");
    Naive_Join naive_join;
    naive_join.join(dd1, dd2);
    t1.finish();

    Timer t2("MMUL Join");
    MMUL_Join mmul_join(dom, dom, dom);
    mmul_join.join(dd1, dd2);
    t2.finish();

    delete hd1;
    delete hd2;
}

