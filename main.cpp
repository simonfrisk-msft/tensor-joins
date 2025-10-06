#include <iostream>
#include "dataset.h"
#include "device_manager.h"
#include "util.h"


int main() {
    Timer td("Creating random dataset");
    Dataset* hd1 = new RandomDataset(10000, 10000, 0.5);
    hd1->print_summary();
    Dataset* hd2 = new RandomDataset(10000, 10000, 0.5);
    hd2->print_summary();
    td.finish();

    Timer tt("Transfer to device");
    DeviceManager device;
    Relation dd1 = device.TransferDataToDevice(hd1);
    Relation dd2 = device.TransferDataToDevice(hd2);
    tt.finish();

    Timer t1("Naive Join");
    device.NaiveJoin(dd1,dd2, hd1->tuple_count(), hd2->tuple_count());
    t1.finish();

    delete hd1;
    delete hd2;
}

