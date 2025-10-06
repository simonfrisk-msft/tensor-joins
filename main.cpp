#include <iostream>
#include "dataset.h"
#include "device_manager.h"


int main() {
    Dataset* hd1 = new RandomDataset(100, 100, 0.5);
    hd1->print_summary();
    hd1->print_data(10);
    Dataset* hd2 = new RandomDataset(100, 100, 0.5);
    hd2->print_summary();
    hd2->print_data(10);
    DeviceManager device;
    device.Echo();
    Relation dd1 = device.TransferDataToDevice(hd1);
    Relation dd2 = device.TransferDataToDevice(hd2);
    device.PrintRelation(dd1, 10);
    device.PrintRelation(dd2, 10);

    delete hd1;
    delete hd2;
}

