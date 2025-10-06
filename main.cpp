#include <iostream>
#include "dataset.h"

int main() {

    Dataset* data = new RandomDataset(100, 100, 0.5);
    data->print_summary();
    data->print_data(10);
}

