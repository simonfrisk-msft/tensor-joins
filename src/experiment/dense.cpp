#include "experiment.h"
#include <iostream>
#include "../dataset.h"
#include "../join/mmul_join.h"
#include "../join/naive_join.h"
#include "../util.h"

void run(int dom1, int dom2, int dom3) {
    std::cout << "------------------ Domains: " << dom1 << ", " << dom2 << ", " << dom3 << std::endl;

    std::vector<float> probs;
    probs.push_back(1.0);
    //probs.push_back(0.3);
    //probs.push_back(0.1);
    //probs.push_back(0.03);
    //probs.push_back(0.01);
    //probs.push_back(0.003);
    //probs.push_back(0.001);
    //probs.push_back(0.0003);
    //probs.push_back(0.0001);

    for (float p : probs) {
        std::cout << "------------------ Probability " << p << std::endl;

        Timer td("Creating random dataset");
        RandomDataset hd1(dom1, dom2, p);
        RandomDataset hd2(dom2, dom3, p);
        td.finish();
        Timer tt("Transfer to device");
        Relation dd1 = hd1.relation().transferToDevice();
        Relation dd2 = hd2.relation().transferToDevice();
        tt.finish();

        MMUL_Join mmul_join(dom1, dom2, dom3);
        Relation mmul = mmul_join.join(dd1, dd2);
        mmul.print_stats();
        mmul.free();

        dd1.free();
        dd2.free();

        cudaDeviceReset();
    }
}

void dense_experiment() {
    run(100, 100, 100);
}
