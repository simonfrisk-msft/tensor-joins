#include "experiment.h"
#include <iostream>
#include "../dataset.h"
#include "../join/mmul_join.h"
#include "../join/naive_join.h"
#include "../join/hybrid_join.h"
#include "../join/csr_join.h"
#include "../util.h"

void run_dense(int dom1, int dom2, int dom3) {
    std::cout << "------------------ Domains: " << dom1 << ", " << dom2 << ", " << dom3 << std::endl;

    std::vector<float> probs;
    probs.push_back(1.0);
    /*probs.push_back(0.3);
    probs.push_back(0.1);
    probs.push_back(0.03);
    probs.push_back(0.01);
    probs.push_back(0.003);
    probs.push_back(0.001);
    probs.push_back(0.0003);
    probs.push_back(0.0001);
    probs.push_back(0.00003);
    probs.push_back(0.00001);*/

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

        CSR_Join joinObj(dom1, dom2, dom3);
        Relation out = joinObj.join(dd1, dd2);
        out.print_stats();
        out.free();

        dd1.free();
        dd2.free();

        cudaDeviceReset();
    }
}

void run_txt(const char* file1) {
    std::cout << "--- File: " << file1 << std::endl;

    Timer td("Creating dataset from txt file");
    TxtFileDataset hd1(file1);
    TxtFileDataset hd2(file1);
    td.finish();
    Timer tt("Transfer to device");
    Relation rel1 = hd1.relation().transferToDevice();
    Relation rel2 = hd2.relation().transferToDevice();
    rel1.print_stats();
    tt.finish();

    Naive_Join joinObj;//(hd1.getX(), (hd1.getY() > hd1.getX() ? hd1.getY() : hd1.getX()), hd1.getY());
    Relation out = joinObj.join(rel1, rel2);
    out.print_stats();
    out.free();

    rel1.free();
    rel2.free();

    cudaDeviceReset();
}

void dense_experiment() {
    run_txt("./data/gplus_combined.txt");

    // TODO make into a test
    /*Tuple t1[4] = {Tuple{x:0,y:0},Tuple{x:0,y:1},Tuple{x:0,y:2},Tuple{x:2,y:2}};
    Relation rel1(t1, 4);
    Relation dr1 = rel1.transferToDevice();
    dr1.print_gpu();

    Tuple t2[3] = {Tuple{x:0,y:0},Tuple{x:1,y:1},Tuple{x:2,y:2}};
    Relation rel2(t2, 3);
    Relation dr2 = rel2.transferToDevice();
    dr2.print_gpu();

    Hybrid_Join joinObj(3, 3, 3);
    Relation output = joinObj.join(dr1, dr2);
    output.print_gpu();*/
}
