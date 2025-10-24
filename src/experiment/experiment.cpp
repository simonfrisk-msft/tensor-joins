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
    probs.push_back(0.3);
    probs.push_back(0.1);
    probs.push_back(0.03);
    probs.push_back(0.01);
    /*probs.push_back(0.003);
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
        Relation<2> dd1 = hd1.relation().transferToDevice();
        Relation<2> dd2 = hd2.relation().transferToDevice();
        tt.finish();

        MMUL_Join joinObj(dom1, dom2, dom3);
        Relation<2> out = joinObj.join(dd1, dd2);
        out.print_stats();
        out.free_gpu();

        dd1.free_gpu();
        dd2.free_gpu();

        cudaDeviceReset();
    }
}

void run_txt(const char* file1) {
    std::cout << "--- File: " << file1 << std::endl;

    Timer td("Creating dataset from txt file");
    TxtFileDataset hd1(file1, 1000000);
    TxtFileDataset hd2(file1, 1000000);
    td.finish();
    Timer tt("Transfer to device");
    Relation<2> rel1 = hd1.relation().transferToDevice();
    Relation<2> rel2 = hd2.relation().transferToDevice();
    rel1.print_stats();
    tt.finish();

    MMUL_Join joinObj(hd1.getX(), (hd1.getY() > hd1.getX() ? hd1.getY() : hd1.getX()), hd1.getY());
    Relation<2> out = joinObj.join(rel1, rel2);
    out.print_stats();
    out.free_gpu();

    rel1.free_gpu();
    rel2.free_gpu();

    cudaDeviceReset();
}

void dense_experiment() {
    run_dense(1000, 1000, 1000);
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
