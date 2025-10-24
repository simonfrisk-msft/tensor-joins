#include "experiment.h"
#include <iostream>
#include "../dataset.h"
#include "../join/mmul_join.h"
#include "../join/naive_join.h"
#include "../join/hybrid_join.h"
#include "../join/csr_join.h"
#include "../util.h"

void tests() {
    std::cout << "-------------------- Running tests" << std::endl;

    // Test 1 -----------------
    Tuple<2> t1[4] = {Tuple<2>{{0, 0}},Tuple<2>{{0, 1}},Tuple<2>{{1, 1}},Tuple<2>{{1, 2}}};
    Relation<2> rel1(t1, 4);
    Relation<2> dr1 = rel1.transferToDevice();

    Tuple<2> t2[3] = {Tuple<2>{{0, 0}},Tuple<2>{{1, 1}},Tuple<2>{{2, 1}}};
    Relation<2> rel2(t2, 3);
    Relation<2> dr2 = rel2.transferToDevice();

    // MMUL Join
    MMUL_Join joinObj1(3, 3, 3);
    Relation<2> output = joinObj1.join(dr1, dr2);
    if(output.count != 3)
        std::cout << "[Test1 MMUL] Wrong output count: expected 3, receieved " << output.count << std::endl;
    else
        std::cout << "[Test1 MMUL] Ok" << std::endl;
    output.free_gpu();
    // Naive Join
    Naive_Join joinObj2;
    output = joinObj2.join(dr1, dr2);
    if(output.count != 3)
        std::cout << "[Test1 Naive] Wrong output count: expected 3, receieved " << output.count << std::endl;
    else
        std::cout << "[Test1 Naive] Ok" << std::endl;
    output.free_gpu();
    // CuSparse Join
    CSR_Join joinObj3(3, 3, 3);
    output = joinObj3.join(dr1, dr2);
    if(output.count != 3)
        std::cout << "[Test1 CuSparse] Wrong output count: expected 3, receieved " << output.count << std::endl;
    else
        std::cout << "[Test1 CuSparse] Ok" << std::endl;
    output.free_gpu();
    // Hybrid Join
    Hybrid_Join joinObj4(3, 3, 3);
    output = joinObj4.join(dr1, dr2);
    if(output.count != 3)
        std::cout << "[Test1 Hybrid] Wrong output count: expected 3, receieved " << output.count << std::endl;
    else
        std::cout << "[Test1 Hybrid] Ok" << std::endl;
    output.free_gpu();

    std::cout << "-------------------- End of tests" << std::endl;
}
