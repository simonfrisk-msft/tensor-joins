#include "experiment.h"
#include <iostream>
#include "../dataset.h"
#include "../join/mmul_join.h"
#include "../join/naive_join.h"
#include "../util.h"

void tests() {
    std::cout << "-------------------- Running tests" << std::endl;
    // Test 1
    Tuple t1[4] = {Tuple{x:0,y:0},Tuple{x:0,y:1},Tuple{x:1,y:1},Tuple{x:1,y:2}};
    Relation rel1(t1, 4);
    Relation dr1 = rel1.transferToDevice();

    Tuple t2[3] = {Tuple{x:0,y:0},Tuple{x:1,y:1},Tuple{x:2,y:1}};
    Relation rel2(t2, 3);
    Relation dr2 = rel2.transferToDevice();

    MMUL_Join joinObj(3, 3, 3);
    Relation output = joinObj.join(dr1, dr2);

    if(output.count != 3)
        std::cout << "[Test1] Wrong output count: expected 3, receieved " << output.count << std::endl;
    else
        std::cout << "[Test1] Ok" << std::endl;
    output.free();

    std::cout << "-------------------- End of tests" << std::endl;
}
