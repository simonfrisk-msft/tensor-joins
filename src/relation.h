#pragma once

struct Tuple {
    int x;
    int y;
};

class Relation {
public:
    int count;
    Tuple* data;
    Relation();
    Relation(Tuple* data, int count);
    void free();
    void print_gpu();
    void print_stats();
    Relation transferToDevice();
};
