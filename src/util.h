#pragma once
#include <chrono>
#include <iostream>

class Timer {
private:
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer(const char* name);
    void finish();
};
