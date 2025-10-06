#include "util.h"

Timer::Timer(const char* timer_name) {
    name = std::string(timer_name);
    start = std::chrono::high_resolution_clock::now();
}

void Timer::finish() {
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Timer (" << name << "): " << duration.count() << "s" << std::endl;
}