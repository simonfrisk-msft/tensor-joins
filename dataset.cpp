#include <random>
#include <iostream>
#include <tuple>
#include <vector>
#include <random>
#include <chrono>

#include "dataset.h"

void Dataset::print_summary() {
    std::cout << "Dataset: " << data.size() << " tuples" << std::endl;
}

void Dataset::print_data(int maxCount) {
    std::cout << "-----------------------------" << std::endl;
    int i = 0;
    for (std::tuple<int,int> tuple : data) {
        auto [x, y] = tuple;
        std::cout << "(" << x << ", " << y << ")" << std::endl;
        i++;
        if(i == maxCount) {
            std::cout << "..." << std::endl;
            break;
        }
    }
    std::cout << "-----------------------------" << std::endl;
}

int Dataset::tuple_count() {
    return data.size();
}

int Dataset::size_bytes() {
    return data.size() * sizeof(std::tuple<int,int>);
}

RandomDataset::RandomDataset(int domX, int domY, float probability) {
    // Set up random generator
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    // Generate tuples
    for (int i = 0; i < domX; i++) {
        for (int j = 0; j < domY; j++) {
            float r = distribution(generator);
            if(r <= probability) {
                data.push_back(std::make_tuple(i,j));
            }
        }
    }
}
