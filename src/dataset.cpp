#include <random>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "dataset.h"

Relation Dataset::relation() {
    std::cout << "[Dataset Relation] " << data.size() << " tuples" << std::endl;
    return Relation(data.data(), data.size());
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
                data.push_back(Tuple { x: i, y: j });
            }
        }
    }
}
