#include <random>
#include <random>
#include <chrono>
#include "dataset.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <sstream>

int Dataset::getX() {
    return domX;
}

int Dataset::getY() {
    return domY;
}

Relation Dataset::relation() {
    std::cout << "[Dataset Relation] " << data.size() << " tuples" << std::endl;
    return Relation(data.data(), data.size());
}

RandomDataset::RandomDataset(int domX, int domY, float probability) {
    this->domX = domX;
    this->domY = domY;
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

TxtFileDataset::TxtFileDataset(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
    }
    std::unordered_map<std::string, int> id_map_a;
    std::unordered_map<std::string, int> id_map_b;
    std::string line;
    int next_id_a = 0;
    int next_id_b = 0;
    int max_tuples = 3000000; // Temporary limit
    int tuple_count = 0;

    while (std::getline(file, line)) {
        if (tuple_count++ >= max_tuples) break;
        std::istringstream iss(line);
        std::string sa, sb;
        if (!(iss >> sa >> sb)) continue; // Skip invalid lines
        if (id_map_a.find(sa) == id_map_a.end())
            id_map_a[sa] = next_id_a++;
        if (id_map_b.find(sb) == id_map_b.end())
            id_map_b[sb] = next_id_b++;
        data.push_back({id_map_a[sa], id_map_b[sb]});
    }

    domX = next_id_a;
    domY = next_id_b;

    std::cout << "[TxtFileDataset] Loaded " << data.size() << " tuples from " << filename << ". [domX: " << domX << ", domY: " << domY << "]." << std::endl;

    file.close();
}