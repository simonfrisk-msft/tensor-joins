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

Relation<2> Dataset::relation() {
    std::cout << "[Dataset Relation] " << data.size() << " tuples" << std::endl;
    return Relation<2>(data.data(), data.size());
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
                data.push_back(Tuple<2> {{i, j}});
            }
        }
    }
}

TxtFileDataset::TxtFileDataset(const char* filename, int max_rows) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
    }
    std::unordered_map<std::string, int> id_map;
    std::string line;
    int next_id = 0;
    int tuple_count = 0;

    while (std::getline(file, line)) {
        if (tuple_count++ >= max_rows) break;
        std::istringstream iss(line);
        std::string sa, sb;
        if (!(iss >> sa >> sb)) continue; // Skip invalid lines
        if (id_map.find(sa) == id_map.end())
            id_map[sa] = next_id++;
        if (id_map.find(sb) == id_map.end())
            id_map[sb] = next_id++;
        data.push_back({id_map[sa], id_map[sb]});
    }

    domX = next_id;
    domY = next_id;

    std::cout << "[TxtFileDataset] Loaded " << data.size() << " tuples from " << filename << ". [domX: " << domX << ", domY: " << domY << "]." << std::endl;

    file.close();
}