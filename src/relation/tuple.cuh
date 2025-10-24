#pragma once
#include <cstdio>

// Arity n tuple
template <int n>
struct Tuple {
    int values[n];
};

template <int n>
__host__ __device__ inline
bool operator==(const Tuple<n>& a, const Tuple<n>& b) {
    bool same = true;
    for (int i = 0; i < n; i++) {
        if (a.values[i] != b.values[i]) {
            same = false;
            break;
        }
    }
    return same;
}

template <int n>
__host__ __device__ inline
bool operator<(const Tuple<n>& a, const Tuple<n>& b) {
    for (int i = 0; i < n; i++) {
        if (a.values[i] < b.values[i]) {
            return true;
        } else if (a.values[i] > b.values[i]) {
            return false;
        }
    }
    return false; // they are equal
}
