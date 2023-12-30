#ifndef KP_SSAA_H
#define KP_SSAA_H

#include "vector.hpp"
#include <vector>
#include <iostream>

struct SSAA_params {
    int threadIdxX;
    int threadIdxY;
};



void cpu_ssaa(std::vector <Vec3f> &data, int width, int height, int coef, int n_threads_cpu);

void* cpu_ssaa_kernel(void *dummyPtr);

#endif //KP_SSAA_H
