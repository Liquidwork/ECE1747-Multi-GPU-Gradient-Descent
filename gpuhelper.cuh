//
// Created by Winter on 2023-11-28.
//

#ifndef PROJECT2_GPUHELPER_CUH
#define PROJECT2_GPUHELPER_CUH

#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "read.h"

#define THREADS_PER_BLOCK 512

class gpuhelper{
private:
    double *d_data, *d_params;
    int *d_shape, *h_shape, block;
    double *d_partial_mse, *d_partial_gradient;

public:
    gpuhelper(vector<double> &data, int *shape);
    ~gpuhelper();

    double gradientCalculate(std::vector<double> &params, std::vector<double> &gradient);
};

#endif //PROJECT2_GPUHELPER_CUH
