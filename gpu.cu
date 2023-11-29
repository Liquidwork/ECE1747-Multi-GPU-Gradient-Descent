//
// Created by Winter on 2023-11-20.
//

#include <iostream>
#include <vector>
#include <chrono>
#include "read.h"

#define THREADS_PER_BLOCK 512

using namespace std;

__global__ void compute(double *d_data, double *params, int *shape, double *d_mse, double *d_gradient) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ double tmp[];
    double *line = &tmp[threadIdx.x * (shape[1] + 1)];

    double *row = &d_data[index * shape[1]];

    if (index >= shape[0]){
//        d_mse[blockIdx.x] = 0.;
//        for (int i = 0; i < shape[1]; i += 0){
//            d_gradient[blockIdx.x * shape[1] + i] = 0.;
//        }
        return;
    }

    double predict = 0.;

    for(int i = 0; i < shape[1] - 1; i++){
        predict += row[i] * params[i];
    }
    predict += params[shape[1] - 1];
    double error = predict - row[shape[1] - 1];
    line[shape[1]] = error * error / shape[0]; // mse
    for (int i = 0; i < shape[1] - 1; i++){
        line[i] = error * row[i] / shape[0]; // gradient
    }
    line[shape[1] - 1] = error / shape[0]; // gradient last term
    __syncthreads();


    // calculate partial sum
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x >= stride && threadIdx.x < 2 * stride) {
            for (int i = 0; i < shape[1] + 1; i++){
                tmp[(threadIdx.x - stride) * (shape[1] + 1) + i] += line[i]; // sum grad and mse
            }
        }
        __syncthreads();
    }
    // Write the result of this block to global memory
    if (threadIdx.x == 0) {
        d_mse[blockIdx.x] = line[shape[1]];
        for (int i = 0; i < shape[1]; i++){
            d_gradient[blockIdx.x * shape[1] + i] = line[i];
        }
    }
}

int main(){
    const double learn_rate = 0.08;
    const int epoch = 1000;
    chrono::time_point<std::chrono::high_resolution_clock> start, stop;

    vector<double> data;
    int shape[2];
    readCSV("./data.csv", data, shape);

    cout << "Rows: " << shape[0] << endl;
    cout << "Columns: " << shape[1] << endl;
    double *d_data, *d_params;

    start = chrono::high_resolution_clock::now();

    double params[shape[1]] = {0};
    int *d_shape;
    cudaMalloc((void**)&d_data, shape[0] * shape[1] * sizeof(double));
    cudaMemcpy(d_data, data.data(), shape[0] * shape[1] * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_params, shape[1] * sizeof(double));

    cudaMalloc((void**)&d_shape, 2 * sizeof(int));
    cudaMemcpy(d_shape, shape, 2 * sizeof(int), cudaMemcpyHostToDevice);

    int block = (shape[0] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    double *d_partial_mse, *d_partial_gradient;

    cudaMalloc((void**)&d_partial_mse, block * sizeof(double));
    cudaMalloc((void**)&d_partial_gradient, block * shape[1] * sizeof(double));

    double mse = 0;
    std::vector<double> gradient(shape[1], 0.);

    for(int i = 0; i < epoch; i++){
        cudaMemcpy(d_params, params, shape[1] * sizeof(double), cudaMemcpyHostToDevice);
        compute <<< block, THREADS_PER_BLOCK, THREADS_PER_BLOCK * (shape[1] + 1) * sizeof(double) >>>
                (d_data, d_params, d_shape, d_partial_mse, d_partial_gradient);

        double partial_mse[block], partial_gradient[block * shape[1]];

        cudaMemcpy(partial_mse, d_partial_mse, block * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(partial_gradient, d_partial_gradient, block * shape[1] * sizeof(double), cudaMemcpyDeviceToHost);

        mse = 0;
        for (int j = 0; j < block; j++){
            mse += partial_mse[j];
            for (int k = 0; k < shape[1]; k++){
                gradient[k] += partial_gradient[j * shape[1] + k];
            }

        }
        cout << "Epoch: " << i + 1 << "/" << epoch << ", Params: [";
        for (int k = 0; k < shape[1]; k++){
            cout << params[k] << ",";
            params[k] -= learn_rate * gradient[k];
            gradient[k] = 0.;
        }
        cout << "], MSE: " << mse << endl;
    }

    cudaFree(d_data);
    cudaFree(d_params);
    cudaFree(d_shape);
    cudaFree(d_partial_mse);
    cudaFree(d_partial_gradient);

    stop = chrono::high_resolution_clock::now();
    double time = (double) chrono::duration_cast<chrono::nanoseconds>(stop - start).count() * 1e-9;
    cout << "Total time taken: " << time << "s." << endl;
}
