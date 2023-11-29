//
// Created by Winter on 2023-11-28.
//

//
// Created by Winter on 2023-11-22.
//
#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include "read.h"
#include "gpuhelper.cuh"

using namespace std;

const double learn_rate = 0.08;
const int epoch = 1000;

int main(int argc, char *argv[]) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    chrono::time_point<std::chrono::high_resolution_clock> start, stop;

    int shape[2];
    double *sendbuf;
    vector<double> data;

    if(rank == 0){
        readCSV("../data.csv", data, shape);

        cout << "Rows: " << shape[0] << endl;
        cout << "Columns: " << shape[1] << endl;

        start = chrono::high_resolution_clock::now();
    }
    MPI_Bcast(shape, 2, MPI_INT, 0, MPI_COMM_WORLD);

    shape[0] = shape[0] / size;
    vector<double> recvbuf(shape[0] * shape[1]);

    MPI_Scatter(data.data(), shape[0] * shape[1], MPI_DOUBLE, recvbuf.data(),
                shape[0] * shape[1], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> params(shape[1]);
    std::vector<double> grad(shape[1]), g_grad(shape[1]);
    double mse, g_mse;

    std::fill(params.begin(), params.end(), 0); // params start from 0

    gpuhelper gpu(recvbuf, shape);

    for (int e = 0; e < epoch; e++){ // epoch of 1000 full batch

        MPI_Bcast(params.data(), shape[1], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        mse = gpu.gradientCalculate(params, grad);
        MPI_Reduce(&mse, &g_mse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(grad.data(), g_grad.data(), shape[1], MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0){
            g_mse /= size;
            cout << "Epoch: " << e + 1 << "/" << epoch << ", Params: [";

            for (int j = 0; j < shape[1]; j++){
                g_grad[j] /= size;
                cout << params[j] << ",";
                params[j] -= learn_rate * g_grad[j];
            }
            cout << "], MSE: " << g_mse << endl;
        }

    }
    if (rank == 0){
        stop = chrono::high_resolution_clock::now();
        double time = (double) chrono::duration_cast<chrono::nanoseconds>(stop - start).count() * 1e-9;
        cout << "Total time taken: " << time << "s." << endl;
    }


    MPI_Finalize();

}
