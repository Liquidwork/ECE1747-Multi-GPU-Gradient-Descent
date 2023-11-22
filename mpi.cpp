//
// Created by Winter on 2023-11-22.
//
#include <iostream>
#include <vector>
#include <mpi.h>
#include "read.h"

using namespace std;

const double learn_rate = 0.08;
const int epoch = 1000;

int main(int argc, char *argv[]) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int shape[2];
    double *sendbuf;
    vector<double> data;

    if(rank == 0){
        readCSV("E:\\Documents\\ECE1747\\project2\\data.csv", data, shape);

        cout << "Rows: " << shape[0] << endl;
        cout << "Columns: " << shape[1] << endl;
    }
    MPI_Bcast(shape, 2, MPI_INT, 0, MPI_COMM_WORLD);

    int row = shape[0] / size;
    vector<double> recvbuf(row * shape[1]);

    MPI_Scatter(data.data(), row * shape[1], MPI_DOUBLE, recvbuf.data(),
                row * shape[1], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> params(shape[1]);
    std::vector<double> grad(shape[1]), g_grad(shape[1]);
    double mse, g_mse;

    std::fill(params.begin(), params.end(), 0); // params start from 0

    for (int e = 0; e < epoch; e++){ // epoch of 1000 full batch

        MPI_Bcast(params.data(), shape[1], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::fill(grad.begin(), grad.end(), 0); // reset grad array
        mse = 0;

        for (int i = 0; i < row * shape[1]; i += shape[1]){
            double predict = 0.;
            for (int j = 0; j < shape[1] - 1; j++){
                predict += params[j] * recvbuf[i + j];
            }
            predict += params[shape[1] - 1]; // Constant term
            double error = predict - recvbuf[i + shape[1] - 1];
            mse += error * error / shape[0];

            for (int j = 0; j < shape[1] - 1; j++){
                grad[j] += error * recvbuf[i + j] / shape[0];
            }
            grad[shape[1] - 1] += error / shape[0];
        }

        MPI_Reduce(grad.data(), g_grad.data(), shape[1], MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mse, &g_mse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0){
            cout << "Epoch: " << e + 1 << "/" << epoch << ", Params: [";

            for (int j = 0; j < shape[1]; j++){
                cout << params[j] << ",";
                params[j] -= learn_rate * g_grad[j];
            }
            cout << "], MSE: " << g_mse << endl;
        }

    }

    MPI_Finalize();

}