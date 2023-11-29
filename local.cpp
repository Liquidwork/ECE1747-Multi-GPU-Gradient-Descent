#include <iostream>
#include <vector>
#include <chrono>
#include "read.h"

using namespace std;

int main() {
    const double learn_rate = 0.08;
    const int epoch = 1000;
    chrono::time_point<std::chrono::high_resolution_clock> start, stop;

    vector<vector<double>> data;
    readCSV("./data.csv", data);
    unsigned int size[2];
    cout << "Rows: " << data.size() << endl;
    if (data.empty()) {
        cerr << "Data is empty" << endl;
        return 1;
    }
    cout << "Columns: " << data[0].size() << endl;
    size[0] = data.size();
    size[1] = data[0].size();

    std::vector<double> params(size[1]);
    std::vector<double> grad(size[1]);
    double mse;

    std::fill(params.begin(), params.end(), 0); // params start from 0

    start = chrono::high_resolution_clock::now();

    for (int i = 0; i < epoch; i++){ // epoch of 1000 full batch

        std::fill(grad.begin(), grad.end(), 0); // reset grad array
        mse = 0;

        for (const auto &row : data){
            double predict = 0.;
            for (int j = 0; j < size[1] - 1; j++){
                predict += params[j] * row[j];
            }
            predict += params[size[1] - 1]; // Constant term
            double error = predict - row[size[1] - 1];
            mse += error * error / size[0];

            for (int j = 0; j < size[1] - 1; j++){
                grad[j] += error * row[j] / size[0];
            }
            grad[size[1] - 1] += error / size[0];
        }

        cout << "Epoch: " << i + 1 << "/" << epoch << ", Params: [";

        for (int j = 0; j < size[1]; j++){
            cout << params[j] << ",";
            params[j] -= learn_rate * grad[j];
        }
        cout << "], MSE: " << mse << endl;

    }

    stop = chrono::high_resolution_clock::now();
    double time = (double) chrono::duration_cast<chrono::nanoseconds>(stop - start).count() * 1e-9;
    cout << "Total time taken: " << time << "s." << endl;

    return 0;
}

