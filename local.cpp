#include <iostream>
#include <vector>
#include "read.h"

using namespace std;

int main() {
    const double learn_rate = 0.08;
    const int epoch = 1000;

    vector<vector<double>> data;
    readCSV("E:\\Documents\\ECE1747\\project2\\data.csv", data);
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


    return 0;
}

