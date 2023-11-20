//
// Created by Winter on 2023-11-20.
//

#include "read.h"

using namespace std;

void readCSV(const string& filename, vector<vector<double>>& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file." << endl;
        return;
    }

    string line, cell;
    while (getline(file, line)) {
        vector<double> row;
        istringstream lineStream(line);

        while (getline(lineStream, cell, ',')) {
            row.push_back(stod(cell));
        }

        data.push_back(row);
    }

    file.close();
}