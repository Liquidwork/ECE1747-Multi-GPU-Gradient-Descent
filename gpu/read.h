//
// Created by Winter on 2023-11-20.
//

#ifndef PROJECT2_READ_H
#define PROJECT2_READ_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

void readCSV(const string& filename, vector<vector<double>>& data);
void readCSV(const string& filename, vector<double>& data, int shape[]);

#endif //PROJECT2_READ_H
