//
// Created by 51btn on 09.03.2021.
//

#ifndef ML_NEURON_H
#define ML_NEURON_H

#include <vector>
#include <cmath>
#include <random>

class Neuron {
public:
    Neuron(int);
    ~Neuron();
    void InitializeWeights(int);

    double Output{};
    double Delta{};
    std::vector<double> Weights{};

};


#endif //ML_NEURON_H
