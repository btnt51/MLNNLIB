//
// Created by 51btn on 09.03.2021.
//

#include <ctime>
#include "../headers/Neuron.h"

Neuron::Neuron(int PreviousLayerSize) {
    InitializeWeights(PreviousLayerSize);
}

Neuron::~Neuron() {
    Weights.clear();
}

void Neuron::InitializeWeights(int PreviousLayerSize) {
    std::mt19937 Gen(std::time(nullptr)*PreviousLayerSize*PreviousLayerSize/10);
    std::normal_distribution<double> Distr(0.0,1.0);
    for(unsigned i = 0; i < PreviousLayerSize+1;i++)
        Weights.push_back(Distr(Gen));
}


