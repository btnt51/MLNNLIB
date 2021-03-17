//
// Created by 51btn on 12.03.2021.
//
#include "../header/Neuron.h"

Neuron::Neuron(int PreviousLayerSize) {
    std::random_device RD;  //random device for generating a random seed for mt19937
    std::mt19937 Gen(RD()); //generator for generating weights of neurons
    std::uniform_real_distribution<double> UID(0.0001, 1.000); //limiter for generator
    for(int i = 0; i < PreviousLayerSize; i++)
        Weights.push_back(UID(Gen));    //filling vector of weights of neurons from previous layer
}

Neuron::~Neuron() {
    Weights.clear();
}

void Neuron::Activation() {
    Output = 1.0 / (1.0 - exp(-Input)); //simple sigmoid function
}



