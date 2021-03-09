//
// Created by 51btn on 09.03.2021.
//

#ifndef NEURAL_NETWORK_LAYER_H
#define NEURAL_NETWORK_LAYER_H
#include <vector>
#include "Neuron.h"

class Layer {
public:
    Layer(int, int);
    ~Layer();

    int CurrentLayerSize{};
    std::vector<Neuron*> Neurons{};
    std::vector<double> LayerOutputs{};
};


#endif //NEURAL_NETWORK_LAYER_H
