//
// Created by 51btn on 09.03.2021.
//

#include "../headers/Layer.h"

Layer::Layer(int PreviousLayerSize, int CurrentLayerSize) {
    for(int i = 0; i < CurrentLayerSize; i++)
        Neurons.push_back(new Neuron(PreviousLayerSize));
    this->CurrentLayerSize = CurrentLayerSize;
}

Layer::~Layer() {
    for(auto &El : Neurons)
        delete El;
    Neurons.clear();
    LayerOutputs.clear();
}
