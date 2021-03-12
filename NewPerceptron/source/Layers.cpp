//
// Created by 51btn on 12.03.2021.
//

#include "../header/Layers.h"

Layers::Layers(int PreviousLayerSize, int CurrentLayerSize) {
    for (int i = 0; i < CurrentLayerSize; ++i)
        Neurons.push_back(new Neuron(PreviousLayerSize));
    this->CurrentLayerSize = CurrentLayerSize;
}

Layers::~Layers() {
    for(auto &El : Neurons)
        delete El;
    Neurons.clear();
}
