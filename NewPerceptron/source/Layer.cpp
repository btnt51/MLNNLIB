//
// Created by 51btn on 12.03.2021.
//

#include "../header/Layer.h"

Layer::Layer(int PreviousLayerSize, int CurrentLayerSize) {
    for (int i = 0; i < CurrentLayerSize; ++i)          //filling layer with neurons
        Neurons.push_back(new Neuron(PreviousLayerSize));
    this->CurrentLayerSize = CurrentLayerSize; //setting size of current layer
}

Layer::~Layer() {
    for(auto &El : Neurons)
        delete El;  //deleting pointers on neurons(free memory)
    Neurons.clear();//clearing vector
}
