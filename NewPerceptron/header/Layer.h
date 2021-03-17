//
// Created by 51btn on 12.03.2021.
//

#ifndef NEWP_LAYER_H
#define NEWP_LAYER_H

#include "Neuron.h"

class Layer {
public:
    Layer(int, int);                        //needs size of previous layer and of current layer
    ~Layer();

    int GetCurrentLayerSize()                   {   return CurrentLayerSize; }
    std::vector<Neuron *> GetNeurons()          {   return Neurons; }

private:
    std::vector<Neuron *> Neurons;          //handling neurons
    int CurrentLayerSize{};                 //handling size of current layer
};


#endif //NEWP_LAYER_H
