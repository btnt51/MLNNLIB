//
// Created by 51btn on 12.03.2021.
//

#ifndef NEWP_LAYERS_H
#define NEWP_LAYERS_H

#include "Neuron.h"

class Layers {
public:
    Layers(int,int);                        //needs size of previous layer and of current layer
    ~Layers();

    int GetCurrentLayerSize()                   {   return CurrentLayerSize; }
    std::vector<Neuron *> GetNeurons()          {   return Neurons; }

private:
    std::vector<Neuron *> Neurons;          //handling neurons
    int CurrentLayerSize{};                 //handling size of current layer
};


#endif //NEWP_LAYERS_H
