//
// Created by 51btn on 12.03.2021.
//

#ifndef ML_NEURON_H
#define ML_NEURON_H

#include <vector>
#include <random>

class Neuron{
public:
    Neuron(int); //need size of a previous layer
    ~Neuron();

    void Activation();  //typical sigmoid
    void SetInput(double NewInput)                      {   Input = NewInput; }
    void SetDouble(double NewDelta)                     {   Delta = NewDelta; }
    void SetWeights(std::vector<double> NewWeights)     {   Weights = NewWeights; }
    double GetOutput()                                  {   return Output; }
    double GetDelta()                                   {   return Delta; }
    std::vector<double> GetWeights()                    {   return Weights; }

private:
    std::vector<double> Weights{};  //handling Weights of Neurons from previous layer
    double Input{};                 //handling Input(it is weight of Neuron and his output)
    double Output{};                //handling Output of current Neuron(calculating in Activation by sigmoid function)
    double Delta{};                 //handling data for correcting weights of Neurons
};

#endif //ML_NEURON_H
