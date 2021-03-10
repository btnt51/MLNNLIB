//
// Created by 51btn on 09.03.2021.
//

#ifndef NEURAL_NETWORK_NEURALNETWORK_H
#define NEURAL_NETWORK_NEURALNETWORK_H

#include <numeric>
#include "Layer.h"
#include "../../ETL/headers/GeneralizedDataContainer.h"

class NeuralNetwork : public GeneralizedDataContainer {
public:
    NeuralNetwork(const std::vector<int>& spec, int, int, double);
    ~NeuralNetwork();

    double ActivateNeuron(std::vector<double>, std::vector<double> , int Fashion = 0);/*Fashion == 0 is Sigmoid,
    Fashion == 1 is Hyperbolic tangent Fashion == 2 is linear function*/
    double DeliveryBack(double); //using for back propagation
    double TestProduce();
    double ValidateProduce();

    int Predict(Data *);

    std::vector<double> ForwardPropagation(Data *);

    void BackPropagation(Data *);
    void UpdateWeights(Data *);
    void Train(int);
    void TrainWhile();

private:
    double LearningRate{};
    double Performance{};
    std::vector<Layer*> Layers{};

};


#endif //NEURAL_NETWORK_NEURALNETWORK_H
