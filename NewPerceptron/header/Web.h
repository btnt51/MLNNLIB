//
// Created by 51btn on 12.03.2021.
//

#ifndef NEWP_WEB_H
#define NEWP_WEB_H

#include "Layers.h"
#include "../../ETL/headers/GeneralizedDataContainer.h"

class Web : GeneralizedDataContainer{
public:
    Web(std::vector<int> Specification, double LearningRate, int InputSize, int NumberOfClasses);/*
 * needs vector of Specifications like how many neurons contain each layer
 * (example {10,10} it`s means that it will be created 3 layers with 2 hidden
 * layers each with 10 neurons and last layer that contain
 * amount neurons = NumberOfClasses), LearningRate using for making step to update weights,
 * Input size is the size of input data*/
    ~Web();

private:
    std::vector<Layers *> Layers;
    double LearningRate{};
    double Produce{};

};


#endif //NEWP_WEB_H
