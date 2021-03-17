//
// Created by 51btn on 12.03.2021.
//

#ifndef NEWP_WEB_H
#define NEWP_WEB_H

#include <algorithm>
#include "Layer.h"
#include "../../ETL/headers/GeneralizedDataContainer.h"

class Web : public GeneralizedDataContainer{
public:
    Web(const std::vector<int>& Specification, double LearningRate, int InputSize, int NumberOfClasses);/*
 * needs vector of Specifications like how many neurons contain each layer
 * (example {10,10} it`s means that it will be created 3 layers with 2 hidden
 * layers each with 10 neurons and last layer that contain
 * amount neurons = NumberOfClasses), LearningRate using for making step to update weights,
 * Input size is the size of input data*/
    ~Web();

    std::vector<double> ForwardFeed(Data *);
    double TestPerformance();
    double ValidatePerformance();
    int Predict(Data *);

    void BackPropagation(Data *);
    void Train();


private:
    std::vector<Layer *> Layers;
    double LearningRate{};
    double Produce{};

};


#endif //NEWP_WEB_H
