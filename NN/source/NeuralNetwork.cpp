//
// Created by 51btn on 09.03.2021.
//

#include "../headers/NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int>& HiddenLayer, int InputSize, int NumberOfClasses, double LearningRate) {
    Layers.push_back(new Layer(InputSize, HiddenLayer.at(0)));
    for(int i = 1; i < HiddenLayer.size();++i)
        Layers.push_back(new Layer(Layers.at(i-1)->Neurons.size(), HiddenLayer.at(i)));
    Layers.push_back(new Layer(Layers.at(Layers.size()-1)->Neurons.size(), NumberOfClasses));
    this->LearningRate = LearningRate;
}

NeuralNetwork::~NeuralNetwork() {
    for(auto &El : Layers)
        delete El;
    Layers.clear();
}

double NeuralNetwork::ActivateNeuron(std::vector<double> Weights, std::vector<double> Input, int Fashion) {
    std::cout << "Weights size " << Weights.size() << " Input size " << Input.size() << "\n";
    //if(Weights.size()-1 == Input.size()) {
        double Activation = Weights.back(); //Bias neuron
        for (int i = 0; i < Weights.size()-1; ++i)
            Activation += Weights[i] * Input[i];
        switch (Fashion) {
            case 0: {
                return 1.0 / (1.0 + exp(-Activation));
            }
            case 1:{
                double Upper = exp(2*Activation) - 1.0;
                double Lower = exp(2*Activation) + 1.0;
                return Upper/Lower;
            }
            case 2:{
                return Activation;
            }
            default:{
                return 1.0 / (1.0 + exp(-Activation));
            }
        }

    /*} else{
        std::cout << "There is some problems with weights and input data in Neuron" << std::endl;
        exit(1);
    }*/
}

double NeuralNetwork::DeliveryBack(double Output) {
    return Output*(1-Output);
}

std::vector<double> NeuralNetwork::ForwardPropagation(Data *Data) {
    std::vector<double> Inputs = Data->GetNormalizedFeatureVector();
    std::vector<double> NewInputs{};

    for(auto TempLayer : Layers){
        for(auto &El : TempLayer->Neurons){
            El->Output = ActivateNeuron(El->Weights, Inputs);
            NewInputs.push_back(El->Output);
        }
    }
    Inputs = NewInputs;
    NewInputs.clear();
    return Inputs;
}

void NeuralNetwork::BackPropagation(Data *Data) {
    for(unsigned i = Layers.size() -1; i>=0;--i){
        Layer *TempLayer = Layers.at(i);
        std::vector<double> Errors;
        if(i != Layers.size()-1){
            for(int j = 0; j < TempLayer->Neurons.size();++j){
                double  Error = 0.0;
                for(auto &El : Layers.at(i+1)->Neurons){
                    Error += (El->Weights.at(j) * El->Delta);
                }
                Errors.push_back(Error);
            }
        } else{
            for(int j = 0; j < TempLayer->Neurons.size(); ++j){
                auto &El = TempLayer->Neurons.at(j);
                Errors.push_back(static_cast<double>(Data->GetClassVector().at(j)-El->Output)); //calculating expected - actual
            }
        }
        for(int j = 0; j < TempLayer->Neurons.size();++j){
            auto &El = TempLayer->Neurons.at(j);
            El->Delta = Errors.at(j) * this->DeliveryBack(El->Output); //calculating gradient for updating weights of nn
        }
    }
}

void NeuralNetwork::UpdateWeights(Data *Data) {
    std::vector<double> Inputs = Data->GetNormalizedFeatureVector();
    for(int i = 0; i < Layers.size();i++){
        if(i != 0){
            for(auto &El : Layers.at(i-1)->Neurons)
                    Inputs.push_back(El->Output);
        }

        for(auto &El : Layers.at(i)->Neurons){
            for(int j = 0; j < Inputs.size(); j++)
                El->Weights.at(j) += LearningRate * El->Delta * Inputs.at(j);

            El->Weights.back() += LearningRate * El->Delta;
        }
    }
    Inputs.clear();
}



void NeuralNetwork::Train(int NumberOfEpochs) {
    for(int i = 0; i < NumberOfEpochs;i++){
        double SumOfErrors{};
        for(auto &El : DataForTraining){
            std::vector<double> Outputs = ForwardPropagation(El);
            std::vector<int> Expected = El->GetClassVector();
            double TempSumOfErrors{};
            for(int j = 0; j < Outputs.size()-1; ++j)
                TempSumOfErrors += pow(static_cast<double>(Expected.at(j) - Outputs.at(j)),2);
            SumOfErrors +=TempSumOfErrors;
            BackPropagation(El);
            UpdateWeights(El);
        }
        std::cout << "Epoch " << i << "Error % = " << SumOfErrors << "\n";
    }
}

int NeuralNetwork::Predict(Data *Data) {
    std::vector<double> Outputs = ForwardPropagation(Data);
    return std::distance(Outputs.begin(), std::max(Outputs.begin(), Outputs.end()));
}

double NeuralNetwork::TestProduce() {
    double AmountOfCorrectPredictions{};
    int Counter{};
    for(auto &El : DataForTesting){
        Counter++;
        int Index = Predict(El);
        if(El->GetClassVector().at(Index)== 1)
            AmountOfCorrectPredictions++;
    }
    Performance = AmountOfCorrectPredictions / Counter;
    return Performance;
}

double NeuralNetwork::ValidateProduce() {
    double AmountOfCorrectPredictions{};
    int Counter{};
    for(auto &El : DataForValidation){
        Counter++;
        int Index = Predict(El);
        if(El->GetClassVector().at(Index)== 1)
            AmountOfCorrectPredictions++;
    }
    Performance = AmountOfCorrectPredictions / Counter;
    return Performance;
}


