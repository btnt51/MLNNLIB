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
    //std::cout << "Weights size " << Weights.size() << " Input size " << Input.size() << "\n";
    double Activation = Weights.back(); //Bias neuron
    //std::cout << "activation start sum" << std::endl;
    for (int i = 0; i < Weights.size()-1; i++)
        Activation += Weights[i] * Input[i];
    //std::cout << "activation end of sum" << std::endl;
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
}

double NeuralNetwork::DeliveryBack(double Output) {
    return Output*(1-Output);
}

std::vector<double> NeuralNetwork::ForwardPropagation(Data *Data) {
    std::vector<double> Inputs = Data->GetNormalizedFeatureVector();
    //std::vector<double> NewInputs{};

    for(auto TempLayer : Layers){
        std::vector<double> NewInputs{};
        //std::cout << "start of Layers" << std::endl;
        for(auto &El : TempLayer->Neurons){
           // std::cout << "start of activation of each neuron" << std::endl;
            El->Output = ActivateNeuron(El->Weights, Inputs);
         //   std::cout << " middle of activation of each neuron" << std::endl;
            NewInputs.push_back(El->Output);
       //     std::cout << "end of activation of each neuron" << std::endl << "---------------------\n";
        }
     //   std::cout << "end of Layers" << std::endl;
        Inputs = NewInputs;
        NewInputs.clear();
    }
   // std::cout << "Is it here?" << std::endl;
    return Inputs;
}

void NeuralNetwork::BackPropagation(Data *Data) {
    //std::cout << "BP dbg msg 1 \n";
    //for(unsigned i = Layers.size() - 1; i>=0;i--)
    unsigned  i = Layers.size()-1;
    while(0 <= i && i > Layers.size()){
      //  std::cout << "BP dbg msg 2 iteration #" << i << "\n";
        Layer *TempLayer = Layers.at(i);
        std::vector<double> Errors;
        if(i != Layers.size() - 1){
        //    std::cout << "BP dbg msg 3 \n";
            for(int j = 0; j < TempLayer->Neurons.size();++j){
          //      std::cout << "BP dbg msg 3 \n";
                double  Error = 0.0;
                for(auto &El : Layers.at(i+1)->Neurons){
            //        std::cout << "BP dbg msg 4 \n";
                    Error += (El->Weights.at(j) * El->Delta);
              //      std::cout << "BP dbg msg 5 \n";
                }
                Errors.push_back(Error);
                //std::cout << "BP dbg msg 6 \n";
            }
        } else{
            //std::cout << "BP dbg msg 7 \n";
            for(int j = 0; j < TempLayer->Neurons.size(); ++j){
              //  std::cout << "BP dbg msg 8 \n";
                auto &El = TempLayer->Neurons.at(j);
                Errors.push_back(static_cast<double>(Data->GetClassVector().at(j) - El->Output)); //calculating expected - actual
                //std::cout << "BP dbg msg 9 \n";
            }
           // std::cout << "BP dbg msg 10 \n";
        }
        //std::cout << "BP dbg msg 11 \n";
        for(int j = 0; j < TempLayer->Neurons.size();++j){
          //  std::cout << "BP dbg msg 12 \n";
            auto &El = TempLayer->Neurons.at(j);
            //std::cout << "BP dbg msg 13 \n";
            El->Delta = Errors.at(j) * this->DeliveryBack(El->Output); //calculating gradient for updating weights of nn
            //std::cout << "BP dbg msg 14 \n";
        }
        //std::cout << "BP dbg msg 15 \n";
        i--;
    }
    //std::cout << "BP dbg msg 16 \n";
}

void NeuralNetwork::UpdateWeights(Data *Data) {
    //std::cout << "UW dbg msg 1 \n";
    std::vector<double> Inputs = Data->GetNormalizedFeatureVector();
    //std::cout << "Inputs size " << Inputs.size() <<"\n";
    //std::cout << "UW dbg msg 2 \n";
    for(int i = 0; i < Layers.size();i++){
      //  std::cout << "UW dbg msg 3 \n";
        if(i != 0){
        //    std::cout << "UW dbg msg 4 \n";
            for(auto &El : Layers.at(i-1)->Neurons) {
          //      std::cout << "UW dbg msg 5 \n";
                Inputs.push_back(El->Output);
            //    std::cout << "UW dbg msg 6 \n";
            }
            //std::cout << "UW dbg msg 7 \n";
        }
        //std::cout << "UW dbg msg 8 \n";
        for(auto &El : Layers.at(i)->Neurons){
         //   std::cout << "UW dbg msg 9 \n";
            for(int j = 0; j < El->Weights.size()-1/*Inputs.size()*/; j++) {
                //std::cout << "UW dbg msg 10 \n";
                //std::cout << "Input size " << Inputs.size() << " Weights size" << El->Weights.size() << "\n";
                El->Weights.at(j) += LearningRate * El->Delta * Inputs.at(j);
                //std::cout << "UW dbg msg 11 \n";
            }
           // std::cout << "UW dbg msg 12 \n";
            El->Weights.back() += LearningRate * El->Delta;
            //std::cout << "UW dbg msg 13 \n";
        }
        //std::cout << "UW dbg msg 14 \n";
    }
    //std::cout << "UW dbg msg 15 \n";
    Inputs.clear();
    //std::cout << "UW dbg msg 16 \n";
}



void NeuralNetwork::Train(int NumberOfEpochs) {
    for(int i = 0; i < NumberOfEpochs;i++){
        double SumOfErrors{};
        for(auto &El : DataForTraining){
            std::vector<double> Outputs = ForwardPropagation(El);
            std::vector<int> Expected = El->GetClassVector();
            double TempSumOfErrors{};
            for(int j = 0; j < Outputs.size(); j++)
                TempSumOfErrors += pow(static_cast<double>(Expected.at(j) - Outputs.at(j)),2);
            SumOfErrors += TempSumOfErrors;
            BackPropagation(El);
            UpdateWeights(El);
        }
        //std::cout << "Epoch " << i << "Error % = " << SumOfErrors << "\n";
        printf("Iteration: %d \t Error=%.4f\n", i, SumOfErrors/NumberOfEpochs);
    }
}

int NeuralNetwork::Predict(Data *Data) {
    std::vector<double> Outputs = ForwardPropagation(Data);
    //std::cout << "Outputs size "<< Outputs.size() << "\n";
    return std::distance(Outputs.begin(), std::max_element(Outputs.begin(), Outputs.end()));
}

double NeuralNetwork::TestProduce() {
    double AmountOfCorrectPredictions{};
    //std::cout << "Is it here?\n";
    int Counter{};
    for(auto &El : DataForTesting){
        Counter++;
        int Index = Predict(El);
        //std::cout << Index<< "\n";
        if(El->GetClassVector().at(Index) == 1)
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

void NeuralNetwork::TrainWhile() {
    double TSumOfErrors= 100000;
    long Epoch{};
    long double Temp{};
    while((TSumOfErrors > 0.4 && 0.5 < TSumOfErrors) || (TSumOfErrors > 1.0 && TSumOfErrors < 2.0)){
        TSumOfErrors = 0.0;
        long Counter{};
        double SumOfErrors{};
        for(auto &El : DataForTraining){
            std::vector<double> Outputs = ForwardPropagation(El);
            std::vector<int> Expected = El->GetClassVector();
            double TempSumOfErrors{};
            for(int j = 0; j < Outputs.size(); j++)
                TempSumOfErrors += static_cast<double>(Expected.at(j) - Outputs.at(j));
            Temp = SumOfErrors - TempSumOfErrors;
            SumOfErrors += TempSumOfErrors;
            BackPropagation(El);
            UpdateWeights(El);
            Counter++;
        }
        TSumOfErrors += SumOfErrors;
        Epoch++;
        //std::cout << "Epoch " << i << "Error % = " << SumOfErrors << "\n";
        printf("Iteration: %d \t Error=%.4f\n", Epoch, SumOfErrors);
        std::cout << "tr " << Temp << std::endl;
    }
}

void NeuralNetwork::NewTypeOfTrain() {
    while(true){
        unsigned Counter;
        double SumOfErrors{};
        for(auto &El : DataForTraining){
            std::vector<double> Outputs = ForwardPropagation(El);
            std::vector<int> Expected = El->GetClassVector();
            double TempSumOfErrors{};
            for(int j = 0; j < Outputs.size(); j++)
                TempSumOfErrors += pow(static_cast<double>(Expected.at(j) - Outputs.at(j)),2);
            SumOfErrors += TempSumOfErrors;
            BackPropagation(El);
            UpdateWeights(El);
            Counter++;
        }
        //std::cout << "Epoch " << i << "Error % = " << SumOfErrors << "\n";
        printf("Iteration: %d \t Error=%.10f\n", Counter, SumOfErrors);
    }
}


