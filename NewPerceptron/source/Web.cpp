//
// Created by 51btn on 12.03.2021.
//

#include "../header/Web.h"

Web::Web(const std::vector<int>& Specification, double LearningRate, int InputSize, int NumberOfClasses) {
    this->LearningRate = LearningRate; //setting Learning rate
    Layers.push_back(new Layer(InputSize, Specification.at(0))); //filling first layer
    for (int i = 1; i < Specification.size(); ++i)  //fill hidden layers
        Layers.push_back(new Layer(Layers.at(i-1)->GetNeurons().size(),Specification.at(i)));
    Layers.push_back(new Layer(Layers.at(Layers.size()-1)->GetNeurons().size(),NumberOfClasses)); /*
 * filling last layer*/
}

Web::~Web() {
    for(auto &El : Layers)
        delete El; //deleting pointers on Layers
    Layers.clear(); //clearing vector of Layers
}

std::vector<double> Web::ForwardFeed(Data *Data) {
    std::vector<double> Inputs = Data->GetNormalizedFeatureVector(); //setting up input data
    for(auto &Layer : Layers){
        std::vector<double> Outputs{};
        for(auto &El : Layer->GetNeurons()){
            double Input = El->GetWeights().back(); //making a bias
            for(int i = 0; i < El->GetWeights().size()-1;++i){
                Input += El->GetWeights().at(i) * Inputs.at(i); //calculating input data for neuron
            }
            El->SetInput(Input); //setting up input data
            Input = 0; //clearing temp variable
            El->Activation(); //activating neuron
            Outputs.push_back(El->GetOutput()); //saving output of neuron
        }
        Inputs = Outputs; // outputs of previous layer becomes input for next layer
        Outputs.clear();
    }
    return Inputs;
}

double CalculatingBP(double Output){
    return Output * (1-Output);
}

void Web::BackPropagation(Data *Data) {
     // iterator that point on the last layer of nn
    for(unsigned i = Layers.size()-1;0 <= i && i > Layers.size();--i) {
        std::vector<double> Errors{}; //vector of error using for
        auto &TempLayer = Layers.at(i);

        if(i != Layers.size() -1) {
            for (int j = 0; j < TempLayer->GetNeurons().size(); ++j) {
                double Error{};

                for(auto &El : Layers.at(i+1)->GetNeurons())
                    Error += El->GetWeights().at(j) * El->GetDelta(); // Calculating error for non last layers
                Errors.push_back(Error);
            }
        } else {

            for (int j = 0; j < TempLayer->GetNeurons().size(); ++j) {
                auto &El = *TempLayer->GetNeurons().at(j);
                Errors.push_back(static_cast<double>(Data->GetClassVector().at(j) - El.GetOutput())); // expected - actual
            }
        }


        for (int j = 0; j < TempLayer->GetNeurons().size(); ++j) {
            auto &El = *TempLayer->GetNeurons().at(j);
            El.SetDelta(Errors.at(j) * CalculatingBP(El.GetOutput())); //calculating gradient for updating weights
        }
    }


    //UPDATING WEIGHTS
    std::vector<double> Input = Data->GetNormalizedFeatureVector();
    for (int i = 0; i < Layers.size(); ++i) {
        if(i != 0){
            std::vector<double> NewInputs;
            for(auto &El : Layers.at(i-1)->GetNeurons())
                NewInputs.push_back(El->GetOutput());
            Input = NewInputs;
            NewInputs.clear();
        }
        for(auto &El : Layers.at(i)->GetNeurons()) {
            for (int j = 0; j < Input.size(); ++j) {
                El->GetWeights().at(j) += LearningRate * El->GetDelta() * Input.at(j);
            }
            El->GetWeights().back() += LearningRate + El->GetDelta();
        }
    }
    Input.clear();
}

int Web::Predict(Data *Data) {
    std::vector<double> Output = ForwardFeed(Data);
    return std::distance(Output.begin(),std::max_element(Output.begin(), Output.end()));
}

void Web::Train() {
    std::vector<double> Output;
    int k{};
    while (true){
        double SumOfErrors{};
        for(auto &El : DataForTraining) {
            Output = ForwardFeed(El);
            std::vector<int> Actual = El->GetClassVector();
            double Temp{};
            for(int i = 0; i < Output.size();i++)
                Temp += pow(static_cast<double>(Actual.at(i) - Output.at(i)),2);
            SumOfErrors += Temp;
            BackPropagation(El);
            std::cout << "Error: " << SumOfErrors << std::endl;
            if (TestPerformance() > 0.8)
                break;
            std::cout << "Test produce " << TestPerformance() << std::endl;
        }
        if(ValidatePerformance() > 0.8)
            break;
        k++;
        printf("Iteration: %d \t Error=%.4f\n", k, SumOfErrors);
    }
}

double Web::TestPerformance() {
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
    Produce = AmountOfCorrectPredictions / Counter;
    return Produce;
}

double Web::ValidatePerformance() {
    double AmountOfCorrectPredictions{};
    int Counter{};
    for(auto &El : DataForValidation){
        Counter++;
        int Index = Predict(El);
        if(El->GetClassVector().at(Index)== 1)
            AmountOfCorrectPredictions++;
    }
    Produce = AmountOfCorrectPredictions / Counter;
    return Produce;
}
