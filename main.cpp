#include <KNN/headers/KNN.h>
#include <K-means/headers/Kmeans.h>
#include "NN/headers/NeuralNetwork.h"


int main() {

    auto *dataProcessor = new DataProcessor();
    //Train dataset on 60 K images
    dataProcessor->ReadInputData("..\\dataset\\train-images.idx3-ubyte");
    dataProcessor->ReadInputLabel("..\\dataset\\train-labels.idx1-ubyte");
    //Test dataset on 10 K images
//     dataProcessor->ReadInputData("..\\dataset\\t10l-images.idx3-ubyte");
//     dataProcessor->ReadInputLabel("..\\dataset\\t10k-labels.idx1-ubyte");
    dataProcessor->SplitData();
    dataProcessor->CountClasses();
    /*  KNN */
    /*auto *Nearest = new KnnMethod(3,dataProcessor->GetDataForTraining(),dataProcessor->GetDataForTesting(),
                                  dataProcessor->GetDataForValidation());

    double performance;
    double best_performance = 0;
    int best_k = 1;
    for(int k = 1; k <= 5; k++)
    {
        if(k == 1)
        {
            performance = Nearest->ValidateProduce();
            best_performance = performance;
        } else
        {
            Nearest->SetTheNumberOfNeighbors(k);
            performance = Nearest->ValidateProduce();
            if(performance > best_performance)
            {
                best_performance = performance;
                best_k = k;
            }
        }
    }
    Nearest->SetTheNumberOfNeighbors(best_k);
    Nearest->TestProduce();*/
    double performance = 0;
    double best_performance = 0;
    int best_k = 1;
    for(int k = dataProcessor->GetCountsOfClasses(); k < dataProcessor->GetSizeOfDataForTraining()*0.1; k++)
    {
        auto *km = new KMeansMethod(k);
        km->SetDataForTraining(dataProcessor->GetDataForTraining());
        km->SetDataForTesting(dataProcessor->GetDataForTesting());
        km->SetDataForValidation(dataProcessor->GetDataForValidation());
        km->InitClusters();
        km->Train();
        performance = km->ValidateProduce();
        printf("Current Performance @ K = %d: %.2f\n", k, performance);
       // std::cout << k << "\n";
        if(performance > best_performance)
        {
            best_performance = performance;
            best_k = k;
        }
    }
    auto *km = new KMeansMethod(best_k);
    km->SetDataForTraining(dataProcessor->GetDataForTraining());
    km->SetDataForTesting(dataProcessor->GetDataForTesting());
    km->SetDataForValidation(dataProcessor->GetDataForValidation());
    km->InitClusters();
    km->Train();
    printf("Overall Performance: %.2f\n",km->TestProduce());
    /*std::vector<int> HiddenLayer = {2, 4, 6, 8};
    auto *Net = new NeuralNetwork(HiddenLayer,
                                  dataProcessor->GetDataForTraining().at(0)->GetNormalizedFeatureVector().size(),
                                           dataProcessor->GetCountsOfClasses(), 0.0000005);
    Net->SetDataForTraining(dataProcessor->GetDataForTraining());
    Net->SetDataForTesting(dataProcessor->GetDataForTesting());
    Net->SetDataForValidation(dataProcessor->GetDataForValidation());
    //Net->Train(15);
    //Net->TrainWhile();
    //Net->Train(1000);
    Net->NewTypeOfTrain();
    std::cout << "Test Performance is " << Net->TestProduce() <<"\n";
    std::cout << "Validation Produce is " << Net->ValidateProduce() << "\n";*/
    return 0;
}

