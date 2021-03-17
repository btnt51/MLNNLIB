#include <KNN/headers/KNN.h>
#include <K-means/headers/Kmeans.h>
#include <header/Web.h>
//#include "NN/headers/NeuralNetwork.h"


int main() {

    auto *dataProcessor = new DataProcessor();
    //pc
    //dataProcessor->ReadInputData("C:\\code\\ML\\train-images-idx3-ubyte\\train-images.idx3-ubyte");
    //dataProcessor->ReadInputLabel("C:\\code\\ML\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");
    //dataProcessor->ReadInputData("C:\\code\\ML\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte");
    //dataProcessor->ReadInputLabel("C:\\code\\ML\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte");
    //laptop
    dataProcessor->ReadInputData("..\\train-images-idx3-ubyte\\train-images.idx3-ubyte");
    dataProcessor->ReadInputLabel("..\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");
    //dataProcessor->ReadInputData("D:\\c++\\ML\\train-images-idx3-ubyte\\train-images.idx3-ubyte");
    //dataProcessor->ReadInputLabel("D:\\c++\\ML\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");
    //dataProcessor->ReadInputData("D:\\c++\\ML\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte");
    //dataProcessor->ReadInputLabel("D:\\c++\\ML\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte");
    //laptop manjaro linux
    //dataProcessor->ReadInputData("/home/btnt51/git/c++/ML/train-images-idx3-ubyte/train-images.idx3-ubyte");
    //dataProcessor->ReadInputLabel("/home/btnt51/git/c++/ML/train-labels-idx1-ubyte/train-labels.idx1-ubyte");
    //dataProcessor->ReadInputData("/home/btnt51/git/c++/ML/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte");
    //dataProcessor->ReadInputLabel("/home/btnt51/git/c++/ML/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte");
    /*for(Data *El : dataProcessor->GetArrayOfData())
        El->PrintFeatureVector();*/
    /*for(Data *El : dataProcessor->GetArrayOfData())
        El->PrintNormalizedVector();*/
    dataProcessor->SplitData();
    dataProcessor->CountClasses();
    /*auto *Nearest = new KnnMethod(3,dataProcessor->GetDataForTraining(),dataProcessor->GetDataForTesting(),
                                  dataProcessor->GetDataForValidation());
    double performance = 0;
    double best_performance = 0;
    int best_k = 1;
    for(int k = 1; k <= 3; k++)
    {
        if(k == 1)
        {
            performance = Nearest->TestProduce();
            best_performance = performance;
        } else
        {
            Nearest->SetTheNumberOfNeighbors(k);
            performance = Nearest->TestProduce();
            if(performance > best_performance)
            {
                best_performance = performance;
                best_k = k;
            }
        }
    }
    Nearest->SetTheNumberOfNeighbors(best_k);
    Nearest->TestProduce();
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
//        printf("Current Performance @ K = %d: %.2f\n", k, performance);
        std::cout << k << "\n";
        if(performance > best_performance)
        {
            best_performance = performance;
            best_k = k;
        }
        km = nullptr;
    }
    auto *km = new KMeansMethod(best_k);
    km->SetDataForTraining(dataProcessor->GetDataForTraining());
    km->SetDataForTesting(dataProcessor->GetDataForTesting());
    km->SetDataForValidation(dataProcessor->GetDataForValidation());
    km->InitClusters();
    km->Train();
    printf("Overall Performance: %.2f\n",km->TestProduce());*/
    //std::vector<int> HiddenLayer = {2, 4, 6, 8};
    /*auto *Net = new NeuralNetwork(HiddenLayer,
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
    std::vector<int> HiddenLayer = {10};
    auto *web = new Web(HiddenLayer,0.55,
                        dataProcessor->GetDataForTraining().at(0)->GetNormalizedFeatureVector().size(),
                        dataProcessor->GetCountsOfClasses());
    web->SetDataForValidation(dataProcessor->GetDataForValidation());
    web->SetDataForTesting(dataProcessor->GetDataForTesting());
    web->SetDataForTraining(dataProcessor->GetDataForTraining());
    web->Train();
    std::cout << "Test Performance is " << web->TestPerformance() <<"\n";
    std::cout << "Validation Produce is " << web->ValidatePerformance() << "\n";
    return 0;
}


/*
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>

int main()
{
    std::array<std::vector<float>, 6> buffers;
    for (int i = 0; i < 6; ++i) {
        buffers[i] = std::vector<float>(480000, 0.5f);
    }

    auto tstart = std::chrono::high_resolution_clock::now();
    auto accum = 0;
    for (int i = 0; i < 6; ++i) {
        for (size_t j = 0; j < buffers[i].size(); ++j) {
            if (buffers[i][j] < 1.0f)
                ++accum;
        }
    }
    auto tend = std::chrono::high_resolution_clock::now();
    auto duration = tend - tstart;
    std::cout << "Raw loop: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << std::endl;

    tstart = std::chrono::high_resolution_clock::now();
    accum = 0;
    for (const auto& buffer : buffers) {
        for (const auto& value : buffer) {
            if (value < 1.0f)
                ++accum;
        }
    }
    tend = std::chrono::high_resolution_clock::now();
    duration = tend - tstart;
    std::cout <<"Range-based for loop: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << std::endl;

    tstart = std::chrono::high_resolution_clock::now();
    accum = 0;
    std::for_each(buffers.begin(), buffers.end(),
                  [&accum](const std::vector<float>& buffer) {
                      std::for_each(buffer.begin(), buffer.end(), [&accum](float value) { if (value < 1.0f) ++accum; });
                  }
    );
    tend = std::chrono::high_resolution_clock::now();
    duration = tend - tstart;
    std::cout << "std::for_each: " << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << std::endl;
}*/