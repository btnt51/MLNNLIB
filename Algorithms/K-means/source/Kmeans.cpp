//
// Created by 51btn on 24.02.2021.
//
#include "../headers/Kmeans.h"
#define max std::numeric_limits<double>::max()


ClusterOfData::ClusterOfData(Data *InitialPoint) {
    for(auto &El : (InitialPoint->GetNormalizedFeatureVector())) {
        if (std::isnan(El)) {
            Centroid.push_back(0);
        }
        else {
            Centroid.push_back(El);
        }
    }
    //std::cout << "Centroid " << Centroid.size() << std::endl;
    ClusterOfPoints.push_back(InitialPoint);
    CountsOfClasses[InitialPoint->GetLabel()] = 1;
    TheMostFrequentClass = InitialPoint->GetLabel();
}


ClusterOfData::~ClusterOfData() {
    for(auto El : ClusterOfPoints)
        delete El;
    ClusterOfPoints.clear();
}


void ClusterOfData::AddToCluster(Data *Point) {
    int PreviousSize = ClusterOfPoints.size();
    ClusterOfPoints.push_back(Point);
    for(int i = 0; i < Centroid.size()-1; ++i){
        double El = Centroid.at(i);
        El *= PreviousSize;
        El *= Point->GetNormalizedFeatureVector().at(i);
        El /= static_cast<double>(ClusterOfPoints.size());
        Centroid.at(i) = El;
    }

   /* (CountsOfClasses.find(Point->GetLabel()) == CountsOfClasses.end()) ?
                            CountsOfClasses[Point->GetLabel()] = 1 : CountsOfClasses[Point->GetLabel()]++;*/
    if(CountsOfClasses.find(Point->GetLabel()) == CountsOfClasses.end())
        CountsOfClasses[Point->GetLabel()] = 1;
    else
        CountsOfClasses[Point->GetLabel()]++;
    SetTheMostFrequentClass();
}


void ClusterOfData::SetTheMostFrequentClass() {
    int PopularClass{};
    int Freq{};
    for(auto El : CountsOfClasses){
        if(El.second > Freq){
            Freq = El.second;
            PopularClass = El.first;
        }
    }
    TheMostFrequentClass = PopularClass;
}


KMeansMethod::KMeansMethod(int NumberOfClusters) {
    this->NumberOfClusters = NumberOfClusters;
    //std::cout << "Clusters could hold " <<  Clusters.max_size() << " elements" << std::endl;
}


KMeansMethod::~KMeansMethod() {
    for(auto El : Clusters)
        delete El;
    Clusters.clear();
    UsedIndexes.clear();
}


void KMeansMethod::InitClusters() {
    for(int i=0; i < NumberOfClusters;i++){
        std::mt19937 Gen(time(nullptr)*DataForTraining.size()*DataForValidation.size());
        std::uniform_int_distribution<> UID(0, DataForTraining.size()-1);
        int Index = UID(Gen);
        while(UsedIndexes.find(Index) != UsedIndexes.end())
            Index = UID(Gen);
        Clusters.push_back(new ClusterOfData(DataForTraining.at(Index)));
        UsedIndexes.insert(Index);
    }
}


void KMeansMethod::Train() {
    //Clusters.resize(DataForTraining.size()+1);
    while (UsedIndexes.size() < DataForTraining.size()){
        std::mt19937 Gen(time(nullptr)*DataForTraining.size()*DataForValidation.size());
        std::uniform_int_distribution<> UID(0, DataForTraining.size()-1);
        int Index = UID(Gen);
        while(UsedIndexes.find(Index) != UsedIndexes.end())
            Index = UID(Gen);

        double MinimalDistance = max;
        int TheBestCluster{};
        for(int i = 0; i < Clusters.size()-1; ++i){
          double Distance = GetDistance(&Clusters.at(i)->Centroid, DataForTraining.at(Index));
            if(Distance < MinimalDistance){
                MinimalDistance = Distance;
                TheBestCluster = i;
            }
        }
        //ClusterOfData SomeCluster(DataForTraining.at(Index));
        //SomeCluster.AddToCluster(DataForTraining.at(Index));
        //Clusters.at(TheBestCluster) = &SomeCluster;
        Clusters.at(TheBestCluster)->AddToCluster(DataForTraining.at(Index));
        UsedIndexes.insert(Index);
    }
}


void KMeansMethod::InitClustersForEachClass() {
    std::unordered_set<int> ProcessedClasses;
    for(int i = 0; i < DataForTraining.size();i++){
        if(ProcessedClasses.find(DataForTraining.at(i)->GetLabel()) == ProcessedClasses.end()){
            Clusters.push_back(new ClusterOfData (DataForTraining.at(i)));
            UsedIndexes.insert(i);
            ProcessedClasses.insert(DataForTraining.at(i)->GetLabel());
        }
    }
}


double KMeansMethod::GetDistance(std::vector<double> *Centroid, Data *QueryPoint, int Fashion) {
    double Distance{};
    //int Dimensionality = Centroid->size();
    switch(Fashion)
    {
        default:
        {//Default method for finding distance in Euclid distance d(x,y)=sqrt((sigma((xi-yi)^2))/m)
            for(unsigned i = 0; i < Centroid->size()-1/*Dimensionality*/;++i)
                Distance += pow( Centroid->at(i) - QueryPoint->GetNormalizedFeatureVector().at(i),2);
            Distance /= Centroid->size()/*Dimensionality*/;
            return sqrt(Distance);
        }

        case 1: {
            //Manhattan distance by Minkowski metric d(x,y) = sigma(|xi-yi|)
            for (unsigned i = 0; i < Centroid->size()-1/*Dimensionality*/; ++i)
                Distance += std::abs(Centroid->at(i) - QueryPoint->GetNormalizedFeatureVector().at(i));
            return Distance;
        }

        case 2:{
            //Euclid distance by Minkowski metric d(x,y) = sqrt(sigma((xi - yi)^2))
            for(unsigned i = 0; i < Centroid->size()-1/*Dimensionality*/;++i)
                Distance += pow(Centroid->at(i) -  QueryPoint->GetNormalizedFeatureVector().at(i),2);
            return sqrt(Distance);
        }
    }
}


double KMeansMethod::ValidateProduce() {
    double Correction{};
    for(auto &El : DataForValidation){
        double MinimalDistance = max;
        int Index{};
        for(int i = 0; i < Clusters.size();i++){
            double Distance = GetDistance(&Clusters.at(i)->Centroid, El);
            if(Distance < MinimalDistance){
                MinimalDistance = Distance;
                Index = i;
            }
        }
        if(Clusters.at(Index)->TheMostFrequentClass == El->GetLabel()) Correction++;
    }
    return 100.0 *(Correction/static_cast<double>(DataForValidation.size()));
}

double KMeansMethod::TestProduce() {
    double Correction{};
    for(auto &El : DataForTesting){
        double MinimalDistance = max;
        int Index{};
        for(int i = 0; i < Clusters.size();i++){
            double Distance = GetDistance(&Clusters.at(i)->Centroid, El);
            if(Distance < MinimalDistance){
                MinimalDistance = Distance;
                Index = i;
            }
        }
        if(Clusters.at(Index)->TheMostFrequentClass == El->GetLabel()) Correction++;
    }
    return 100.0 *(Correction/static_cast<double>(DataForTesting.size()));
}







