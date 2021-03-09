//
// Created by 51btn on 24.02.2021.
//

#ifndef ML_KMEANS_H
#define ML_KMEANS_H

#include <limits>
#include <cmath>
#include <map>
#include <random>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <unordered_set>
//#include <headers/GenerealizedDataContainer.h>
#include <headers/GeneralizedDataContainer.h>
#include <headers/DataProcessor.h>

/*typedef struct cluster
{
    std::vector<double> *centroid;
    std::vector<Data *> *clusterPoints;
    std::map<int, int> classCounts;
    int mostFrequentClass;
    cluster(Data *initialPoint)
    {
        centroid = new std::vector<double>;
        clusterPoints = new std::vector<Data *>;
        for(auto val : *(initialPoint->GetNormalizedFeatureVector()))
        {
            if(_isnan(val))
                centroid->push_back(0);
            else
                centroid->push_back(val);
        }
        clusterPoints->push_back(initialPoint);
        classCounts[initialPoint->GetLabel()] = 1;
        mostFrequentClass = initialPoint->GetLabel();
    }

    void add_to_cluster(Data* point)
    {
        int previous_size = clusterPoints->size();
        clusterPoints->push_back(point);
        for(int i = 0; i < centroid->size(); i++)
        {
            double val = centroid->at(i);
            val *= previous_size;
            val += point->GetNormalizedFeatureVector()->at(i);
            val /= (double)clusterPoints->size();
            centroid->at(i) = val;
        }
        if(classCounts.find(point->GetLabel()) == classCounts.end())
        {
            classCounts[point->GetLabel()] = 1;
        } else
        {
            classCounts[point->GetLabel()]++;
        }
        set_mostFrequentClass();
    }
    void set_mostFrequentClass()
    {
        int best_class;
        int freq = 0;
        for(auto kv : classCounts)
        {
            if(kv.second > freq)
            {
                freq = kv.second;
                best_class = kv.first;
            }
        }
        mostFrequentClass = best_class;
    }
} cluster_t;*/

class ClusterOfData{
public:
    ClusterOfData(Data *InitialPoint);
    ~ClusterOfData();

    void AddToCluster(Data * Point);
    void SetTheMostFrequentClass();

    std::vector<Data *> ClusterOfPoints;
    std::vector<double> Centroid;
    std::map<int, int> CountsOfClasses;
    int TheMostFrequentClass{};

};

class KMeansMethod : public GeneralizedDataContainer {
public:
    KMeansMethod(int);
    ~KMeansMethod();

    void InitClusters();
    void InitClustersForEachClass();
    void Train();

    double GetDistance(std::vector<double> *, Data *, int Fashion = 2);
    double ValidateProduce();
    double TestProduce();

    //std::vector<ClusterOfData *> *GetClusters() { return this->Clusters;}
    std::vector<ClusterOfData*> GetClusters() { return Clusters;}

private:
    int NumberOfClusters{};
    std::vector<ClusterOfData *> Clusters;
    std::unordered_set<int> UsedIndexes;
};

#endif //ML_KMEANS_H
