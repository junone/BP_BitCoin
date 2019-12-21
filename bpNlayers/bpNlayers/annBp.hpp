//
//  annBp.hpp
//  bpNlayers
//
//  Created by Junone on 12/18/19.
//  Copyright © 2019 Junone. All rights reserved.
//

//ann_bp.h//
#ifndef _ANN_BP_H_
#define _ANN_BP_H_
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
//#include <windows.h>
#include <ctime>
#include <vector>
 
class Ann_bp
{
public:
    explicit Ann_bp(int _SampleN, int nNIL, int nNOL,int nHidden, const int nNHL,float tolEerr, float _sR = 0.2);
    //~Ann_bp();
 
    void train(int _sampleNum, float** _trainMat, int** _labelMat,const int trainingRounds);
    float *predict(float* in, float* proba);
    float Accuracy(const float& dataset, const float &label);
    void weightTofile(const std::string fileName);
    void readWeight(const std::string fileName);
 
private:
    int numNodesInputLayer;     //number of nodes in inputlayer
    int numNodesOutputLayer;    //number of nodes in outputlayer
    int numNodesHiddenLayer;    //number of node in hiddenlayer
    int nLayer;                 // number of hidden Layers
    int trainingRounds;            //round of training
    int SampleCount;               //number of samples
    double ***weights;            //node's each weight
    double **bias;                 //network's bias
    float studyRate;               //learning rate
    float toleranceEerr;            //tolerance error
 
    double **hidenLayerOutput;     //output value of each hiddenlayer
    double *outputLayerOutput;     //output value of outputlayer
    double **predictedValue;        //predicted value
    double ***allDeltaBias;        //every sample's delta bias
    double ****allDeltaWeights;    //every sample's delta weight
    double **outputMat;            //every samples's output layer
    std::vector<float> lossArray;
    void train_vec(const float* _trainVec, const int* _labelVec, int index);
    double sigmoid(double x){ return 1 / (1 + exp(-1 * x)); }
    bool isNotConver(const int _sampleNum, int** _labelMat, double _thresh);
 
};
#endif
