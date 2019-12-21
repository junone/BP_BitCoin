//
//  annBp.cpp
//  bpNlayers
//
//  Created by Junone on 12/18/19.
//  Copyright © 2019 Junone. All rights reserved.
//
#include "annBp.hpp"
#include <math.h>
#include <vector>
#include <fstream>
#include <sstream>
using namespace std;
Ann_bp::Ann_bp(int _SampleN, int nNIL, int nNOL,int nHidden,const int nNHL,float tolEerr, float _sR) :
SampleCount(_SampleN), numNodesInputLayer(nNIL), numNodesOutputLayer(nNOL),nLayer(nHidden),numNodesHiddenLayer(nNHL),toleranceEerr(tolEerr),studyRate(_sR)
{
 
    //create and initialize weight array
    //sett the random seed
    srand(1);
    weights = new double**[nHidden+1];
    //input layer connect with hiddernlayer
    weights[0] = new double *[numNodesInputLayer];
    for (int i = 0; i < numNodesInputLayer; ++i){
        weights[0][i] = new double[numNodesHiddenLayer];
        for (int j = 0; j <numNodesHiddenLayer; ++j){
            weights[0][i][j] = (rand() % (2000) / 1000.0 - 1); //from -1 to 1
        }
    }
    if(nHidden<2){
        weights[1] = new double *[numNodesHiddenLayer];
        for (int i = 0; i < numNodesHiddenLayer; ++i){
            weights[1][i] = new double[numNodesOutputLayer];
            for (int j = 0; j < numNodesOutputLayer; ++j){
                weights[1][i][j] = (rand() % (2000) / 1000.0 - 1); //from -1 to 1
            }
        }
    }
    else{
        for (int k=1;k<nHidden;k++){
            weights[k] = new double *[numNodesHiddenLayer];
            for (int i = 0; i < numNodesHiddenLayer; ++i){
                weights[k][i] = new double[numNodesHiddenLayer];
                for (int j = 0; j < numNodesHiddenLayer; ++j){
                    weights[k][i][j] = (rand() % (2000) / 1000.0 - 1); //from -1 to 1
                }
            }
        }
        //the last hidden layer connected with output layer
        weights[nHidden]=new double *[numNodesHiddenLayer];
        for (int i = 0; i < numNodesHiddenLayer; ++i){
            weights[nHidden][i] = new double[numNodesOutputLayer];
            for (int j = 0; j < numNodesOutputLayer; ++j){
                weights[nHidden][i][j] = (rand() % (2000) / 1000.0 - 1); //from -1 to 1
            }
        }
    }
    //create and initialize bias array
    bias = new double *[nHidden+1];
    for(int j=0;j<nHidden;j++){
        bias[j]= new double[numNodesHiddenLayer];
        for (int i = 0; i < numNodesHiddenLayer; ++i){
            bias[j][i] = (rand() % (2000) / 1000.0 - 1); //from -1 to 1
        }
    }
    //the last layer is output Layer
    bias[nHidden] = new double[numNodesOutputLayer];
    for (int i = 0; i < numNodesOutputLayer; ++i){
        bias[nHidden][i] = (rand() % (2000) / 1000.0 - 1); //from -1 to 1
    }
    
    //create and initialize hiddenLayer nodes
    hidenLayerOutput=new double *[nHidden];
    for(int i=0;i<nHidden;i++){
        hidenLayerOutput[i]=new double[numNodesHiddenLayer];
    }
    //create and initialize output layer nodes
    outputLayerOutput = new double[numNodesOutputLayer];
 
    // create each sample's  weight update array
    allDeltaWeights = new double ***[_SampleN];
    //only one hiddenlayer
    if(nHidden<2){
        for (int k = 0; k < _SampleN; ++k){
            allDeltaWeights[k] = new double**[2];
            allDeltaWeights[k][0] = new double *[numNodesInputLayer];
            for (int i = 0; i < numNodesInputLayer; ++i){
                allDeltaWeights[k][0][i] = new double[numNodesHiddenLayer];
            }
            allDeltaWeights[k][1] = new double *[numNodesHiddenLayer];
            for (int i = 0; i < numNodesHiddenLayer; ++i){
                allDeltaWeights[k][1][i] = new double[numNodesOutputLayer];
            }
        }
    }
    //n hiddenlayer
    else{
        for (int k = 0; k < _SampleN; ++k){
            allDeltaWeights[k] = new double**[nHidden+1];
            allDeltaWeights[k][0] = new double *[numNodesInputLayer];
            for (int i = 0; i < numNodesInputLayer; ++i){
                allDeltaWeights[k][0][i] = new double[numNodesHiddenLayer];
            }
            for(int j=1;j<nHidden;j++){
                allDeltaWeights[k][j] = new double *[numNodesHiddenLayer];
                for (int i = 0; i < numNodesHiddenLayer; ++i){
                    allDeltaWeights[k][j][i] = new double[numNodesHiddenLayer];
                }
            }
            allDeltaWeights[k][nHidden] = new double *[numNodesHiddenLayer];
            for (int i = 0; i < numNodesHiddenLayer; ++i){
                allDeltaWeights[k][nHidden][i] = new double[numNodesOutputLayer];
            }
        }
    }
 
    //create each sample's bias update array
    allDeltaBias = new double **[_SampleN];
    for (int k = 0; k < _SampleN; ++k){
        allDeltaBias[k] = new double *[nHidden+1];
        for(int i=0;i<nHidden;i++){
            allDeltaBias[k][i] = new double[numNodesHiddenLayer];
        }
        allDeltaBias[k][nHidden] = new double[numNodesOutputLayer];
    }
    //create each samples's output array
    outputMat = new double*[ _SampleN ];
    for (int k = 0; k < _SampleN; ++k){
        outputMat[k] = new double[ numNodesOutputLayer ];
    }
}
 
void Ann_bp::train(const int _sampleNum, float** _trainMat, int** _labelMat, const int trainingRounds)
{
    for (int i = 0; i < _sampleNum; ++i){
        train_vec(_trainMat[i], _labelMat[i], i);
    }
    int tt = 0;
    while (isNotConver(_sampleNum, _labelMat, toleranceEerr) && tt<trainingRounds){
        tt++;
        //change weight
        if(nLayer<2){
            //change weight if have only one hidden layer
            for (int index = 0; index < _sampleNum; ++index){
                for (int i = 0; i < numNodesInputLayer; ++i){
                    for (int j = 0; j < numNodesHiddenLayer; ++j){
                        weights[0][i][j] -= studyRate* allDeltaWeights[index][0][i][j];
                    }
                }
                for (int i = 0; i < numNodesHiddenLayer; ++i){
                    for (int j = 0; j < numNodesOutputLayer; ++j){
                        weights[1][i][j] -= studyRate* allDeltaWeights[index][1][i][j];
                    }
                }
            }
            //set bias
               for (int index = 0; index < _sampleNum; ++index){
                   for (int i = 0; i < numNodesHiddenLayer; ++i){
                       bias[0][i] -= studyRate* allDeltaBias[index][0][i];
                   }
                   for (int i = 0; i < numNodesOutputLayer; ++i){
                       bias[1][i] -= studyRate*allDeltaBias[index][1][i];
                   }
               }
        
               for (int i = 0; i < _sampleNum; ++i){
                   train_vec(_trainMat[i], _labelMat[i], i);
               }
        }
        else{
            for (int index = 0; index < _sampleNum; ++index){
                //first hidden layer connectted with input layer
                for (int i = 0; i < numNodesInputLayer; ++i){
                    for (int j = 0; j < numNodesHiddenLayer; ++j){
                        weights[0][i][j] -= studyRate* allDeltaWeights[index][0][i][j];
                    }
                }
                for(int k=1;k<nLayer;k++){
                    for (int i = 0; i < numNodesHiddenLayer; ++i){
                        for (int j = 0; j < numNodesHiddenLayer; ++j){
                            weights[k][i][j] -= studyRate* allDeltaWeights[index][k][i][j];
                        }
                    }
                }
                //last hidden layer connected with output layer
                for (int i = 0; i < numNodesHiddenLayer; ++i){
                    for (int j = 0; j < numNodesOutputLayer; ++j){
                        //cout<<nLayer<<weights[nLayer][i][j]<<endl;
                        weights[nLayer][i][j] -= studyRate* allDeltaWeights[index][nLayer][i][j];
                    }
                }
            }
            //set bias
           for (int index = 0; index < _sampleNum; ++index){
               for(int k=0;k<nLayer;++k){
                   for (int i = 0; i < numNodesHiddenLayer; ++i){
                       bias[k][i] -= studyRate* allDeltaBias[index][k][i];
                   }
               }
               //last layer output layer
               for (int i = 0; i < numNodesOutputLayer; ++i){
                   bias[nLayer][i] -= studyRate*allDeltaBias[index][nLayer][i];
               }
           }
           for (int i = 0; i < _sampleNum; ++i){
               train_vec(_trainMat[i], _labelMat[i], i);
           }
        }
    }
    cout<<"Succeed!"<<endl;
}
void Ann_bp::train_vec(const float* _trainVec, const int* _labelVec, int index)
{
    //calculate first layer's output
    for (int i = 0; i < numNodesHiddenLayer; ++i){
        double z = 0.0;
        for (int j = 0; j < numNodesInputLayer; ++j){
            z += _trainVec[j] * weights[0][j][i];
        }
        z += bias[0][i];
        hidenLayerOutput[0][i] = sigmoid(z);
        
    }
    //other hidden layer's output
    if(nLayer>=2){
        for(int k=1;k<nLayer;k++){
            for (int i = 0; i < numNodesHiddenLayer; ++i){
                double z = 0.0;
                for (int j = 0; j < numNodesHiddenLayer; ++j){
                    z += hidenLayerOutput[k-1][j] * weights[k][j][i];
                }
                z += bias[k][i];
                hidenLayerOutput[k][i] = sigmoid(z);
                
            }
        }
    }
    // calculate each node's output
    for (int i = 0; i < numNodesOutputLayer; ++i){
        double z = 0.0;
        for (int j = 0; j < numNodesHiddenLayer; ++j){
            z += hidenLayerOutput[nLayer-1][j] * weights[nLayer][j][i];
        }
        z += bias[nLayer][i];
        outputLayerOutput[i] = sigmoid(z);
        outputMat[index][i] = outputLayerOutput[i];//each sample's output node
    }
 
    //calculate delta bias  and delta weight,but do not update the array
    
    // calculate the delta bias and delta weight of the last layer
     for (int j = 0; j <numNodesOutputLayer; ++j){
         allDeltaBias[index][nLayer][j] = (-0.1)*(_labelVec[j] - outputLayerOutput[j])*outputLayerOutput[j]
             * (1 - outputLayerOutput[j]);
         for (int i = 0; i < numNodesHiddenLayer; ++i){
             allDeltaWeights[index][nLayer][i][j] = allDeltaBias[index][nLayer][j] *hidenLayerOutput[nLayer-1][i];
             
         }
     }
    if(nLayer==1){
        for (int j = 0; j < numNodesHiddenLayer; ++j){
            double z = 0.0;
            for (int k = 0; k < numNodesOutputLayer; ++k){
                z += weights[1][j][k] * allDeltaBias[index][1][k];//BP from the second layer
            }
            allDeltaBias[index][0][j] = z*hidenLayerOutput[0][j] * (1 - hidenLayerOutput[0][j]);// in the same layer
            for (int i = 0; i < numNodesInputLayer; ++i){
                allDeltaWeights[index][0][i][j] = allDeltaBias[index][0][j] * _trainVec[i];
            }
        }
    }
    else{
        for(int bpIndex=nLayer-1;bpIndex>=0;bpIndex--){
            if (bpIndex==nLayer-1){
                for (int j = 0; j < numNodesHiddenLayer; ++j){
                    double z = 0.0;
                    for (int k = 0; k < numNodesOutputLayer; ++k){
                        z += weights[bpIndex+1][j][k] * allDeltaBias[index][bpIndex+1][k];//BP from next layer
                    }
                    allDeltaBias[index][nLayer-1][j] = z*hidenLayerOutput[nLayer-1][j] * (1 - hidenLayerOutput[nLayer-1][j]);
                    for (int i = 0; i < numNodesHiddenLayer; ++i){
                        //change the weight of this layer.
                        allDeltaWeights[index][nLayer-1][i][j] = allDeltaBias[index][nLayer-1][j] *hidenLayerOutput[nLayer-2][i];
                    }
                }
            }
            //delata weight of the first layer
            else if (bpIndex==0){
                for (int j = 0; j < numNodesHiddenLayer; ++j){
                    double z = 0.0;
                    for (int k = 0; k < numNodesHiddenLayer; ++k){
                        z += weights[1][j][k] * allDeltaBias[index][1][k];//BP from the second layer
                    }
                    allDeltaBias[index][0][j] = z*hidenLayerOutput[0][j] * (1 - hidenLayerOutput[0][j]);// in the same layer
                    for (int i = 0; i < numNodesInputLayer; ++i){
                        allDeltaWeights[index][0][i][j] = allDeltaBias[index][0][j] * _trainVec[i];
                    }
                }
            }
            //delta weight of normal hidden layer
            else{
                for (int j = 0; j < numNodesHiddenLayer; ++j){
                    double z = 0.0;
                    for (int k = 0; k < numNodesHiddenLayer; ++k){
                        z += weights[bpIndex+1][j][k] * allDeltaBias[index][bpIndex+1][k];//BP from next layer
                    }
                    allDeltaBias[index][bpIndex][j] = z*hidenLayerOutput[bpIndex][j] * (1 - hidenLayerOutput[bpIndex][j]);
                    for (int i = 0; i < numNodesHiddenLayer; ++i){
                        //change the weight of this layer.
                        allDeltaWeights[index][bpIndex][i][j] = allDeltaBias[index][bpIndex][j] *hidenLayerOutput[bpIndex-1][i];
                    }
                }
            }
        }
    }
}

//calculate the total loss if the total loss is lower than tolerance error then return True
bool Ann_bp::isNotConver(const int _sampleNum,
    int** _labelMat, double _thresh)
{
    double lossFunc = 0.0;
    for (int k = 0; k < _sampleNum; ++k){
        double loss = 0.0;
        for (int t = 0; t < numNodesOutputLayer; ++t){
            loss += (outputMat[k][t] - _labelMat[k][t])*(outputMat[k][t] - _labelMat[k][t]);
        }
        lossFunc += (1.0 / 2)*loss;
    }
    lossFunc = lossFunc / _sampleNum;
    static int tt = 0;
    printf("The %d th training's loss：", ++tt);
    printf("%0.12f\n", lossFunc);
 
 
    if (lossFunc > _thresh)
        return true;
 
    return false;
}

float *Ann_bp::predict(float* in, float* proba)
{

    for (int i = 0; i < numNodesHiddenLayer; ++i){
        double z = 0.0;
        for (int j = 0; j < numNodesInputLayer; ++j){
            z += in[j] * weights[0][j][i];
        }
        z += bias[0][i];
        hidenLayerOutput[0][i] = sigmoid(z);
        
    }
    //other hidden layer's output
    if(nLayer>=2){
        for(int k=1;k<nLayer;k++){
            for (int i = 0; i < numNodesHiddenLayer; ++i){
                double z = 0.0;
                for (int j = 0; j < numNodesHiddenLayer; ++j){
                    z += hidenLayerOutput[k-1][j] * weights[k][j][i];
                }
                z += bias[k][i];
                hidenLayerOutput[k][i] = sigmoid(z);
                
            }
        }
    }
    float *predicted=new float[numNodesOutputLayer];
    //calculate outputlayerNodes' output
    for (int i = 0; i < numNodesOutputLayer; ++i){
        double z = 0.0;
        for (int j = 0; j < numNodesHiddenLayer; ++j){
            z += hidenLayerOutput[nLayer-1][j] * weights[nLayer][j][i];
        }
        z += bias[1][i];
        outputLayerOutput[i] = sigmoid(z);
        std::cout << outputLayerOutput[i] << " ";
        predicted[i]=outputLayerOutput[i];
    }
    return predicted;
}
//save the weight
void Ann_bp::weightTofile(const std::string fileName){
    ofstream oFile;
    ofstream reuseFile;
    oFile.open(fileName,ios::out | ios::trunc);
    reuseFile.open("/Users/Junone/Desktop/nut/MQF/Semester1/Programming/Final/bpNlayers/bpNlayers/weightReused.csv",ios::out | ios::trunc);
    for(int k=0;k<nLayer+1;k++){
        if(k==0){
            oFile<<"weight of input layer"<<endl;
            for(int i=0;i<numNodesInputLayer;i++){
                oFile<<i<<"th node's weight"<<",";
                for(int j=0;j<numNodesHiddenLayer;j++){
                    oFile<<weights[k][i][j]<<",";
                    reuseFile<<weights[k][i][j]<<",";
                }
                oFile<<endl;
                reuseFile<<endl;
            }
        }
        else if(k==nLayer){
                oFile<<"weight of last layer"<<endl;
                for(int i=0;i<numNodesHiddenLayer;i++){
                    oFile<<i<<"th node's weight"<<",";
                    for(int j=0;j<numNodesOutputLayer;j++){
                        oFile<<weights[k][i][j]<<",";
                        reuseFile<<weights[k][i][j]<<",";
                        }
                    oFile<<endl;
                    reuseFile<<endl;
                }
        }
        else{
            oFile<<k<<"th hidden layer's weight"<<endl;
            for(int i=0;i<numNodesHiddenLayer;i++){
                oFile<<i<<"th node's weight"<<",";
                for(int j=0;j<numNodesHiddenLayer;j++){
                    oFile<<weights[k][i][j]<<",";
                    reuseFile<<weights[k][i][j]<<",";
                }
                oFile<<endl;
                reuseFile<<endl;
            }
        }

    }
    oFile.close();
}
//load weight to the model, which should have the same parameter
void Ann_bp::readWeight(const std::string fileName){
    ifstream inFile;
    inFile.open(fileName);
    string line;
    string cell;
    for(int k=0;k<nLayer+1;k++){
        if(k==0){
            for(int i=0;i<numNodesInputLayer;i++){
                getline(inFile,line);
                stringstream lineStream(line);
                for(int j=0;j<numNodesHiddenLayer;j++){
                    getline(lineStream,cell,',');
                    weights[k][i][j]=stof(cell);
                }
            }
        }
        else if(k==nLayer){
            for(int i=0;i<numNodesHiddenLayer;i++){
                getline(inFile,line);
                stringstream lineStream(line);
                for(int j=0;j<numNodesOutputLayer;j++){
                    getline(lineStream,cell,',');
                    weights[k][i][j]=stof(cell);
                }
            }
        }
        else{
            for(int i=0;i<numNodesHiddenLayer;i++){
                getline(inFile,line);
                stringstream lineStream(line);
                for(int j=0;j<numNodesHiddenLayer;j++){
                    getline(lineStream,cell,',');
                    weights[k][i][j]=stof(cell);
                }
            }
        }
    }
}
