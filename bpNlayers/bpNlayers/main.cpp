//
//  main.cpp
//  bpNlayers
//
//  Created by Junone on 12/18/19.
//  Copyright © 2019 Junone. All rights reserved.
//
#include <iostream>
#include "annBp.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <valarray>
#include <iterator>
using namespace std;

float **readCSVX(string fileName,int numNode,int sampleN)
{
    ifstream  data;
    data.open(fileName);//Set the file name with its path
    string line;
    float** dataSet = new float* [sampleN];
    getline(data,line);
    int i{0};
    while(getline(data,line) && i<sampleN )
    {
        stringstream lineStream(line);
        string cell;
        dataSet[i] = new float[numNode];//numNode denotes numbers of features
        int j{0};
        while(getline(lineStream,cell,','))
        {
            dataSet[i][j++]=stof(cell);
        }
        i++;
    }
    data.close();
    return dataSet;
}

int **readCSVY(string fileName,int sampleN,int dataClass)
{
    ifstream  data;
    data.open(fileName);//Set the file name with its path
    string line;
    int** dataSet = new int* [sampleN];
    int i{0};
    while(getline(data,line) && i<sampleN )
    {
        stringstream lineStream(line);
        string cell;
        //initialize label dataSet
        dataSet[i] = new int[dataClass]{0};
        while(getline(lineStream,cell))
        {
            dataSet[i][stoi(cell)]=1;
        }
        //cout<<endl;
        i++;
    }
    data.close();
    return dataSet;
}



int main()
{
    cout<<"Welcome to use ANN classification system !"<<endl;
    cout<<"1. train the model"<<endl;
    cout<<"2. Load weight matrix "<<endl;
    cout<<"3. Predict"<<endl;
    cout<<"4.Exit"<<endl;
//test network
    cout<<"Please input parameters: "<<endl;
    cout<<"They are hidnodes, innodes,outnodes,"<<endl;
    cout<<"trainrounds,nlayer,ntrain and toleranceError:"<<endl;
    int hidnodes = 100; //numbers of hiddenNodes
    int inNodes = 13;   //numbers of inputNodes
    int outNodes = 3;  //numbers of outputNodes
    int trainRounds=300; //numbers of trainning rounds
    int nlayer=2;
    int nTest=10000;
    int nTrain=11409;
    float toleranceEerr=-0.1;
    float learningRate=0.0001;
    cin>>hidnodes>>inNodes>>outNodes>>trainRounds>>nlayer>>nTrain>>toleranceEerr>>learningRate;
    Ann_bp ann_classify(nTrain, inNodes, outNodes,nlayer, hidnodes,toleranceEerr, learningRate);
    cout<<endl;
    cout<<"Please choose want you want to do next:";
    int choose=0;
    cin>>choose;
    while(choose<1||choose>4){
        cout<<"Please input right number!"<<endl;
        cout<<"Please choose want you want to do next:";
        cin>>choose;
    }
    while(choose!=4){
        switch (choose) {
            // loading train data set and train the model
            case 1:{
                //the parameters tested:
                //100 13 3 300 2 11409 0.1 0.001
                //100 13 3 200 3 11409 0.1 0.001
                //100 13 3 200 1 11409 0.1 0.001
                string xName;
                cout<<"Please input the file name of train data:";
                cin>>xName;
                string yName;
                cout<<"Please input the file name of train label:";
                cin>>yName;
                //file's name
                //x_trainBalanceFinal.csv
                //y_trainBalanceFinal.csv
                int **labelMat=readCSVY("/Users/Junone/Desktop/nut/MQF/Semester1/Programming/Final/bpNlayers/bpNlayers/"+yName,nTrain,outNodes);//2 labels
                float **trainMat=readCSVX("/Users/Junone/Desktop/nut/MQF/Semester1/Programming/Final/bpNlayers/bpNlayers/"+xName,inNodes, nTrain);
                ann_classify.train(nTrain, trainMat, labelMat,trainRounds);
                ann_classify.weightTofile("/Users/Junone/Desktop/nut/MQF/Semester1/Programming/Final/bpNlayers/bpNlayers/weightForModel3.csv");
                break;
            }
            //  load the weight matrix into the model
            case 2:{
                //file's name
                //weightReusedLast.csv
                //weightReused_2_100_300.csv
                cout<<"Please input the file name of weight:";
                string weightName;
                cin>>weightName;
                    ann_classify.readWeight("/Users/Junone/Desktop/nut/MQF/Semester1/Programming/Final/bpNlayers/bpNlayers/"+weightName);

                break;
            }
            // test the model and calculate accuracy
            case 3:{
                // file's name and number of samples used
                //x_test.csv
                //y_test.csv
                //10000
                cout<<"Please input the file name of test data:";
                string testName;
                string testL;
                cin>>testName;
                cout<<"Please input the file name of test label:";
                cin>>testL;
                cout<<"Please input numbers of test data:";
                cin>>nTest;
                float **testMat=readCSVX("/Users/Junone/Desktop/nut/MQF/Semester1/Programming/Final/bpNlayers/bpNlayers/"+testName,inNodes, nTest);
                int **testLabel=readCSVY("/Users/Junone/Desktop/nut/MQF/Semester1/Programming/Final/bpNlayers/bpNlayers/"+testL,nTest,outNodes);//2 labels
                float sumLabel0=0;// sum the negative label's value of each sample
                vector<float*> prdLabel;// value predicted

                for (int i = 0; i < nTest; ++i){
                    cout<<"Predict: ";
                    prdLabel.push_back(ann_classify.predict(testMat[ i], NULL));
                    sumLabel0+=prdLabel.at(i)[0];
                    cout <<"Real:"<<testLabel[i][0]<<testLabel[i][1]<<testLabel[i][2];
                    cout<<endl;
                }
                //calculate the total accuracy
                float totalAccuracy=0;
                int nAccurate=0;
                float upAccuracy=0,downAccuracy=0,sideAccuracy=0;
                int nUp=0,nDown=0,nSide=0;
                //save the result to csv
                fstream predictedOutput;
                predictedOutput.open("/Users/Junone/Desktop/nut/MQF/Semester1/Programming/Final/bpNlayers/bpNlayers/predictedLabel.csv",ios::out | ios::trunc);
                for(int i=0;i<nTest;i++){
                    predictedOutput<<distance(prdLabel.at(i),max_element(prdLabel.at(i), prdLabel.at(i)+3))<<endl;
                    if (distance(testLabel[i],max_element(testLabel[i], testLabel[i]+3)) ==distance(prdLabel.at(i),max_element(prdLabel.at(i), prdLabel.at(i)+3))){
                        nAccurate+=1;
                    }

                    if(distance(testLabel[i],max_element(testLabel[i], testLabel[i]+3))==0){
                        nDown+=1;
                        if(distance(prdLabel.at(i),max_element(prdLabel.at(i), prdLabel.at(i)+3))==0){
                            downAccuracy+=1;
                        }
                    }
                    if(distance(testLabel[i],max_element(testLabel[i], testLabel[i]+3))==1){
                        nUp+=1;
                        if(distance(prdLabel.at(i),max_element(prdLabel.at(i), prdLabel.at(i)+3))==1){
                            upAccuracy+=1;
                        }
                    }
                    if(distance(testLabel[i],max_element(testLabel[i], testLabel[i]+3))==2){
                        nSide+=1;
                        if(distance(prdLabel.at(i),max_element(prdLabel.at(i), prdLabel.at(i)+3))==2){
                            sideAccuracy+=1;
                        }
                    }
                }
                //print the result on to the screen
                totalAccuracy=(float)nAccurate/nTest;
                upAccuracy=(float)upAccuracy/nUp;
                downAccuracy=(float)downAccuracy/nDown;
                sideAccuracy=(float)sideAccuracy/nSide;
                cout<<"Total Accuracy:"<<(float)totalAccuracy<<endl;
                cout<<"up Accuracy:"<<(float)upAccuracy<<endl;
                cout<<"down Accuracy:"<<(float)downAccuracy<<endl;
                cout<<"side Accuracy:"<<(float)sideAccuracy<<endl;
                break;
            }
        }
        cout<<"Please choose want you want to do next:";
        cin>>choose;
    }
     cout<<"Quit the system! Thank you!"<<endl;


}
