# -*- coding: utf-8 -*-
# @Author: Junone
# @Date:   2019-12-19 14:23:29
# @Last Modified by:   Junone
# @Last Modified time: 2019-12-19 15:35:39
from sklearn.datasets import load_iris
import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
bitCoinData=pd.read_csv('bitCoin.csv')
#bitCoinData.loc[bitCoinData[''] !=' pre','col2']=Nonpre
bitCoinDataAfter=bitCoinData[~bitCoinData['direction'].isin([0])]
print(bitCoinDataAfter)
bitCoinDataAfter.loc[bitCoinData['direction'] ==-1,'direction']=0
print(bitCoinDataAfter)
data=bitCoinDataAfter.drop(['direction','Date/Time'],axis=1)
label=bitCoinDataAfter['direction']
data=np.array(data)
label=np.array(label)

data, X_test, label, y_test = train_test_split(data, label, test_size=0.6, random_state=42)


print(label)
from imblearn.under_sampling import RepeatedEditedNearestNeighbours,RandomUnderSampler,CondensedNearestNeighbour, NeighbourhoodCleaningRule,OneSidedSelection,ClusterCentroids,EditedNearestNeighbours,AllKNN, NearMiss,TomekLinks
from imblearn.over_sampling import RandomOverSampler,ADASYN,SMOTE,SVMSMOTE
from imblearn.combine import SMOTEENN,SMOTETomek
x_resample=data
y_resample=label
#x_resample,y_resample=NeighbourhoodCleaningRule().fit_resample(data,label)
#x_resample,y_resample=SMOTEENN().fit_resample(x_resample,y_resample)
print(len(x_resample),list(y_resample).count(0),list(y_resample).count(1))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
gbc = GradientBoostingClassifier()
gbc = gbc.fit(x_resample, y_resample)

score_gbc = gbc.score(X_test,y_test)

print(score_gbc)
print(metrics.roc_auc_score(y_test,gbc.predict_proba(X_test)[:,1]))
#random
#0.75879760098273
#0.8443246885496424
#smote
# 0.6299496040316774
# 0.6795494845534789


