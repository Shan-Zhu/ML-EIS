from numpy import *
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
from sklearn import preprocessing
from sklearn import svm
import os
from sklearn import manifold
import matplotlib.pyplot as plt

# input data
data_train=pd.read_csv('',sep=',')

labels=data_train['Type'][:,np.newaxis]
features=data_train.drop('Type', axis=1)#'DOI',


X_train,X_test,Y_train,Y_test=train_test_split(features, labels, test_size=0.2, random_state=0) #

####   grid search start
best_score = 0
for gamma in [0.25,5,1,0.4,0.1,0.15,0.2,0.3,0.35,,0.45,0.5,2]:#
    for C in [0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,10]:
        clf_rbf = svm.SVC(kernel='rbf',C=C,gamma=gamma,probability=True)
        clf_rbf.fit(X_train,Y_train)
        rbf_test = clf_rbf.score(X_test,Y_test)
        parameters = {'gamma':gamma,'C':C}
        print(rbf_test)
        print("Best parameters:{}".format(parameters))
