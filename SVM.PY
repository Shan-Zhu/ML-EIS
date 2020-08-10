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
data_train=pd.read_csv('...',sep=',')

labels=data_train['Type'][:,np.newaxis]
features=data_train.drop('Type', axis=1)#'DOI',

C_value=5#2.5#1#5,2.5
gamma_value=0.35#0.45#0.2#0.35,0.45

def ml_eis(features, labels, random_state,C_value,gamma_value):

    X_train,X_test,Y_train,Y_test=train_test_split(features, labels, test_size=0.2) #, random_state=0

    # n_samples, n_features = X_train.shape

    # kernel = 'rbf'
    clf_rbf = svm.SVC(kernel='rbf',C=C_value,gamma=gamma_value,probability=True)
    clf_rbf.fit(X_train,Y_train)
    rbf_test = clf_rbf.score(X_test,Y_test)
    rbf_train = clf_rbf.score(X_train,Y_train)

    # kernel = 'linear'
    clf_linear = svm.SVC(kernel='linear',C=C_value,gamma=gamma_value)
    clf_linear.fit(X_train,Y_train)
    linear_test = clf_linear.score(X_test,Y_test)
    linear_train = clf_linear.score(X_train,Y_train)

    #kernel = 'poly'
    clf_poly = svm.SVC(kernel='poly',C=C_value)
    clf_poly.fit(X_train,Y_train)
    poly_test = clf_poly.score(X_test,Y_test)
    poly_train = clf_poly.score(X_train,Y_train)

    result_score=pd.DataFrame([rbf_test,rbf_train,linear_test,linear_train,poly_test, poly_train], columns=[random_state], index=['rbf_test','rbf_train','linear_test','linear_train','poly_test', 'poly_train'])

    return result_score

result_score= ml_eis(features, labels, 0, C_value,gamma_value)

for num in range(1,101):
    result_score_1= ml_eis(features, labels, num, C_value,gamma_value)

    result_score=result_score.join(result_score_1)


result_score.to_csv('...',sep=',')

print ('OK')
