'''
SVM for EIS
'''

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
data_train=pd.read_csv('EIS-Data.csv',sep=',')

labels=data_train['label'][:,np.newaxis]
features=data_train.drop('label', axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(features, labels, test_size=0.2, random_state=0) 

n_samples, n_features = X_train.shape

C_value=1
gamma_value=1

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf',C=C_value,gamma=gamma_value,probability=True)
clf_rbf.fit(X_train,Y_train)
score_rbf = clf_rbf.score(X_test,Y_test)
score_rbf2 = clf_rbf.score(X_train,Y_train)

# # kernel = 'linear'
clf_linear = svm.SVC(kernel='linear',C=C_value)
clf_linear.fit(X_train,Y_train)
score_linear = clf_linear.score(X_test,Y_test)
score_linear2 = clf_linear.score(X_train,Y_train)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly',C=C_value)
clf_poly.fit(X_train,Y_train)
score_poly = clf_poly.score(X_test,Y_test)
score_poly2 = clf_poly.score(X_train,Y_train)

print('C= %f' %C_value)
print (Y_test)
print(clf_rbf.predict_proba(X_test))
print("The score of rbf is : %f"%score_rbf)
print("The score of rbf2 is : %f"%score_rbf2)
print("The score of linear is : %f"%score_linear)
print("The score of linear2 is : %f"%score_linear2)
print("The score of poly is : %f"%score_poly)
print("The score of poly2 is : %f"%score_poly2)

