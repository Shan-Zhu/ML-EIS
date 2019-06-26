'''
data-all-13,test_size=0.2,random_state=0, linear=86%,70%, rbf(gamma=0.05)=77,74,rbf(gamma=0.1)=84,78, rbf(gamma=0.15)=86,76, rbf(gamma=0.2)=87,72,rbf(gamma=0.5)=90,62,rbf(gamma=1)=93,44
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


# input data
data_train=pd.read_csv('C:/Users/Administrator/Desktop/EIS-DATA-CSV-4/data-all-13.csv',sep=',')
# data_test=pd.read_csv('C:/Users/Administrator/Desktop/EIS-DATA-CSV-4/test-final.csv',sep=',')

labels=data_train['label'][:,np.newaxis]
features=data_train.drop('label', axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(features, labels, test_size=0.2, random_state=0) #

n_samples, n_features = X_train.shape

tsne = manifold.TSNE(n_components=2,perplexity=5,early_exaggeration=20,n_iter=10000)
X_tsne = tsne.fit_transform(X_train)
X_test_tsne = tsne.fit_transform(X_test)

C_value=1


# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf',C=C_value,gamma=0.1)
clf_rbf.fit(X_train,Y_train)
score_rbf = clf_rbf.score(X_test,Y_test)
score_rbf2 = clf_rbf.score(X_train,Y_train)

# kernel = 'rbf'
clf_rbf_tsne = svm.SVC(kernel='rbf',C=C_value,gamma=0.1)
clf_rbf_tsne.fit(X_tsne,Y_train)
score_rbf_tsne = clf_rbf_tsne.score(X_test_tsne,Y_test)
score_rbf2_tsne = clf_rbf_tsne.score(X_tsne,Y_train)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear',C=C_value)
clf_linear.fit(X_train,Y_train)
score_linear = clf_linear.score(X_test,Y_test)
score_linear2 = clf_linear.score(X_train,Y_train)


# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear',C=C_value)
clf_linear.fit(X_tsne,Y_train[:,0])
score_linear_tnse = clf_linear.score(X_test_tsne,Y_test)
score_linear2_tnse = clf_linear.score(X_tsne,Y_train)


# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly',C=C_value)
clf_poly.fit(X_train,Y_train)
score_poly = clf_poly.score(X_test,Y_test)

print('C= %f' %C_value)
print("The score of rbf is : %f"%score_rbf)
print("The score of rbf2 is : %f"%score_rbf2)
print("The score of rbf_tsne is : %f"%score_rbf_tsne)
print("The score of rbf2_tsne is : %f"%score_rbf2_tsne)
print("The score of linear is : %f"%score_linear)
print("The score of linear2 is : %f"%score_linear2)
print("The score of linear_tnse is : %f"%score_linear_tnse)
print("The score of linear2_tnse is : %f"%score_linear2_tnse)
print("The score of poly is : %f"%score_poly)
