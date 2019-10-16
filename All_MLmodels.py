from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import sklearn as sk
from sklearn import preprocessing
from sklearn import tree
import os

# input data
data_train=pd.read_csv('C:/Users/Administrator/Desktop/JEAC-review/EIS-Data-Nor-2.csv',sep=',')

labels=data_train['Type'][:,np.newaxis]
features=data_train.drop('Type', axis=1)

X_train,X_test,Y_train,Y_test=train_test_split(features, labels, test_size=0.2, random_state=0)#


names = ["Decision Tree", "Nearest Neighbors","Neural Net", "AdaBoost","Gaussian Process",
         "Random Forest", "Linear SVM", "RBF SVM", "Naive Bayes"]

classifiers = [
    DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                            max_features=None, random_state=None, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            class_weight=None, presort=False),
    KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto',
                            leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),
    MLPClassifier(hidden_layer_sizes=(100, ), activation='relu',
                    solver='adam', alpha=1, batch_size='auto', learning_rate='constant',
                    learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
                    random_state=None, tol=0.0001, verbose=False, warm_start=False,
                    momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                    validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10),
    AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0,
                        algorithm='SAMME.R', random_state=None),
    GaussianProcessClassifier(kernel=1.0 * RBF(1.0), optimizer='fmin_l_bfgs_b',
                                n_restarts_optimizer=0, max_iter_predict=100,
                                warm_start=False, copy_X_train=True,
                                random_state=None, multi_class='one_vs_rest', n_jobs=None),
    RandomForestClassifier(n_estimators='warn', criterion='gini',
                            max_depth=None, min_samples_split=2, min_samples_leaf=1,
                            min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True,
                            oob_score=False, n_jobs=None, random_state=None, verbose=0,
                            warm_start=False, class_weight=None),
    SVC(kernel="linear", C=5, degree=3, gamma='auto_deprecated', coef0=0.0,
        shrinking=True, probability=False, tol=0.001, cache_size=200,
        class_weight=None, verbose=False, max_iter=-1,
        decision_function_shape='ovr', random_state=None),
    SVC(C=5, kernel='rbf', gamma=0.35),
    GaussianNB(priors=None, var_smoothing=1e-09)]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    score_test = clf.score(X_test, Y_test)
    score_train = clf.score(X_train, Y_train)
    print (name,score_test,score_train )
