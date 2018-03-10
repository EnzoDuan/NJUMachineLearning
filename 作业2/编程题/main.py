# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import linear_model
import pandas as pd 
#    np.genfromtxt()                    load csv
#    np.exp()                           e^x
#    np.dot()                           matrix inner mul
#    np.linalg.inv()
#    sklearn.model_selection.KFold()    
data = np.genfromtxt('data.csv',delimiter = ',',skip_header=0,dtype=None)
targets = np.genfromtxt('targets.csv',delimiter = ',',skip_header=0,dtype=None)

def Sigmoid(X):
       return 1.0/(1+np.exp(-X))
   
def Newton(train_X, labels, max_iter):
    X = np.mat(train_X)
    Y = np.mat(labels).transpose()
    rows, columns = np.shape(X)
    weights = np.zeros((columns, 1))
    t = 0
    while t < max_iter:
        P1 = Sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (Y - P1))
        P_mat = np.array(P1) * np.array(P1 - 1) * np.eye(rows)
        hessian = X.T * P_mat * X
        weights -= np.linalg.inv(hessian) * gradient
        t += 1
    return weights
def z_score(x, axis):
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    # print(x)
    return x

if __name__ == "__main__":
    X = data
    y = targets
    kf1 = KFold(n_splits=10)
#    kf2 = KFold(n_splits=10)
    kf1.get_n_splits(X)
#    kf2.get_n_splits(y)
    KF_data = kf1.split(X)
#    train_targets = kf2.split(y)
    i = 1
    for train_index, test_index in KF_data:
        data_train, data_test = X[train_index], X[test_index]
        label_train, label_test = y[train_index], y[test_index]
        min_max = preprocessing.MinMaxScaler()
        data_train = min_max.fit_transform(data_train)
        data_train = 3 * data_train
#        clr = linear_model.LogisticRegression()
        weight = Newton(data_train, label_train, 1)
#        clr.fit(data_train, label_train)
        data_test = min_max.fit_transform(data_test)
        data_test = 3 * data_test
        check = np.dot(data_test, weight)
        f = Sigmoid(check)
#        print(f)
 #       f = clr.predict(data_test)
 #       f = np.mat(f).transpose()
 #       print(f)
        f[f > 0.5] = 1
        f[f < 0.5] = 0
        output = pd.DataFrame(np.hstack((np.array([[x] for x in test_index]), f)))
        output.to_csv('fold%d.csv' %i, index = False, header = None, sep = ',')
        i += 1
         
         
         
         