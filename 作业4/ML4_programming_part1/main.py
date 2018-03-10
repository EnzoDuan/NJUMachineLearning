# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:23:14 2017

@author: djh
"""
#import sklearn 
import numpy as np
def predict(X, model):
    ans = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        # from evaluate.py I can see gamma, so it only needs to add a support vector
        s = 1 / np.exp(model.gamma * np.sum((model.support_vectors_ - X[i])**2, axis=1))
        s = np.sum(s * model.dual_coef_[0]) + model.intercept_
        ans[i] = 1 if s > 0 else 0
    return ans