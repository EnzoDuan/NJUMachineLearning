# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:11:07 2017

@author: djh
"""

import numpy as np

data = np.array((1,2,3,4))
print(data)
data = np.mat(data)
print(data.T)
data2 = np.mat((1,2,3,5))
print(np.dot(data.T, data2))