# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:01:58 2017

@author: djh
"""

import numpy as np #数值计算包
import matplotlib.pyplot as plt #绘图包
import seaborn as sns # 绘图美化包 用于配色

data = [[6.9, 3.1, 1.0], [5.0, 3.5, 0.0], [5.0, 2.3, 1.0], [4.6, 3.4, 0.0], [5.5, 2.4, 1.0], [5.5, 3.5, 0.0], [5.2, 4.1, 0.0], [4.7, 3.2, 0.0], [5.4, 3.9, 0.0], [5.6, 3.0, 1.0], [6.1, 2.8, 1.0], [5.8, 2.7, 1.0], [6.2, 2.2, 1.0], [5.0, 3.4, 0.0], [5.7, 3.8, 0.0], [6.3, 2.5, 1.0], [6.0, 3.4, 1.0], [4.9, 3.1, 0.0], [4.9, 3.0, 0.0], [5.6, 2.7, 1.0], [5.7, 2.8, 1.0], [5.0, 3.2, 0.0], [5.9, 3.0, 1.0], [6.1, 2.8, 1.0], [5.1, 3.5, 0.0], [4.6, 3.2, 0.0], [5.1, 2.5, 1.0], [5.1, 3.7, 0.0], [5.4, 3.7, 0.0], [5.3, 3.7, 0.0], [4.8, 3.0, 0.0], [4.3, 3.0, 0.0], [4.9, 2.4, 1.0], [5.5, 2.5, 1.0], [5.8, 4.0, 0.0], [5.1, 3.8, 0.0], [5.7, 2.8, 1.0], [5.1, 3.8, 0.0], [5.0, 3.0, 0.0], [4.4, 3.0, 0.0], [5.4, 3.4, 0.0], [5.7, 2.6, 1.0], [6.4, 3.2, 1.0], [5.2, 3.5, 0.0], [4.8, 3.4, 0.0], [6.0, 2.7, 1.0], [5.6, 3.0, 1.0], [5.1, 3.3, 0.0], [5.4, 3.9, 0.0], [6.8, 2.8, 1.0], [5.9, 3.2, 1.0], [4.5, 2.3, 0.0], [6.1, 3.0, 1.0], [4.4, 2.9, 0.0], [5.4, 3.4, 0.0], [5.5, 2.4, 1.0], [5.6, 2.9, 1.0], [5.5, 2.6, 1.0], [5.7, 4.4, 0.0], [5.0, 3.6, 0.0], [5.0, 3.4, 0.0], [4.6, 3.6, 0.0], [5.7, 2.9, 1.0], [5.0, 2.0, 1.0], [6.3, 2.3, 1.0], [7.0, 3.2, 1.0], [4.6, 3.1, 0.0], [5.0, 3.3, 0.0], [5.1, 3.8, 0.0], [4.9, 3.1, 0.0], [4.7, 3.2, 0.0], [5.5, 2.3, 1.0], [6.7, 3.1, 1.0], [6.5, 2.8, 1.0], [6.7, 3.0, 1.0], [4.4, 3.2, 0.0], [6.6, 2.9, 1.0], [5.0, 3.5, 0.0], [6.4, 2.9, 1.0], [5.1, 3.5, 0.0], [6.3, 3.3, 1.0], [5.4, 3.0, 1.0], [6.2, 2.9, 1.0], [5.1, 3.4, 0.0], [5.2, 3.4, 0.0], [4.8, 3.0, 0.0], [6.0, 2.9, 1.0], [5.5, 4.2, 0.0], [4.8, 3.4, 0.0], [5.8, 2.6, 1.0], [5.8, 2.7, 1.0], [6.7, 3.1, 1.0], [4.8, 3.1, 0.0], [5.7, 3.0, 1.0], [6.1, 2.9, 1.0], [6.0, 2.2, 1.0], [6.6, 3.0, 1.0], [4.9, 3.1, 0.0], [5.6, 2.5, 1.0], [5.2, 2.7, 1.0]]

def preprocess(data):
    # labels 用来标识每个数据的类别:0或1
    labels = [] 
    # train_X 输入数据列表: [[1.0, x_i1, x_i2],..., [1.0, x_n1, x_n2]]
    train_X = [] 
    for x in data:
        # 二维平面的直线方程为： ax + by + c = 0，因此我们需要再原来的输入数据中增加一常数项 1.0
        train_X.append([1.0, x[0], x[1]])
        labels.append(x[2])
    return np.array(train_X), np.array(labels)


def logit(z):
    '''
        logistic function 即：
        牛顿迭代公式中的p(yi=1 | xi; w)
    '''
    return 1.0 / (1.0 + np.exp(-z))
    
def grad(train_X, labels, max_iter = 500):
    X = np.mat(train_X) # 100 行 3列
    Y = np.mat(labels).transpose() # 100 行 1列
    rows, columns = np.shape(X)
    # 初始回归系数 weights 为全1的向量, weights的行数与输入的训练数据维度相同
    weights = np.ones((columns, 1))
    # 步进alpha: 0.001
    alpha = 0.001
    # t 表示当前迭代的次数
    t = 0
    while t < max_iter:
        P1 = logit(np.dot(X, weights))
        weights += alpha * X.T * (Y - P1)
        t += 1
    
    return weights

def newton(train_X, labels, max_iter = 10):
    X = np.mat(train_X)
    Y = np.mat(labels).transpose()
    rows, columns = np.shape(X)
    # 初始回归系数 weights 为全0的向量, weights的行数与输入的训练数据维度相同。
    weights = np.zeros((columns, 1))
    # t 表示当前迭代的次数
    t = 0
    while t < max_iter:
        P1 = logit(np.dot(X, weights))
        gradient =  np.dot(X.T, (Y - P1))
        P_mat = np.array(P1) * np.array(P1 - 1) * np.eye(rows)
        hessian = X.T * P_mat * X
        weights -= np.linalg.inv(hessian) * gradient
        t += 1
    
    return weights
    
# 牛顿法
weights = newton(preprocess(data)[0], preprocess(data)[1])
print("weight", weights)
# 梯度法
#weights = grad(preprocess(data)[0], preprocess(data)[1])
# 根据数据的类别，得到两个数据集
x0=list()
y0=list()
x1=list()
y1=list()
for i in data:
    if i[-1] == 0:
        x0.append(i[0])
        y0.append(i[1])
    else:
        x1.append(i[0])
        y1.append(i[1])

# c=sns.color_palette()[1] ： seaborn 包中的配色方案
plt.scatter(x0, y0, marker='^', c=sns.color_palette()[1])
plt.scatter(x1, y1, marker='o', c=sns.color_palette()[2])

x = np.arange(4.0, 7.0, 0.1)
# 画出回归直线： w0 + w1*x1 + w2*x2 = 0, 我们定义x1为横轴坐标，x2为纵轴坐标
y = (-weights[0] - weights[1]*x)/weights[2]
plt.plot(x, y, color=sns.color_palette()[0])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()