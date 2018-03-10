# -*- coding: utf-8 -*-
"""
@author: djh
"""

# We have 400 input layer's neurals
# We have 100 hidden layer's neurals
# We have 10 output layer's neurals , they are used to output different handwritten numbers

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd

def Sigmoid(x):  
    return 1/(1+np.exp(-x))  

def DSigmoid(x):  
    return Sigmoid(x)*(1-Sigmoid(x)) 
    
class BPNerualNetwork:
#    for data in train_data:
    def __init__(self, Layer_input, Layer_hid, Layer_output):

        self.ItoHweights = (0.1 * np.random.random((Layer_hid, Layer_input))-0.05)
        self.HtoOweights = (0.1 * np.random.random((Layer_output, Layer_hid))-0.05)
        self.HidBias = np.zeros((100, 1))
        self.OutBias = np.zeros((10, 1))

        
        self.learnRate = 0.3
        train_data = np.genfromtxt('train_data.csv', delimiter = ',', skip_header = 0, dtype = float)
        targets = np.genfromtxt('train_targets.csv', delimiter = ',', skip_header = 0, dtype = int)
        Kf = KFold(n_splits = 5)
        Kf.get_n_splits(train_data)
        Kf_data = Kf.split(train_data)
        for train_index, test_index in Kf_data:
            self.train_data, self.test_data = train_data[train_index], train_data[test_index]
            self.train_targets, self.test_targets = targets[train_index], targets[test_index]
    
    def Train(self):
        """
            每次所有的训练结束需要有一个循环终止条件,如果训练集误差减小，测试集的误差增加，return
        """
        preTrainError = 100000
        preTestError  = 100000
        Ek_train = 50000
        Ek_test = 50000
        n = 0
        while(n < 10):
            n += 1
            preTrainError = Ek_train
            preTestError = Ek_test
            Ek_train_t = []
            Ek_test_t = []
            num1 = 0
            num2 = 0
            for data,label in zip(self.train_data, self.train_targets):
                # Step1: Compute Yj - Output
                num1 += 1
                bh = []
                data = np.mat(data)
                for i in range(100):
                    weights = np.mat(self.ItoHweights[i])
                    total = np.dot(data, weights.T)
                    out = total - self.HidBias[i].T
                    out = float(Sigmoid(out))
                    bh.append(out)
                    
                # alpha is hidden layer's output
                
                myYk = []
                for i in range(10):
                    weights = np.mat(self.HtoOweights[i])
                    total = np.dot(bh, weights.T)
                    out = total - self.OutBias[i].T
                    out = float(Sigmoid(out))
                    myYk.append(out)
                
                # myYk is Y- output
                # Error computation
                yk = np.zeros(10)
                yk[label] = 1
                t = myYk - yk;
                Ek_train_t.append(float(0.5*np.dot(t, t.T)))
                # 计算输出层梯度项
                Gi = []
                for i in range(10):
                    Gi.append(float(myYk[i] * (1 - myYk[i]) * (yk[i] - myYk[i])))
                # 计算隐层神经元梯度项
                Eh = []
                Gi = np.mat(Gi)
                W = self.HtoOweights.transpose()
                for i in range(100):
                    Eh.append(float(bh[i] * (1 - bh[i]) * np.dot(W[i], Gi.T)))

                #计算所有的更新值
                bh = np.mat(bh)
                Eh = np.mat(Eh)
                delta_HtoOweights = self.learnRate * np.dot(Gi.T, bh)
                self.HtoOweights += delta_HtoOweights
                delta_Outbias = -self.learnRate * Gi
                self.OutBias += delta_Outbias.T
                delta_ItoOweights = self.learnRate * np.dot(Eh.T, data)
                self.ItoHweights += delta_ItoOweights
                delta_Hidbias = -self.learnRate * Eh
                self.HidBias += delta_Hidbias.T
                
            # 用测试集进行验证
            for test, label in zip(self.test_data, self.test_targets):
                num2 += 1
                bh = []
                test = np.mat(test)
                for i in range(100):
                    weights = np.mat(self.ItoHweights[i])
                    total = np.dot(test, weights.T)
                    out = total - self.HidBias[i].T
                    out = float(Sigmoid(out))
                    bh.append(out)
                    
                # alpha is hidden layer's output
                
                myYk = []
                for i in range(10):
                    weights = np.mat(self.HtoOweights[i])
                    total = np.dot(bh, weights.T)
                    out = total - self.OutBias[i].T
                    out = float(Sigmoid(out))
                    myYk.append(out)
                # myYk is Y- output
                # Error computation
                yk = np.zeros(10)
                yk[label] = 1
                t = myYk - yk;
                Ek_test_t.append(float(0.5*np.dot(t, t.T)))
            Ek_train_t = np.array(Ek_train_t)
            Ek_train = np.sum(Ek_train_t)/num1
            Ek_test_t = np.array(Ek_test_t)
            Ek_test = np.sum(Ek_test_t)/num2
                
    def TestModel(self):
        label = []
        test_data = np.genfromtxt('test_data.csv', delimiter = ',', skip_header = 0, dtype = float)
        for test in test_data:
            bh = []
            test = np.mat(test)
            for i in range(100):
                weights = np.mat(self.ItoHweights[i])
                total = np.dot(test, weights.T)
                out = total - self.HidBias[i].T
                out = float(Sigmoid(out))
                bh.append(out)
            
            # alpha is hidden layer's output
            
            myYk = []
            for i in range(10):
                weights = np.mat(self.HtoOweights[i])
                total = np.dot(bh, weights.T)
                out = total - self.OutBias[i].T
                out = float(Sigmoid(out))
                myYk.append(out)
            maxnum = -1
            k = -1
            for i in range(10):
                if myYk[i] > maxnum:
                    maxnum = myYk[i]
                    k = i
            
            label.append(k)
        
        output = pd.DataFrame((label))
        output.to_csv('test_predictions_library.csv', index = False, header = None, sep = ',')
        
        
if __name__ == '__main__':
    Layer_input = 400
    Layer_hid = 100
    Layer_output = 10
    BPNN = BPNerualNetwork(Layer_input, Layer_hid, Layer_output)
    BPNN.Train()
    print('Train Done')
    BPNN.TestModel()
#    train_data = np.genfromtxt("train_data.csv")
    
#    plt.imshow(x.reshape((20,20),order='F'),cmap='gray')
#    plt.show()