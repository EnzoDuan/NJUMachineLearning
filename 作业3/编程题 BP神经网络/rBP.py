# coding=utf-8
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0- np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self,layers,activation='tanh'):
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation=='tanh':
            self.activation = tanh
            self.activation_deriv=tanh_deriv

        self.weights=[]
        self.weights.append((2*np.random.random((layers[0]+1,layers[1]))-1)*0.25)
        for i in range(2,len(layers)):
            self.weights.append((2*np.random.random((layers[i-1],layers[i]))-1)*0.25)
            #self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)

    def fit(self,X,y,learning_rate=0.5,epochs=10000):
        X = np.atleast_2d(X)
            # atlest_2d函数:确认X至少二位的矩阵
        temp = np.ones([X.shape[0],X.shape[1]+1])
            #初始化矩阵全是1（行数，列数+1是为了有B这个偏向）
        temp[:,0:-1]=X
            #行全选，第一列到倒数第二列
        X=temp
        y=np.array(y)
            #数据结构转换
        for k in range(epochs):
                # 抽样梯度下降epochs抽样
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
                # 向前传播，得到每个节点的输出结果
            error = y[i]-a[-1]
                # 最后一层错误率
            deltas=[error*self.activation_deriv(a[-1])]

            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] +=learning_rate*layer.T.dot(delta)

    def predict(self,x):
        x=np.array(x)
        temp= np.ones(x.shape[0]+1)
        temp[0:-1]=x
        a = temp
        for l in range(0,len(self.weights)):
            a=self.activation(np.dot(a,self.weights[l]))
        return(a)
    

nn = NeuralNetwork([2,12,1],'tanh')
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
nn.fit(x,y)
for i in [[0,0],[0,1],[1,0],[1,1]]:
    print(i,nn.predict(i))