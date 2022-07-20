from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd;


class linearRegression :
    def find_meanX(self,X):
        (sample, feature)= np.shape(X);
        mean = np.zeros(feature);
        j = sum(i for i in X)
        j = j+mean
        mean = j/sample
        return np.array([mean])

    def find_meanY(self,Y):
        mean=np.zeros(1)
        j = sum(i for i in Y)
        j = j+mean
        mean = j/len(Y)
        mean = np.array([mean])
        return mean

    def regression(self,x, y):
        x_bar = self.find_meanX(x)
        y_bar = self.find_meanY(y)
        y= np.array([y]).transpose();
        print(np.shape(x_bar),np.shape(y_bar),np.shape(x),np.shape(y))       
        param = np.matmul(np.matmul(y_bar,x_bar)-np.matmul(y.transpose(), x),
                      np.linalg.inv(np.matmul(x_bar.transpose(), x_bar)-np.matmul(x.transpose(), x)))
        return param;

    def fit(self,X, Y):
        params = self.regression(X, Y);
        self.params = params
        return
    
    def predict(self,x):
        result = np.zeros(len(x))
        for i in range(len(x)-1):
            result[i]= np.dot(x[i],self.params[0])
        return result

