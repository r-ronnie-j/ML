from random import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import math

def k_nearest_neighbour(data,predict,k=None):
    if(k==None):
        k =2*len(data)+1
    distance =[]
    for i in data:
        for ii in data[i]:
            for sample in ii:
                d = np.linalg.norm(np.array(sample)-np.array(predict))
                distance.append([d,i])
    
    distance.sort()
    valid = (j[1] for j in distance[:k])
    act_valid = Counter(valid).most_common(1)
    return (act_valid[0][0],act_valid[0][1]/k)

data = pd.read_csv('breast-cancer-wisconsin.csv')
data.replace('?',-99999,inplace=True)
data.drop(['Sample code number'],1,inplace=True)
data = data.astype('float')
X= np.array(data.drop(['Class'],1))
Y= np.array(data['Class'])

X_train = X[:-math.ceil(-len(X)*0.1)]
Y_train = Y[:-math.ceil(-len(X)*0.1)]
X_test = X[-math.ceil(-len(X)*0.1):]
Y_test = Y[-math.ceil(-len(X)*0.1):]

X_datatrain={2:[],4:[]}
X_datatest={2:[],4:[]}

for i in range(len(X_train)):
    X_datatrain[Y_train[i]].append(X_train[i])

for i in range(len(X_train)):
    X_datatest[Y_test[i]].append(X_test[i])

correct = 0
total = 0
for i in range(len(X_test)):
    predict = k_nearest_neighbour(X_datatrain,X_test[i],k=19);
    if(predict[0] == Y_test[i]):
        correct+=1
    total+=1

accuracy = correct/total
print(accuracy)