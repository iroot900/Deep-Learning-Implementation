from __future__ import division

from sklearn import datasets
import numpy as np
import convnet

print ("[X], downloading data...")
mnist = datasets.fetch_mldata("MNIST Original")

index = np.array(range(0, 60000, 1)) 
testIndex = np.array(range(60000, 70000, 1)) 

np.random.shuffle(index)
np.random.shuffle(testIndex)

data  = np.array(mnist.data)
target = np.array(mnist.target)

data_test = data[testIndex]
target_test = target[testIndex]

data = data[index]
target = target[index]

n = len(data)
data.resize(n, 28 * 28) 
target.resize(n, 1)
y = np.zeros((n, 10))
for i in range(n):
        y[i][int(target[i])] = 1 

alpha = 0.00001
k = 10000
batch = 600

data.resize(n, 1, 28, 28)
data = data * 1.01
#data = data / (np.std(data, axis = 0) + 1.01)
print (data)
model = convnet.modelPre()

for i in range(k):
    pos = i % batch
    batch_index = range(pos * 100, pos * 100 + 100, 1)
    data_batch = data[batch_index]
    target_batch = target[batch_index] 
    y_batch = y[batch_index]

    cost, rate = convnet.train(data_batch, target_batch, y_batch, model)

    print ("iteration : ", i , "rate : ", rate,  "cost: " , cost)
    
