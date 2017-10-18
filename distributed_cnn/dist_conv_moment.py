from __future__ import division

from pyspark import SparkConf, SparkContext

import numpy as np
import convnet
import layers

conf = SparkConf().set("spark.authenticate.secret", "1234")
sc = SparkContext(conf = conf)
sc.setLogLevel("ERROR")

def dataPrepare(iterator):
    rawData = list(iterator); n = len(rawData)
    k = len(rawData[0].split(","))
    if (k != 28 * 28 + 1) :
      raise Exception("bad things")

    data = np.zeros((n, k)); i = 0
    for sample in rawData:
        data[i] = np.fromstring(sample, sep=","); i += 1
    data 
    
    return [data]

#gradients aggregates
def add(x, y):
    gradients_a, cost_a, rate_a = x
    gradients_b, cost_b, rate_b = y
    gradients = {}
    for key in gradients_a:
      gradients[key] = gradients_a[key] + gradients_b[key]
    return [gradients, cost_a +  cost_b, rate_a + rate_b]

def addNum(x, y):
    return x + y

dataPath = "/tmp/mnist/"
rawData = sc.textFile(dataPath + "./mnistTrainingData", 40)
testRawData = sc.textFile(dataPath + "./testDataMnist", 40)
data = rawData.mapPartitions(dataPrepare)
testData = testRawData.mapPartitions(dataPrepare)
data.persist()

rate_test_last = 0.0
#l2 regulartion  
decayRate = 0.01# 1 / 200

alpha = 0.0001
k = 100000
model = convnet.modelPre()

#sgd with momentum 
mu = 0.1;  v = {}

for key in model:
  v[key] = np.zeros_like(model[key])

for i in range(k):
    #spark driver is acting as parameter server -- gather gradients then broadcast updated weights
    gradientsByPartition = data.map(lambda x : convnet.train(x, i, model, alpha))
    gradients, cost, rate = gradientsByPartition.reduce(add)
    for key in model:
        if (key.startswith("b")):
            model[key] -= alpha * gradients[key]
        else:
            v[key] = mu * v[key] - alpha * (gradients[key] + decayRate *  model[key]) 
            model[key] += v[key]

    rate_test = 0.00
    
    #estimate test rate
    if (i % 300 == 1):
        count = testData.map(lambda x : convnet.getTestRightCount(x, model)).reduce(addNum)
        rate_test = count / 10000.0
        rate_test_last = rate_test

    print ("iteration : ", i , "200 sample rate : ", rate / 40.0,  " 10000 test_rate : ", rate_test_last,  "200 sample cost: " , cost)
