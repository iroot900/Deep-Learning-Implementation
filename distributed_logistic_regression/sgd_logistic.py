from pyspark import SparkConf, SparkContext

import numpy as np
import time

def trainingDataPrepare(iterator):
	rawData = list(iterator); n = len(rawData)
	k = len(rawData[0].split(","))
	data = np.zeros((n, k)); i = 0
	for sample in rawData:
	  data[i] = np.fromstring(sample, sep=","); i += 1
	
	#print (data.shape)
	return [data]

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def getGradient(data, i, w, b):
	print ("exectuor : ", i )
	print ("w : ", w )
	print ("b : ", b )
	n = data.shape[0]
	t = 50
	start = i * t; start = start % n; end = start + t
	data = data[start : end , :]
	X = data[:,:-1]
	Y = np.array(data[:,-1])
	Y.resize(Y.shape[0], 1)
	Z = X.dot(w) + b
	Y_pred = sigmoid(Z)
	return [X.T.dot(Y_pred - Y), (Y_pred - Y).sum()]

def addGrad(dw1, dw2):
	return [dw1[0] + dw2[0], dw1[1] + dw2[1]]

def getCount(data, w, b):
	X = data[:,:-1]
	Y = np.array(data[:,-1])
	Y.resize(Y.shape[0], 1)
	Z = X.dot(w) + b
	Y_pred = sigmoid(Z)
	Y_hat_arg = Y_pred * (Y_pred > 0.5)
	Y_hat_arg = np.ones_like(Y_pred) * (Y_hat_arg > 0.5)
	Y_hat_arg.resize(Y.shape)
	n = Y.shape[0]; count = 0
	for i in range(n):
		if (Y_hat_arg[i] == Y[i]):
			count += 1
	return count

#spark session 
conf = SparkConf().set("spark.authenticate.secret", "1234")
sc = SparkContext(conf = conf)
#sc.setLogLevel("INFO")
sc.setLogLevel("ERROR")

#data preparation
rawData = sc.textFile("trainingData", 2)
data = rawData.mapPartitions(trainingDataPrepare)
print (data.count())

#model init
w = np.random.randn(5, 1)
b = 1

alpha = 0.0001; k = 30
n = rawData.count()

begin = time.time()
data.cache()
#training
for i in range(k) : 
  gradw_b = data.map(lambda x : getGradient(x, i, w, b)).reduce(addGrad)
  w = w - alpha * gradw_b[0]
  b = b - alpha * gradw_b[1]
  print ("driver : ", i )
  print ("w : ", w )
  print ("b : ", b )
  #evaluate
  count = data.map(lambda x : getCount(x, w, b)).reduce(lambda a, b : a + b)
  print ((float)(count)/n)

during = time.time() - begin
print ("takes : ", during, "s")
