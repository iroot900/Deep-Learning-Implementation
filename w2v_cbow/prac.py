from collections import Counter
import numpy as np
import json

#words doc path
path = "./contents"

def trainingDataPrepare(path):
    dataFile = open(path, "r")

    #get all the words
    words = []
    for line in dataFile:
        wrd = line.split()
        wrd = wrd[0:500000]
        if (len(wrd) > 20):
            words.extend(wrd)

    #get top 1000 words
    counter = Counter(words)
    pairs = counter.most_common(1000 - 1)
    for item in pairs:
        print (item)

    #index word
    toIndex = {"UNK" : 0} #word to index. --- only those word will be index.
    for key in pairs:
        toIndex[key[0]] = len(toIndex)

    #reverse index
    toWord = {}
    for word in toIndex:
        toWord[toIndex[word]] = word

    #index words doc
    data = []
    for word in words:
        index = 0
        if word in toIndex:
            index = toIndex[word]
        data.append(index)

    dataFile.close()
    return words, data, toIndex, toWord

def getDerivativeV(Y_hat, Y, U, data): #this is simply derivatieve to H
    dH = (Y_hat - Y).dot(U.T)
    return data.T.dot(dH) 

def getDerivativeU(Y_hat, Y, Vc): #this is totally the same..
    return Vc.T.dot(Y_hat - Y)  #n * D  ( D * n ,  n * k)  --> D, k   

def softmax(x):
    top = np.max(x, axis = 1, keepdims = True)
    x = x - top
    x = np.exp(x) / np.exp(x).sum(axis = 1, keepdims = True)
    return x

def forward(data, V, U):
    H = data.dot(V)
    O = H.dot(U) 
    Y_hat = softmax(O)
    return H, Y_hat

def getLabelY(data, K):
    #data : 2,9,3,9,2,91,8,1,6,0
    n = len(data)
    n5 = n - 4 #windown is 2.   every 5 there is a sample..
    sample = np.zeros((n5, K))
    label = np.zeros(n5)
    Y = np.zeros((n5, K))

    for i in range(2, n - 2, 1):
        sample[i - 2][data[i - 2]] = 1
        sample[i - 2][data[i - 1]] = 1
        sample[i - 2][data[i + 1]] = 1
        sample[i - 2][data[i + 2]] = 1
        label[i - 2] = data[i]
    for i in range(n5):
        Y[i][int(label[i])] = 1

    return sample, label, Y

def cost(Y, T):
        return -(T * np.log(Y)).sum()
    

if __name__ == "__main__":
    words, data, toIndex, toWord = trainingDataPrepare(path)
    toIndexFile = open("toIndexFile.txt", "w")
    toWordFile = open("toWordFile.txt", "w")

    json.dump(toIndex, toIndexFile)
    json.dump(toWord, toWordFile)
    toIndexFile.close()
    toWordFile.close()
   
    K = 1000; D = 50
    V = np.random.uniform(-.5, .5, (K, D))
    U = np.random.uniform(-.5, .5, (D, K))

    epics = 1000
    batch_size = 100
    alpha = 0.001

    epic_cost = 0
    for epic in range(epics):
        i = 0
        np.save("model_V" + str(epic), V)
        np.save("model_U" + str(epic), U)
        print ("epic : ", epic, " cost : ", epic_cost)
        print ("-------------------------------------")
        print ("-------------------------------------")
        last_epic_cost = epic_cost
        epic_cost = 0
        while i < len(data):
            print ("epic : ", epic, " last_cost : ", last_epic_cost ,"    batch : ", i)
            data_batch = data[i : i + batch_size];  i += batch_size
            if (len(data_batch) < 100) :
                continue
            data_batch, label, Y = getLabelY(data_batch, K)

            Vc, Y_hat = forward(data_batch, V, U)
            d_V = getDerivativeV(Y_hat, Y, U, data_batch)
            d_U = getDerivativeU(Y_hat, Y, Vc)

            cst = cost(Y_hat, Y)
            print (" cost : ", cst) 
            epic_cost += cst

            U = U - alpha * d_U
            V = V - alpha * d_V
            
