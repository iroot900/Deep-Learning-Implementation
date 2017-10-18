import numpy as np
from prac import trainingDataPrepare
import json


#version = "V500"
version = input("model version : ")
model = np.load("model_" + version + ".npy")
if (version[0] == 'U'):
    model = model.T

toIndexFile = open("toIndexFile.txt", "r")
toWordFile = open("toWordFile.txt", "r")

toIndex = json.load(toIndexFile)
toWord = json.load(toWordFile)
print (toWord)


tests = ["american", "government", "language","december", 
        "strong", "famous", "today", "television", "nearly", 
        "einstein", "knowledge", "cannot", "difficult", 
        "help", "difficult", "force", "working", "mother", "study"]

i = 0
for word in toIndex:
#    if (i % 25 == 0):
#        tests.append(word)
    i += 1

#for key in toIndex:
#    tests.append(key)

for test in tests:
    goodVec = model[toIndex[test]]
    neibs = []
    for i in range(len(model)):
        vec = model[i]
        similarity = vec.dot(goodVec.T) / (np.linalg.norm(vec) * np.linalg.norm(goodVec))
        neibs.append([similarity, toWord[str(i)]])
    neibs.sort()

    print (test, " : ")
    for pair in neibs[-9:]:
        print ("    ", pair[1], " : " , pair[0])

