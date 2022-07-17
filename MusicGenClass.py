import pandas as pd
import numpy as np
from python_speech_features import mfcc #Mel frequency cepstral coefficient
import scipy.io.wavfile as wav

from tempfile import TemporaryFile

import os
import pickle
import random
import operator

import math


# function to get the distance between feature vecotrs and find neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    # gives the distance from inst to all possible data from trainingSet
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    #group the k-Neareset neighbour into the neighbour list
    for x in range(k):
        neighbors.append(distances[x][0])
    
    return neighbors #return list


# identify the class of the instance/neighbour
def nearestClass(neighbors):
    classVote = {} #dictionary
#matlab agar hmre pass neighbour list me let 10 element h jisme se kuch class a se belong h kuch class b se kuch class c
# se to hme count krna h konse element k lie class value max aaai diction = output(class with max value)
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

#sorted dictionary in sorted array after that return value at index [0][0]
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)

    return sorter[0][0]

def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    
    return (1.0 * correct) / len(testSet)

# directory that holds the dataset

# All the wav file getting read here 
directory = "D:\Music_genre_Classification\Data\genres_original/"
#binary file where we will collect all the feature extracted using mfcc
f = open("my.dat", 'wb')

i = 0

for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+folder):        
        try:
            (rate, sig) = wav.read(directory+folder+"/"+file)
            #specfies the feature to categories the file
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            #once we have the features identify we will dumpinf feature into the my.dat binary file
            pickle.dump(feature, f)
        except Exception as e:
            print('Got an exception: ', e, ' in folder: ', folder, ' filename: ', file)        

f.close()



# There are different approaches to do train test split. here I am using a random module and running a 
# loop till the length of a dataset
dataset = []
def loadDataset(filename, split,trSet, teSet):
    with open("my.dat",'rb') as f:
        #append data randomly into the dataset list because we want catergorized data in such a way that all category
        #data will get available into the both test and train set
        while True:
            try:
                #pickle : -it's the process of converting a Python object into a byte stream to store it in a file/database
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break

    for x in range(len(dataset)):
        #randomly append data into the train set from dataset List which me make above onlhy 66%
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])

trainingSet = []
testSet = []

loadDataset("my.dat", 0.66, trainingSet, testSet)

def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

leng = len(testSet)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)

# testing the code with external samples
# URL: https://uweb.engr.arizona.edu/~429rns/audiofiles/audiofiles.html

test_dir = "D:\Music_genre_Classification\Test/"
test_file = test_dir + "test.wav"
# test_file = test_dir + "test2.wav"
# test_file = test_dir + "test4.wav"

#extract the feature from the test_file(main purpose)
(rate, sig) = wav.read(test_file)
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature = (mean_matrix, covariance, i)



from collections import defaultdict
results = defaultdict(int) #dictionary 

directory = "D:\Music_genre_Classification\Data\genres_original/"

# it make the result where id will be link with the genre name
i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1

# predict the result from dataset and feature we get from test_file
# here nearestClass return the in vallue that will consider as id to the dictionary result
# and we will print the value with key(id) as you can see below
pred = nearestClass(getNeighbors(dataset, feature, 5))
print(results[pred])