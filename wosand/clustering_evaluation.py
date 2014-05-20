import math
import numpy as np


def get_clusters_shannon_entropy(elements_in_clusters):
    counter = {}
    num_elements = 0
    #print(str(elements_in_clusters))
    for el in elements_in_clusters:
        if el not in counter.keys():
            counter[el] = 1
        else:
            counter[el] += 1
        num_elements = num_elements + 1
    return -sum(c/num_elements * math.log(c/num_elements, 2) for c in counter.values())


def evaluate(foundClusters,groundTruth):
    foundPairs = set()
    groundTruthPairs = set()

    for cluster in foundClusters:
        cluster.sort()
        for i in range(0,len(cluster)):
            for j in range(i+1,len(cluster)):
                foundPairs.add((cluster[i],cluster[j]))

    for cluster in groundTruth:
        cluster.sort()
        for i in range(0,len(cluster)):
            for j in range(i+1,len(cluster)):
                groundTruthPairs.add((cluster[i],cluster[j]))
                
    if len(foundPairs) == 0:
        precision = 0 
    else: 
        precision = float(len(foundPairs.intersection(groundTruthPairs)))/float(len(foundPairs))
    
    recall = float(len(foundPairs.intersection(groundTruthPairs)))/float(len(groundTruthPairs))
    
    if (precision + recall) != 0:
        f_measure = float(2 * precision * recall) / float(precision+recall)
    else: 
        f_measure = 0
    
    return precision, recall, f_measure


def elbow(points):
    windowSize = 3
    differences = []
    for i in range(0,len(points)-1):
        differences.append(points[i+1]-points[i])

    print(points)
    print(differences)
    secondDerivative = []
    #print(range(windowSize,len(differences)-windowSize))
    for i in range(windowSize,len(differences)-windowSize):
        #print(' ', str(differences[i:i+windowSize]))
        #print(' ', str(differences[i-windowSize:i]))
        if np.min(differences[i-windowSize:i])>=0:
            secondDerivative.append(np.mean(differences[i:i+windowSize])-np.mean(differences[i-windowSize:i]))

    #print secondDerivative
    print(points[secondDerivative.index(max(secondDerivative)) + windowSize])
    return secondDerivative.index(max(secondDerivative)) + windowSi
