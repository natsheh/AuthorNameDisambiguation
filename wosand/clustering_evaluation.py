import math
import copy

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
    windowSize = 1
    differences = []
    for i in range(0,len(points)-1):
        differences.append(points[i+1]-points[i])

    #print(points)
    #print(differences)
    secondDerivative = []
    #print(range(windowSize,len(differences)-windowSize))
    for i in range(windowSize,len(differences)-windowSize):
        #print(' ', str(differences[i:i+windowSize]))
        #print(' ', str(differences[i-windowSize:i]))
        before = np.array(differences[i-windowSize:i])
        before[before<0] = 0
        secondDerivative.append(np.mean(differences[i:i+windowSize])-np.mean(before))
    #print(secondDerivative)
    #print(points[secondDerivative.index(max(secondDerivative)) + windowSize])
    return secondDerivative.index(max(secondDerivative)) + windowSize



"""
def first_derivative(c,x):
    coef = copy.deepcopy(c)
    coef.pop()
    degree = len(coef)
    exponents = range(1,degree+1)
    acum = 0
    for i in range(0,degree):
        c = coef[i]
        exponent = i+1
        acum += x**(i) * c * exponent

    return acum

def second_derivative(c,x):
    coef = copy.deepcopy(c)
    coef.pop()

    degree = len(coef)
    exponents = range(1,degree+1)
    temp = zip(coef,exponents)

    firstDerCoef = [a*b for (a,b) in temp]
    firstDerCoef.reverse()
    dummy = firstDerCoef.pop()
    firstDerCoef.reverse()
    firstDerCoef.append(dummy)

    return first_derivative(firstDerCoef,x)


def curvature(coef,x):
    return second_derivative(coef,x) / (1+first_derivative(coef,x)**2)**1.5

def elbow(x_values,coef):
    max = curvature(coef,x_values[0])
    print(str(x_values))
    index = 0
    for i in range(0,len(x_values)):
        x = x_values[i]
        print("index ", i)
        print("curvature ", curvature(coef,x_values[i]))
        if curvature(coef,x_values[i])>max:
            max = curvature(coef,x_values[i])
            #print("index ", index)
            index = i
    return index
"""

def between_clust_sim(similarity_matrix,catalog, clusters):
    cluster_acc = 0
    for clust, author_uids in clusters.items():
        for clust2, author_uids2 in clusters.items():
            if clust!=clust2:
                paper_acc = 0
                for p1 in author_uids:
                    for p2 in author_uids2:
                        paper_acc = paper_acc + similarity_matrix[catalog[p1]][catalog[p2]]
                cluster_acc = cluster_acc + paper_acc / (len(author_uids)*len(author_uids2))
    return cluster_acc/2


def within_clust_sim(similarity_matrix, catalog, clusters):
    cluster_acc = 0
    for clust, author_uids in clusters.items():
        paper_acc = 0
        for p1 in author_uids:
            for p2 in author_uids:
                if p1!=p2:
                    paper_acc = paper_acc + similarity_matrix[catalog[p1]][catalog[p2]]
        cluster_acc = cluster_acc + paper_acc / len(author_uids)**2 # maybe cluster_acc = cluster_acc + paper_acc / len(catalog)**2
    return cluster_acc
        
        
def concensus(similarity_matrix, catalog,clusters):
    k1= 1
    k2 = -.5
    return k1*within_clust_sim(similarity_matrix, catalog,clusters) + k2*between_clust_sim(similarity_matrix, catalog,clusters)
