import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import nltk
import math

#default distance value: if aggregating by sum-0.5 if by max-1
DEFAULT_VALUE = 0.5

def get_min_edit_dist(str1, str2):
    if not str1 or not str2:
        return DEFAULT_VALUE, True
    set1 = str1.split()
    set2 = str2.split()
    if len(set1) > len(set2):
        set1, set2 = set2, set1
    distance_strings = 0
    for word1 in set1: #loop words in attribute field of paper1
        min_dist_words = np.inf
        for word2 in set2: #loop words in attribute field of paper2
            dist = nltk.distance.edit_distance(word1, word2)
            if dist < min_dist_words: #store the minimal distance
                min_dist_words = dist
        distance_strings = distance_strings + min_dist_words  #math.pow(min_dist_words, 2)
    return distance_strings, False #math.sqrt(distance_strings)


def get_syllabes_jaccard_dist(str1, str2):
    if not str1 or not str2:
        return DEFAULT_VALUE, True
    tmp1 = str1.split()#set()
    tmp2 = str2.split()#set()
    tmp1.remove(tmp1[0])
    tmp2.remove(tmp2[0])
    set1 = set(''.join(tmp1))
    set2 = set(''.join(tmp2))
    if not set1 or not set2:
        return DEFAULT_VALUE, True
    distance_strings = nltk.distance.jaccard_distance(set1, set2)
    return distance_strings, False
    

def get_exact_match_dist(str1, str2):
    if not str1 or not str2:
        return DEFAULT_VALUE, True
    set1 = str1.split()
    set2 = str2.split()
    if len(set1) > len(set2):
        set1, set2 = set2, set1
    distance_strings = 0
    for word1 in set1: #loop words in attribute field of paper1
        min_dist_words = np.inf
        for word2 in set2: #loop words in attribute field of paper2
            if word1.strip() == word2.strip(): dist = 0
            else: dist = 1
            if dist < min_dist_words: #store the minimal distance
                min_dist_words = dist
        distance_strings = distance_strings + min_dist_words  #math.pow(min_dist_words, 2)
    return distance_strings, False #math.sqrt(distance_strings)


def get_at_least_one_dist(str1, str2):
    if not str1 or not str2:
        return DEFAULT_VALUE, True
    dist = 1
    set1 = str1.split()
    set2 = str2.split()
    for word1 in set1:
        for word2 in set2: 
            if word1.strip() == word2.strip(): 
                dist = 0
                break
        if dist is 0:
            break
    return dist, False 
                 

def get_cosine_dist(corpus):
    try:
        #calculate tf-idf matrix in 1 step
        #bring the feature values closer to a Gaussian distribution, compensating for LSA’s erroneous assumptions about textual data
        tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        #reduces the space by half the number of variables
        #tfidf_lsa =  TruncatedSVD(n_components=int((tfidf_matrix.shape[1])*0.75), algorithm='randomized', n_iterations=5, random_state=None, tol=0.0)
        #tfidf_reduced_matrix = tfidf_lsa.fit_transform(tfidf_matrix)
        #print("SAMPLES: %d\nFEATURES: %d" % tfidf_reduced_matrix.shape)
        distance_mat = 1 - pairwise_distances(tfidf_matrix.todense(), metric='cosine')
        #return np.nan_to_num(distance_mat), False
        distance_mat[np.isnan(distance_mat)] = 0
        return distance_mat, False
    except ValueError as ve:
        print ("Error with corpus: {0}".format(corpus))
        return np.ones([len(corpus), len(corpus)]) * DEFAULT_VALUE,  True 
