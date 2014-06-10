import fileinput
import re
import time
import collections

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from nltk import stem

from connection import Connection
from sql_queries_production import PAPER_ALL_INFO
from graph_queries_production import COAUTHORSHIP_GRAPH_MATRIX
from graph_queries_production import REFERENCES_GRAPH_MATRIX
from graph_queries_production import SUBJECTS_GRAPH_MATRIX
from graph_queries_production import KEYWORDS_GRAPH_MATRIX

from classes import Paper

from distances import *

EXCLUDE = ['!', '?', '.', ',', ':', ';', '_', '-', '+', '*', '/', '\\', '^', 
"'", '"', '’' , '(', ')', '[', ']', '=', '°', '|', '{', '}', '\n', '\t',
'%', '¡', '¨', '“', '”', '`', '<', '>', '$', '&', '@', '#', '°']

STOP_WORDS = ['a','able','about','across','after','all','almost','also','am','among',
'an','and','any','are','as','at','be','because','been','but','by','can',
'cannot','could','dear','did','do','does','either','else','ever','every',
'for','from','get','got','had','has','have','he','her','hers','him','his',
'how','however','i','if','in','into','is','it','its','just','least','let',
'like','likely','may','me','might','most','must','my','neither','no','nor',
'not','of','off','often','on','only','or','other','our','own','rather','said',
'say','says','she','should','since','so','some','than','that','the','their',
'them','then','there','these','they','this','tis','to','too','twas','us',
'wants','was','we','were','what','when','where','which','while','who',
'whom','why','will','with','would','yet','you','your']

PREFIXES = ['von', 'de', 'vant', 'van', 'der', 'vom', 'vander', 'zur',
'ten', 'la', 'du', 'ter', 'dos', 'al', 'del', 'st', 'le', 'dos', 'da', 
'do', 'mc', 'des', 'den', 'di', 'abu', 'vander', 'den', 'della', 'vande', 
'dit', 'bin', 'ibn', 'el', 'los', 'dello', 'vanden', 'ap', 'las', 'delli', 
'mac', 'mrs', 'mr', 'miss', 'jr', 'sr', 'II', 'III', 'IV', 'in']

#default distance value: if aggregating by sum-0.5 if by max-1
DEFAULT_VALUE = 0.5

#return the cosine similarity matrix (institutions) among papers and the library object
#that keeps track of the association: matrix line --> paper/author information 
def get_library(focus_name, debug):
    #initialization
    all_info_query = PAPER_ALL_INFO.format(focus_name)
    conn = Connection()
    paper_tracker = 0
    result_row_count = 0
    library = []
    #catalog keeps track of the association: paper_id+author_id --> matrix_line
    catalog = {}
    conn.set_group_limit(1000000)
    #try:
    sql_results = conn.execute(all_info_query)
    if sql_results:
        for el in sql_results:
            #print(el[1],el[2],el[3],el[4],el[5],el[6],el[7],el[8],el[9],el[10],el[11]) 
            
            #feature cleaning     
            author_id = str(el[0]) #the disambiguated one for test purposes
            author_name = preprocessing(el[1], stemming=False, stop_words=PREFIXES, min_word_length=0)
            paper_id = el[2]
            paper_title = preprocessing(el[3], stemming=True, stop_words=STOP_WORDS, min_word_length=2)
            year = preprocessing(el[4], stemming=True, stop_words=STOP_WORDS, min_word_length=2)
            coauthors = preprocessing(el[5], stemming=False, stop_words=PREFIXES, min_word_length=0).replace(author_name, "") #remove the author himself
            subjects = preprocessing(el[6], stemming=True, stop_words=STOP_WORDS, min_word_length=2)
            keywords = preprocessing(el[7], stemming=True, stop_words=STOP_WORDS, min_word_length=2)
            journals = preprocessing(el[8], stemming=True, stop_words=STOP_WORDS, min_word_length=0)
            institutions = preprocessing(el[9], stemming=True, stop_words=STOP_WORDS, min_word_length=2)
            ref_authors = preprocessing(el[10], stemming=False, stop_words=PREFIXES, min_word_length=0)
            ref_journals = preprocessing(el[11], stemming=True, stop_words=STOP_WORDS, min_word_length=2)
            countries = preprocessing(el[12], stemming=True, stop_words=STOP_WORDS, min_word_length=2).replace("franc", "") #remove france
            
            #unique identifier is paper_id concat author_id (solve the problem of 2 authors with the same name in the same paper)
            unique_identifier = str(paper_id) + str(author_id)
            
            #printout for production environment
            printout = str(el[3]) + " [" + str(el[4]) + "]"
            
            #add to library
            library.insert(paper_tracker, 
                Paper(paper_id, paper_title, author_id, author_name, coauthors, 
                    institutions, journals, year, subjects, keywords, ref_authors,
                    ref_journals, countries, unique_identifier, printout))
            catalog[unique_identifier] = paper_tracker
            paper_tracker = paper_tracker + 1
            if debug:
                print("{0} - AuthorID: {1} - PaperID: {2} - Title:{3}".format(paper_tracker, author_id, paper_id, author_name)) 
                
    #close connection to db
    conn.close()
    return library, catalog


def update_library_with_mask_blocks(library, catalog, mask_matrix):
    mask_block_id
    #loop paper1
    for i in range(len(library)-1):         
        p1 = library[i]
        #loop paper2
        for j in range(i+1, len(library)):  
            p2 = library[j]
            if error: 
                index_error[i].append(j)
            else:
                mat_result[catalog[p1.unique_identifier], catalog[p2.unique_identifier]] = dist_string
                mat_result[catalog[p2.unique_identifier], catalog[p1.unique_identifier]] = dist_string


def get_string_distance_matrix(library, catalog, field, distance_type):
    mat_result = np.zeros((len(library), len(library)))
    error = False
    index_error = collections.defaultdict(list)
    
    #set distance function type
    if distance_type is "min_edit":
        f = lambda x,y: get_min_edit_dist(getattr(x, field), getattr(y, field))
    elif distance_type is "syllab_jaccard":
        f = lambda x,y: get_syllabes_jaccard_dist(getattr(x, field), getattr(y, field))
    elif distance_type is "exact_match":
        f = lambda x,y: get_exact_match_dist(getattr(x, field), getattr(y, field))
    elif distance_type is "at_least_one":
        f = lambda x,y: get_at_least_one_dist(getattr(x, field), getattr(y, field))    
        
    #loop paper1
    for i in range(len(library)-1):         
        p1 = library[i]
        #loop paper2
        for j in range(i+1, len(library)):  
            p2 = library[j]
            dist_string, error = f(p1,p2) 
            if error: 
                index_error[i].append(j)
            else:
                mat_result[catalog[p1.unique_identifier], catalog[p2.unique_identifier]] = dist_string
                mat_result[catalog[p2.unique_identifier], catalog[p1.unique_identifier]] = dist_string
    
    #max distance for normalization
    max_dist = np.max(mat_result) if mat_result != [] else 0
    #normalize [0,1]
    if max_dist != 0:
        mat_result = np.divide(mat_result, max_dist)
    #correct errors with DEFAULT_VALUE distance
    for i in index_error:
        for j in index_error[i]:
            mat_result[i][j] = DEFAULT_VALUE
    #subtract matrices from 1 to get similarity and not difference
    m = 1 - mat_result
    return m


def get_cosine_distance_matrix(library, catalog, field, distance_type, mask_matrix=None):
    #if mask is not provided use a matrix of ones
    if mask_matrix is None:
        mask_matrix = np.ones((len(library), len(library)))
    
    #initialize
    mat_result = np.identity(len(library))
    error = False
    index_error = collections.defaultdict(list)
    
    #tfid cosine distance on mask blocks 
    if distance_type is "cosine":
        block_tracker = []
        for p1 in library: #loop paper1
            if p1.unique_identifier in block_tracker:
                continue
            corpus = []
            block_tracker.append(p1.unique_identifier)
            block_mask = np.zeros(mat_result.shape)
            #corpus.append(getattr(p1, field))
            for p2 in library: #loop paper2
                #if p1 == p2: continue
                if mask_matrix[catalog[p1.unique_identifier]][catalog[p2.unique_identifier]] > 0.4:
                    block_mask[catalog[p1.unique_identifier]][catalog[p2.unique_identifier]] = 1
                    block_tracker.append(p2.unique_identifier)
                    corpus.append(getattr(p2, field))
                else:
                    corpus.append("")
            #calculate tfidf on the block
            mat_block, error = get_cosine_dist(corpus)
            mat_block[np.diag_indices_from(mat_block)] = 0
            #add to result_matrix with max
            #print(mat_block)
            mat_result = np.maximum(mat_result, mat_block)

    
    #this technique requires to set diagonal to 1s
    return mat_result


def get_graph(library, catalog, focus_name, type):
    graph_types = {
        "coauthorship": COAUTHORSHIP_GRAPH_MATRIX,
        "references": REFERENCES_GRAPH_MATRIX,
        "subjects": SUBJECTS_GRAPH_MATRIX,
        "keywords": KEYWORDS_GRAPH_MATRIX
    }
    if type in graph_types:
        mat_graph = np.identity(len(library))
        graph_query = graph_types[type].format(focus_name)
        #print(graph_query)
        conn = Connection()
        sql_results = conn.execute(graph_query, mult=True)
        if sql_results:
            for el in sql_results:
                unique_identifier1 = str(el[0]) + str(el[1])
                unique_identifier2 = str(el[2]) + str(el[3])
                mat_graph[catalog[unique_identifier1],catalog[unique_identifier2]] = el[4]
                mat_graph[catalog[unique_identifier2],catalog[unique_identifier1]] = el[4]
        conn.close()
        return mat_graph
    else:
        return None
        #error catch here

def preprocessing(text, stemming=False, min_word_length=1, stop_words=None, only_first_word=False):
    cleaned_text = " "
    if text:
        #lowercase
        text = str(text).lower()
        #remove punctuation
        new_text = ""
        for c in text:
            if c in EXCLUDE:
                new_text = new_text + " "
            else:
                new_text = new_text + c
        #remove double spaces
        new_text = ' '.join(new_text.split())
        #bag of words
        bow = new_text.split(" ")
        #stemming
        if (stemming):
            tmp = []
            for word in bow:
                #stem.PorterStemmer()               light
                #stem.snowball.EnglishStemmer()     medium          SUGGESTED!
                #stem.LancasterStemmer()            aggressive
                s = stem.snowball.EnglishStemmer()   
                tmp.append(s.stem(word))
            bow = tmp
        #remove stop words and short words
        for word in bow:
            if len(word)>=min_word_length and word not in stop_words:
                cleaned_text = cleaned_text + word + " "
                if (only_first_word):
                    break
        #debug
        #print(cleaned_text)
        return cleaned_text.strip()    

