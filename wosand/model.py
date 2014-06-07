import time

from connection import *

import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform as squareform

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from core import get_library
from core import get_cosine_distance_matrix
from core import get_string_distance_matrix
from core import get_graph

from clustering_evaluation import get_clusters_shannon_entropy
from clustering_evaluation import evaluate
from clustering_evaluation import elbow
from clustering_evaluation import concensus
from clustering_evaluation import within_clust_sim
from clustering_evaluation import between_clust_sim
from sklearn import metrics
from sklearn import linear_model
from sklearn.externals import joblib

import math

def findMCF(name):
    # FOCUS NAME
    focus_name = name
    
    # Cosine distance [True, False]
    all = {
        'title'         :   True, 
        'coauthors'     :   False,
        'institutions'  :   True,
        'journals'      :   False,
        'year'          :   False,  
        'subjects'      :   False,
        'keywords'      :   False, 
        'ref_authors'   :   False,
        'ref_journals'  :   True
    }
    # Graphs [True, False]
    graphs = {
        "coauthorship"  :   True,
        "references"    :   True,
        "subjects"      :   False,
        "keywords"      :   False
    }
    
    #get objects from data 
    library,catalog = get_library(focus_name)
    
    #initialize matrices
    final_matrix = np.zeros((len(library),len(library)))
    mask_matrix = np.identity(len(library))
    graph_mat = np.identity(len(library))
    
    #get optimal matrix DISTANCES
    mat = 1-get_string_distance_matrix(library,catalog,'author_id','exact_match')

    #get mask
    mask_matrix = get_string_distance_matrix(library,catalog,'author_name','syllab_jaccard')
    #plt.figure(k)
    #plt.imshow(mask_matrix, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
    #plt.colorbar()

    #get distance matrices aggregated by sum (counter increases each time) 
    counter = 0
    mat_temp = []
    mat_list = []
    for (k,v) in all.items():
        if v:
            mat_temp = get_cosine_distance_matrix(library,catalog,k,'cosine',mask_matrix)
            """plt.subplot(140+counter+1)
            plt.imshow(mat_temp, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
            plt.title(k)
            plt.colorbar()"""
            mat_list.append(mat_temp)
            counter = counter + 1
    
    #get graph matrices aggregated by max (counter increases only of 1) 
    graph_mat = None
    for (k,v) in graphs.items():
        if v:
            if graph_mat is None:
                graph_mat = np.zeros((len(library), len(library)))
            mat_temp = get_graph(library, catalog, focus_name, k)
            graph_mat = np.maximum(mat_temp, graph_mat)
    if graph_mat is not None:
        counter = counter + 1
        #get final matrix, apply also mask to graph matrix
        mat_list.append(graph_mat * mask_matrix)
        """plt.figure()
        plt.imshow(graph_mat, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
        plt.title("Graphs")
        plt.colorbar()"""

    #IMPORTANT BEFORE CLUSTERING
    #normalize final matrix and convert to DISTANCE matrix
    final_matrix = 1 - np.median(mat_list, axis=0)
    #correct negative values due to floating point precision to zero
    final_matrix[final_matrix<0] = 0
    #statistics
    final_matrix_mean = np.mean(final_matrix[(final_matrix>0)&(final_matrix<1)])
    print("Mean", final_matrix_mean) 
    #print("Median", np.median(final_matrix[final_matrix<2]))
    #print("Variance", np.var(final_matrix[final_matrix<2]))
    """final_matrix[final_matrix>0.7] = 1
    final_matrix[final_matrix<=0.7] = 0 """
    
    """plt.figure()
    plt.subplot(121)
    plt.title("Optimal distance matrix")
    plt.imshow(mat, cmap=cm.coolwarm, interpolation='none')
    plt.colorbar()
    plt.subplot(122)
    plt.title("Obtained distance matrix")
    plt.imshow(final_matrix, cmap=cm.coolwarm, interpolation='none')
    plt.colorbar()"""
    
    #save to file
    np.savetxt("distance_matrix_{0}.csv".format(focus_name.split()[0]), final_matrix, fmt="%.2e", delimiter=",")
    
    #plot distance matrices
    
    #Compute optimal clustering
    ground_truth = {}
    for p1 in library:
        flag = 0
        if p1.author_id not in ground_truth:
            ground_truth[p1.author_id] = []
        ground_truth[p1.author_id].append(p1.unique_identifier)
    #print("Ground truth: ", str(list(ground_truth.values())))

    #Hierarchical clustering on final_mat
    #single linkage = minimum spanning tree
    #complete linkage, average distance, etc...
    #http://pages.cs.wisc.edu/~jerryzhu/cs769/clustering.pdf
    vector_distances = squareform(final_matrix, force='tovector', checks=False)
    print("Mat shape: {0} Vector shape: {1}".format(final_matrix.shape, vector_distances.shape))
    
    #return if cluster is individual
    if len(vector_distances) == 0:
        return 1, 0, 1
    linkage_matrix = hac.linkage(vector_distances, method='average', metric='euclidean') #single, average
    #print("Linkage matrix:\n", linkage_matrix)
    relevant_thresholds =  np.concatenate(([0],np.array(linkage_matrix[:,2])), axis=0)
    year_entropy_history = []
    coauthor_entropy_history = []
    country_entropy_history = []
    subject_entropy_history = []
    journal_entropy_history = []
    concensus_history = []
    within_history = []
    between_history = []
    variance_history = []
    
    truth_fmeasure = 0
    truth_cut = 0
    for x in relevant_thresholds:
        clusters_list = hac.fcluster(linkage_matrix, x, criterion='distance') #'maxclust'
        
        #mapping structure and evaluation structure
        clusters = {}
        for k,v in enumerate(clusters_list):     
            #print(library[k].author_name + " - " + str(library[k].author_id) + " - Cluster: " + str(clusters_list[k]))
            #print(k, v)
            if v not in clusters:
                clusters[v] = []
            #like assigning Id to clusters... library[k].unique_identifier would be better
            clusters[v].append(library[k].unique_identifier)
        
        #num of clusters for current threshold
        num_clusters = len(clusters)
        
        #evaluation
        concensus_history.append(concensus(1 -final_matrix, catalog, clusters))
        within_history.append(within_clust_sim(1-final_matrix, catalog, clusters))
        between_history.append(between_clust_sim(1-final_matrix, catalog, clusters))
        #print("Within clusters: {0}".format(within_clust_sim(final_matrix, catalog, clusters)))
        #print("GT:", ground_truth)
        precision, recall, f_measure = evaluate(clusters.values(), ground_truth.values())
        #print("Precision: {0}".format(precision))
        #print("Recall: {0}".format(recall))
        if f_measure > truth_fmeasure:
            truth_fmeasure = f_measure
            truth_cut = x
        
        #variance
       
    
    #best cut
    best_cut_index = concensus_history.index(max(concensus_history))
    best_cut = relevant_thresholds[best_cut_index]
    print("`\nBEst cut found index: {0} Value: {1}".format(best_cut_index,best_cut))
    print("Ground truth best cut value {0} Fmeasure: {1}".format(truth_cut, truth_fmeasure))
    
    if np.abs(final_matrix_mean) is np.inf:
        final_matrix_mean = 0
    #TEST!!!!!!!
    return final_matrix_mean, truth_cut, truth_fmeasure
    
    #plot the overall results
    plt.figure("Optimal cut representation")
  
    plt.plot(relevant_thresholds, variance_diff, 'r', label="Variance")

    #plot clustering according to the best cut
    fig, ax = plt.subplots() 
    fig.canvas.set_window_title('Hierarchical clustering')
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.95)
    ax.set_title("Dendrogram")
    hac.dendrogram(linkage_matrix, color_threshold=best_cut, orientation='top', leaf_font_size=10, 
        leaf_label_func=lambda id: library[id].author_name + " - " + str(library[id].author_id))
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(850,45,800, 500)


def normalize(values):
    norm_values = [(x-min(values))/(max(values)-min(values)) if (max(values)-min(values))!= 0 else 0 for x in values]
    return norm_values
    
if __name__ == "__main__":
    focus_names = ["Rodriguez F%", "Lefebvre A%", "Liu W%", "Meyer R%", "Morel M%", "Abe %", "Ades %"] #\
        #+ ['Abe '+x+'%' for x in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']] \
        #+ ['Ades '+x+'%' for x in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']] \
        
    #focus_names = []
    conn = Connection()
    sql_results = None #conn.execute("SELECT DISTINCT(author) FROM articles_authors_disambiguated WHERE authorId>0 AND authorId<3000")
    if sql_results:
        for el in sql_results:
            focus_names.append(el[0] + "%")
    best_cuts = []
    means = []
    f_measures = []
    f_names = []
    for n in focus_names:
        m, c, f = findMCF(n)
        if m >= 0 and m <1:
            f_names.append(n)
            best_cuts.append(c)
            means.append(m)
            f_measures.append(f)
    x,y,z,fm = zip(*sorted(zip(means, best_cuts, f_names, f_measures)))
    
    print(x, '\n', y,'\n', z, '\n', fm)
    plt.figure()
    plt.plot(list(x), list(y), marker='o', linestyle='None')
    for i in range(len(z)):
        plt.annotate(z[i]+"\n{0:.2f}".format(fm[i]), xy=(x[i], y[i]), xytext=(-10,10), textcoords='offset points', ha='center', va='bottom') #+"\n{0:.2f}".format(fm[i])
    
    #REgression   
    regr = np.polyfit(x, y, 3)
    polModel = np.poly1d(regr)
    plt.plot([x/1000 for x in range(500,1000)], [polModel(x/1000) for x in range(500,1000)], color='k', linewidth=2, label="Regression")
    
    """regr = linear_model.Ridge()
    regr_X = x
    regr_Y = y
    regr.fit(np.vander(regr_X, 10000), regr_Y)
    #regr_results = [x[0] for x in regr.predict(np.vander(regr_X, 9)).tolist()]
    plt.plot(regr_X, regr.predict(np.vander(regr_X, 10000)), color='k', linewidth=2, label="Regression")
    joblib.dump(regr, 'model.pkl', compress=9)"""

    #regr = joblib.load('model.pkl')