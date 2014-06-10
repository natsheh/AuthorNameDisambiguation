import time

from connection import *

import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform as squareform
from scipy.stats import mode

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

DEBUG = False
PRODUCTION = True
plots = True

import warnings
warnings.filterwarnings('ignore')


def main(name):
    # FOCUS NAME
    focus_name = name
    print("\nFOCUS NAME: {0}".format(focus_name))
    
    # Cosine distance [True, False]
    all = {
        'title'         :   False, 
        'coauthors'     :   False,
        'institutions'  :   True,
        'journals'      :   False,
        'year'          :   False,  
        'subjects'      :   False,
        'keywords'      :   False, 
        'ref_authors'   :   False,
        'ref_journals'  :   False
    }
    # Graphs [True, False]
    graphs = {
        "coauthorship"  :   True,
        "references"    :   True,
        "subjects"      :   False,
        "keywords"      :   False
    }
    
    #get objects from data 
    library,catalog = get_library(focus_name, DEBUG)
    
    #initialize matrices
    final_matrix = np.zeros((len(library),len(library)))
    mask_matrix = np.identity(len(library))
    graph_mat = np.identity(len(library))
    
    #get optimal matrix DISTANCES
    mat = 1 - get_string_distance_matrix(library,catalog,'author_id','exact_match')

    #get mask
    mask_matrix = get_string_distance_matrix(library,catalog,'author_name','syllab_jaccard')
        
    #plots
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Mask with initials", fontdict={'fontsize': 18})
    im = ax.imshow(mask_matrix, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.colorbar(im)
    return
    """
    """
    mat_title = 1 - get_cosine_distance_matrix(library,catalog,'title','cosine',mask_matrix)
    mat_coauthors = 1- get_cosine_distance_matrix(library,catalog,'coauthors','cosine',mask_matrix)
    mat_institutions = 1-get_cosine_distance_matrix(library,catalog,'institutions','cosine',mask_matrix)
    mat_journals = 1-get_cosine_distance_matrix(library,catalog,'journals','cosine',mask_matrix)
    mat_year = 1-get_cosine_distance_matrix(library,catalog,'year','cosine',mask_matrix)
    mat_subjects = 1-get_cosine_distance_matrix(library,catalog,'subjects','cosine',mask_matrix)
    mat_keywords = 1-get_cosine_distance_matrix(library,catalog,'keywords','cosine',mask_matrix)
    mat_ref_authors = 1-get_cosine_distance_matrix(library,catalog,'ref_authors','cosine',mask_matrix)
    mat_ref_journals = 1-get_cosine_distance_matrix(library,catalog,'ref_journals','cosine',mask_matrix)
    """
    """
    fig = plt.figure()
    #("Cosine similarity matrices", fontdict={'fontsize': 18})
    ax = fig.add_subplot(251)
    plt.title("Optimal")
    ax.imshow(mat, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(252)
    ax.set_title("Title")
    ax.imshow(mat_title, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(253)
    ax.set_title("Coauthors")
    ax.imshow(mat_coauthors, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(254)
    ax.set_title("Institutions")
    ax.imshow(mat_institutions, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(255)
    ax.set_title("Journals")
    ax.imshow(mat_journals, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(256)
    ax.set_title("Year") 
    ax.imshow(mat_year, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(257)
    ax.set_title("Subjects")
    ax.imshow(mat_subjects, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(258)
    ax.set_title("Keywords")
    ax.imshow(mat_keywords, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(259)
    ax.set_title("ReferenceAuthor")
    ax.imshow(mat_ref_authors, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(250)
    ax.set_title("ReferenceJournal")
    ax.imshow(mat_ref_journals, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return
    """
    
    mat_coauthorship = 1 - (get_graph(library, catalog, focus_name, 'coauthorship') * mask_matrix)
    mat_references = 1 - (get_graph(library, catalog, focus_name, 'references') * mask_matrix)
    """
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title("Coauthorship")
    ax.imshow(mat_coauthorship, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(122)
    ax.set_title("References")
    ax.imshow(mat_references, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return
    """
    
    #get distance matrices aggregated by sum (counter increases each time) 
    counter = 0
    mat_temp = []
    mat_list = []
    for (k,v) in all.items():
        if v:
            mat_temp = get_cosine_distance_matrix(library,catalog,k,'cosine',mask_matrix)
            """plt.subplot(250+counter+2)
            plt.imshow(mat_temp, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
            plt.title(k)
            plt.colorbar()"""
            mat_list.append(mat_temp)
            counter = counter + 1
    
    #get graph matrices aggregated by max (counter increases only of 1) 
    graph_mat = None
    for (k,v) in graphs.items():
        #mat_temp = get_graph(library, catalog, focus_name, k)
        #mat_list.append(mat_temp)
        #counter = counter + 1
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
    #print(final_matrix)
    if final_matrix.size == 0 or final_matrix.size == 1:
        print("Only one paper found.")
        return 
    #else:
    #    #print("Number of papers: {0}".format(len(library)))
    #    #print("Number of authors: {0}".format(len(set([x.author_id for x in library]))))
        
    #statistics
    if DEBUG:
        print("Mean", np.mean(final_matrix[final_matrix<1]))
        print("Median", np.median(final_matrix[final_matrix<1]))
        print("Variance", np.var(final_matrix[final_matrix<1]))
    
    #save to file
    np.savetxt("distance_matrix_{0}.csv".format(focus_name.split()[0]), final_matrix, fmt="%.2e", delimiter=",")

    #plot distance matrices
    if plots:
        plt.figure()
        plt.subplot(121)
        plt.title("Optimal distance matrix")
        plt.imshow(mat, cmap=cm.GnBu, interpolation='none')
        plt.colorbar()
        plt.subplot(122)
        plt.title("Obtained distance matrix")
        plt.imshow(final_matrix, cmap=cm.GnBu, interpolation='none')
        plt.colorbar()
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(15,45,800, 500)
    
    #other plots for the report 
    """fig = plt.figure()
    ax = fig.add_subplot(151)
    ax.set_title("Optimal", fontdict={'fontsize': 15})
    ax.imshow(mat, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(152)
    ax.set_title("Institutions", fontdict={'fontsize': 15})
    ax.imshow(mat_institutions, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(153)
    ax.set_title("Coauthorship", fontdict={'fontsize': 15})
    ax.imshow(mat_coauthorship, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(154)
    ax.set_title("References", fontdict={'fontsize': 15})
    ax.imshow(mat_references, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(155)
    ax.set_title("Aggregated", fontdict={'fontsize': 15})
    im = ax.imshow(final_matrix, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return
    """
    
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
    #print("Mat shape: {0} Vector shape: {1}".format(final_matrix.shape, vector_distances.shape))
    
    linkage_matrix = hac.linkage(vector_distances, method='average', metric='euclidean') #single, average
    #print("Linkage matrix:\n", linkage_matrix)
    relevant_thresholds =  np.concatenate(([0],np.array(linkage_matrix[:,2]), [1]), axis=0)
    
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
        x+=0.00001
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
        precision, recall, f_measure = evaluate(clusters.values(), ground_truth.values())
        if f_measure >= truth_fmeasure:
            truth_fmeasure = f_measure
            truth_cut = x
            truth_precision = precision
            truth_recall = recall
        
        # entropies with respect to features, for each cluster:
        # - first split the feature of interest (e.g. coauthors) and put them in the "entropy_feature" list, keep repetitions 
        # - then count the ocurrences of each class (e.g. each coauthor)
        # - finally calculate the shannon entropy   
        # - sum up all the entropies 
        entropy_coauthors = {}
        entropy_subjects = {}
        entropy_countries = {}
        entropy_years = {}
        entropy_journals = {}
        for clust, author_uids in clusters.items():
        
            #entropy coauthors for the current cluster
            coauthors_in_clusters = []
            for uid in author_uids:
                for coauthor in library[catalog[uid]].coauthors.split(" "):
                    if coauthor.strip(): coauthors_in_clusters.append(coauthor)
            entropy_coauthors[clust] = get_clusters_shannon_entropy(coauthors_in_clusters)
            
            #entropy test on subjects
            subjects_in_clusters = []
            for uid in author_uids:
                for subject in library[catalog[uid]].subjects.split(" "):
                    if subject.strip(): subjects_in_clusters.append(subject)
            entropy_subjects[clust] = get_clusters_shannon_entropy(subjects_in_clusters)
            
            #entropy test on countries
            countries_in_clusters = []
            for uid in author_uids:
                for country in library[catalog[uid]].countries.split(" "):
                    if country.strip(): countries_in_clusters.append(country)
            entropy_countries[clust] = get_clusters_shannon_entropy(countries_in_clusters)
            
            #entropy on years
            years_in_clusters = []
            for uid in author_uids:
                for year in library[catalog[uid]].year.split(" "):
                    if year.strip(): years_in_clusters.append(year)
            entropy_years[clust] = get_clusters_shannon_entropy(years_in_clusters)
            
            #entropy on journal
            journals_in_clusters = []
            for uid in author_uids:
                for journal in library[catalog[uid]].journals.split(" "):
                    if journal.strip(): journals_in_clusters.append(journal)
            entropy_journals[clust] = get_clusters_shannon_entropy(journals_in_clusters)

        #statistics for the current cluster
        #sum up the entropies or average them
        year_entropy_history.append(sum(entropy_years.values())/num_clusters)
        coauthor_entropy_history.append(sum(entropy_coauthors.values())/num_clusters)
        country_entropy_history.append(sum(entropy_countries.values())/num_clusters)
        subject_entropy_history.append(sum(entropy_subjects.values())/num_clusters)
        journal_entropy_history.append(sum(entropy_journals.values())/num_clusters)
    
    #best cut
    my_cut_index = concensus_history.index(max(concensus_history))
    my_cut = relevant_thresholds[my_cut_index]
    my_clusters_list = hac.fcluster(linkage_matrix, my_cut, criterion='distance')
    my_clusters = {}
    for k,v in enumerate(my_clusters_list):     
        if v not in my_clusters:
            my_clusters[v] = []
        my_clusters[v].append(library[k].unique_identifier)
    
    #print fmeasure if not in PRODUCTION mode
    if not PRODUCTION:
        my_precision, my_recall, my_f_measure = evaluate(my_clusters.values(), ground_truth.values())
        print("My cut value: {0} F-Measure: {1} Precision: {2} Recall: {3}".format(my_cut, my_f_measure, my_precision, my_recall))
        print("Ground truth cut value {0} F-Measure: {1} Precision: {2} Recall: {3}:".format(truth_cut, truth_fmeasure, truth_precision,truth_recall))
        #print("{0:.3f} / {1:.3f} / {2:.3f}".format(my_precision,my_recall,my_f_measure))
    #else print titles of the articles and clusters
    else:
        print("\nHIERARCHICAL CLUSTERING RESULTS")
        for my_clust, my_author_uids in my_clusters.items():
            print("Author {0} assigned to the following papers:".format(my_clust))
            for uid in my_author_uids:
                print(" - {0}".format(library[catalog[uid]].printout))
            print("\n")

    if plots:
        #normalization for plotting
        coauthor_entropy_history_norm = normalize(coauthor_entropy_history)
        subject_entropy_history_norm = normalize(subject_entropy_history)
        country_entropy_history_norm = normalize(country_entropy_history)
        year_entropy_history_norm = normalize(year_entropy_history)
        journal_entropy_history_norm = normalize(journal_entropy_history)
    
        
        #plot the overall results
        max_y = max(max(
            year_entropy_history_norm,
            subject_entropy_history_norm,
            coauthor_entropy_history_norm,
            country_entropy_history_norm,
            journal_entropy_history_norm
            ))
        min_y = min(min(
            year_entropy_history_norm,
            subject_entropy_history_norm,
            coauthor_entropy_history_norm,
            country_entropy_history_norm,
            journal_entropy_history_norm
            ))
        plt.figure("Optimal cut representation")
        plt.title('Hierarchical agglomerative clustering thresholds and entropies', fontdict={'fontsize': 18})
        plt.ylabel('Normalized information entropy', fontdict={'fontsize': 14})
        plt.xlabel('HAC cutting threshold', fontdict={'fontsize': 14})
        plt.axis([.5, 1 , 0, max_y])
        plt.plot(relevant_thresholds, year_entropy_history_norm, 'r', label="Year")
        plt.plot(relevant_thresholds, subject_entropy_history_norm, 'b', label="Subjects")
        plt.plot(relevant_thresholds, coauthor_entropy_history_norm, 'g', label="Coauthors")
        plt.plot(relevant_thresholds, country_entropy_history_norm, 'm', label="Countries")
        plt.plot(relevant_thresholds, journal_entropy_history_norm, 'y', label="Journals")
        plt.legend(loc=2)
        plt.vlines(truth_cut, min_y, max_y)
        plt.legend(loc=2)
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(15,600,800, 400)
        plt.figure()
        plt.title('Hierarchical agglomerative clustering thresholds and similarities', fontdict={'fontsize': 18})
        plt.plot(relevant_thresholds, within_history, color='r', linewidth=1, label="Intra clusters")
        plt.plot(relevant_thresholds, between_history, color='g', linewidth=1, label="Inter clusters")
        plt.plot(relevant_thresholds, concensus_history, color='b', linewidth=1, label="Consensus")
        plt.plot(my_cut, concensus_history[list(relevant_thresholds).index(my_cut)], 'k', marker='o', markersize=10)
        min_y = min(min(concensus_history,within_history,between_history))
        max_y = max(max(concensus_history,within_history,between_history))
        plt.legend(loc=2)
        plt.ylabel('Similarity', fontdict={'fontsize': 14})
        plt.xlabel('HAC cutting threshold', fontdict={'fontsize': 14})
        plt.vlines(truth_cut, min_y, max_y)
        plt.axis([0.5, 1 , min_y, max_y])
        
        #plot clustering according to the best cut
        fig, ax = plt.subplots() 
        fig.canvas.set_window_title('Hierarchical clustering')
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
        ax.set_title("Hierarchical Agglomerative Clustering for {0}".format(focus_name.split()[0]), fontdict={'fontsize': 18})
        hac.dendrogram(linkage_matrix, color_threshold=my_cut, orientation='top', leaf_font_size=12, 
            leaf_label_func=lambda id: library[id].author_name + " - " + str(library[id].author_id))
        plt.ylabel("Threshold", fontdict={'fontsize': 14})
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(850,45,800, 500)


def normalize(values):
    norm_values = [(x-min(values))/(max(values)-min(values)) if (max(values)-min(values))!= 0 else 0 for x in values]
    return norm_values
    
if __name__ == "__main__":
    
    focus_names = ["Abe %"]
    
    """focus_names = []
        "Lefebvre A%", "Ades %", "Zighed %", "Liu W%", "Bassand %", "Boussaid %", "Meyer R%", "Morel M%", "Abe %", \
        "Karakiewicz %", "Nikolic %", "Allemand %", "Arlot %", "Barba %", "Bassetti %", "Blaise %", "Casteilla %", "Eklund %", "Chiron %", "Gaillot %",\
        "Godard %", "Gosse %", "Jouet %", "Pita %", "Puget %", "Stokes %", "Rico %", "Rohrmann %", \
        "Muller J%", "Velcin %", "Pujolle %", "Rodriguez F%", "Louvet %", "De Dinechin %", "Daumas %"]"""
    
    """focus_names = []
    conn = Connection()
    #This query is wrong is too detailed, we need only focus names without initials
    sql_results = conn.execute("SELECT DISTINCT(author) FROM articles_authors_disambiguated WHERE authorId>0 AND authorId>=3000 ORDER BY author")
    if sql_results:
        for el in sql_results:
            focus_names.append(el[0] + "%")"""
            
    focus_names.sort()
    for n in focus_names:
        main(n)