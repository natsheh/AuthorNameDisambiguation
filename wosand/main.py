import time

import numpy as np
import scipy.cluster.hierarchy as hac

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from core import get_library
from core import get_cosine_distance_matrix
from core import get_string_distance_matrix
from core import get_graph

from clustering_evaluation import get_clusters_shannon_entropy
from clustering_evaluation import evaluate
from clustering_evaluation import elbow

from sklearn import metrics

import math

def main():
    # FOCUS NAME
    focus_name = "Liu W%"  #Rodriguez Bassand Abe Boussaid
    
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
    # Graphs [True, False]e]
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
    
    #get optimal matrix
    mat = get_string_distance_matrix(library,catalog,'author_id','exact_match')

    #get mask
    mask_matrix = get_string_distance_matrix(library,catalog,'author_name','syllab_jaccard')
    #plt.figure(k)
    #plt.imshow(mask_matrix, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
    #plt.colorbar()

    #get distance matrices aggregated by sum (counter increases each time) 
    counter = 0
    for (k,v) in all.items():
        if v:
            mat_temp = get_cosine_distance_matrix(library,catalog,k,'cosine',mask_matrix)
            #if k == 'coauthors':
            #    np.savetxt("coauthors_distance_matrix.csv", mat_temp, fmt="%.2e", delimiter=",")
            #plt.figure(k)
            #plt.imshow(mat_temp, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
            final_matrix = final_matrix + mat_temp
            counter = counter + 1
    
    #get graph matrices aggregated by max (counter increases only of 1) 
    graph_mat = None
    for (k,v) in graphs.items():
        if v:
            if graph_mat is None:
                graph_mat = np.zeros((len(library), len(library)))
            mat_temp = get_graph(library, catalog, focus_name, k)
            #if k == 'coauthorship':
            #    np.savetxt("coauthors_distance_matrix2.csv", mat_temp, fmt="%.2e", delimiter=",")
            #plt.figure("Graphs matrix")
            #plt.imshow(mat_temp, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
            graph_mat = np.maximum(mat_temp, graph_mat)
    if graph_mat is not None:
        counter = counter + 1
        #get final matrix, apply also mask to graph matrix
        final_matrix = final_matrix + (graph_mat * mask_matrix)
                
    #normalize final matrix
    final_matrix = final_matrix / counter #/ counter #* mask_matrix 
    
    #save to file
    #np.savetxt("distance_matrix.csv", final_matrix, fmt="%.2e", delimiter=",")
    
    #plot distance matrices
    plt.figure("Summary")
    plt.subplot(121)
    plt.title("Optimal distance matrix")
    plt.imshow(mat, cmap=cm.coolwarm, interpolation='none')
    plt.colorbar()
    plt.subplot(122)
    plt.title("Obtained distance matrix")
    plt.imshow(final_matrix, cmap=cm.coolwarm, interpolation='none')
    plt.colorbar()
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(15,45,800, 500)
    
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
    linkage_matrix = hac.linkage(final_matrix, method='single', metric='euclidean') #single, average
    
    #loop through some thresholds and find the optimal through entropy
    year_entropy_history = []
    coauthor_entropy_history = []
    country_entropy_history = []
    subject_entropy_history = []
    journal_entropy_history = []
    thresholds = [float(x/100) for x in range(0,200,10)]
    for threshold in thresholds:
        print("\nClustering with threshold {0}".format(threshold))
        #get the same clusters as the dendrogram 
        clusters_list = hac.fcluster(linkage_matrix, threshold, criterion='distance')
        
        #mapping structure and evaluation structure
        clusters = {}
        for k,v in enumerate(clusters_list):
            #print(l[k].author_name + " - " + str(library[k].author_id) + " - Cluster: " + str(clusters_list[k]))
            if v not in clusters:
                clusters[v] = []
            #like assigning Id to clusters... library[k].unique_identifier would be better
            clusters[v].append(library[k].unique_identifier)
        
        #num of clusters for current threshold
        num_clusters = len(clusters)

        #evaluation
        precision, recall, f_measure = evaluate(clusters.values(), ground_truth.values())
        print("Precision: {0}".format(precision))
        print("Recall: {0}".format(recall))
        print("Pairwise F1 score: {0}".format(f_measure))
        print("Complexty (number of clusters): ", num_clusters)
        
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
        
    #normalization for plotting
    coauthor_entropy_history_norm = [x/max(coauthor_entropy_history) if max(coauthor_entropy_history)!= 0 else 0 for x in coauthor_entropy_history]
    subject_entropy_history_norm = [x/max(subject_entropy_history) if max(subject_entropy_history)!= 0 else 0 for x in subject_entropy_history]
    country_entropy_history_norm = [x/max(country_entropy_history) if max(country_entropy_history)!= 0 else 0 for x in country_entropy_history]
    year_entropy_history_norm = [x/max(year_entropy_history) if max(year_entropy_history)!= 0 else 0 for x in year_entropy_history]
    journal_entropy_history_norm = [x/max(journal_entropy_history) if max(journal_entropy_history)!= 0 else 0 for x in journal_entropy_history]

    #log
    #print("\nthresholds = {0}".format(thresholds))
    #print("coauthor_entropy_history_norm = {0}".format(coauthor_entropy_history_norm))
    #print("country_entropy_history_norm = {0}".format(country_entropy_history_norm))
    #print("subject_entropy_history_norm = {0}".format(subject_entropy_history_norm))
    #print("year_entropy_history_norm = {0}".format(year_entropy_history_norm))
    #print("journal_entropy_history_norm = {0}".format(journal_entropy_history_norm))

    #find elbows
    coauthor_elbow_index = elbow(coauthor_entropy_history_norm) if max(coauthor_entropy_history)!= 0 else 0
    subject_elbow_index = elbow(subject_entropy_history_norm) if max(subject_entropy_history)!= 0 else 0
    country_elbow_index = elbow(country_entropy_history_norm) if max(country_entropy_history)!= 0 else 0
    year_elbow_index = elbow(year_entropy_history_norm) if max(year_entropy_history)!= 0 else 0
    journal_elbow_index = elbow(journal_entropy_history_norm) if max(journal_entropy_history)!= 0 else 0

    #best cut: average the non zero elbows' x-coordinates
    #Not using coauthors
    #TO FIX
    index_best_cut = 0
    count = 0
    for e in [subject_elbow_index, country_elbow_index, journal_elbow_index]:  #coauthor_elbow_index, year_elbow_index
        if e > 0.5: #avoids to include very low entropies
            index_best_cut = index_best_cut + e
            count = count + 1
    index_best_cut = float(index_best_cut/count) if count != 0 else 0
    #print(index_best_cut)
    best_cut = thresholds[math.ceil(index_best_cut)]
    print("\nBest cut according to entropies: {0}".format(best_cut))

    #plot the overall results
    plt.figure("Optimal cut representation")
    plt.ylabel('Hierarchical clustering threshold')
    plt.ylabel('Entropy')
    plt.plot(thresholds, year_entropy_history_norm, 'r', label="Year")
    plt.plot(thresholds, subject_entropy_history_norm, 'b', label="Subjects")
    plt.plot(thresholds, coauthor_entropy_history_norm, 'g', label="Coauthors")
    plt.plot(thresholds, country_entropy_history_norm, 'm', label="Countries")
    plt.plot(thresholds, journal_entropy_history_norm, 'y', label="Journals")
    plt.plot(thresholds[year_elbow_index], year_entropy_history_norm[year_elbow_index], 'r', marker='o', markersize=10)
    plt.plot(thresholds[subject_elbow_index], subject_entropy_history_norm[subject_elbow_index], 'b', marker='o', markersize=10)
    plt.plot(thresholds[coauthor_elbow_index], coauthor_entropy_history_norm[coauthor_elbow_index], 'g', marker='o', markersize=10)
    plt.plot(thresholds[country_elbow_index], country_entropy_history_norm[country_elbow_index], 'm', marker='o', markersize=10)
    plt.plot(thresholds[journal_elbow_index], journal_entropy_history_norm[journal_elbow_index], 'y', marker='o', markersize=10)
    max_x = max(thresholds)
    max_y = max(max(
        year_entropy_history_norm,
        subject_entropy_history_norm,
        coauthor_entropy_history_norm,
        country_entropy_history_norm,
        journal_entropy_history_norm
    ))
    plt.vlines(best_cut, 0, max_y)
    plt.axis([0, max_x, 0, max_y])  
    plt.legend()
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(15,600,800, 400)

    #plot clustering according to the best cut
    fig, ax = plt.subplots() 
    fig.canvas.set_window_title('Hierarchical clustering')
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.10, top=0.95)
    ax.set_title("Dendrogram")
    hac.dendrogram(linkage_matrix, color_threshold=best_cut, orientation='top', leaf_font_size=10, 
        leaf_label_func=lambda id: library[id].author_name + " - " + str(library[id].author_id))
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(850,45,800, 500)

    
if __name__ == "__main__":
    main()