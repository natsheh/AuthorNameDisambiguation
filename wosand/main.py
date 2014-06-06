import time

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

import math

def main(name):
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
    mat = 1 - get_string_distance_matrix(library,catalog,'author_id','exact_match')

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
        plt.title("Graphs")"""

    #IMPORTANT BEFORE CLUSTERING
    #normalize final matrix and convert to DISTANCE matrix
    final_matrix = 1 - np.median(mat_list, axis=0)
    print("Mean", np.mean(final_matrix[final_matrix>0])) 
    print("Median", np.median(final_matrix[final_matrix>0]))
    print("Variance", np.var(final_matrix[final_matrix>0]))
    """final_matrix[final_matrix>0.7] = 1
    final_matrix[final_matrix<=0.7] = 0 """
    
    
    #save to file
    np.savetxt("distance_matrix_{0}.csv".format(focus_name.split()[0]), final_matrix, fmt="%.2e", delimiter=",")
    
    #plot distance matrices
    """plt.figure("Summary")
    plt.subplot(121)
    plt.title("Optimal distance matrix")
    plt.imshow(mat, cmap=cm.coolwarm, interpolation='none')
    plt.colorbar()
    plt.subplot(122)
    plt.title("Obtained distance matrix")
    plt.imshow(final_matrix, cmap=cm.coolwarm, interpolation='none')
    plt.colorbar()
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(15,45,800, 500)"""
    
    #final matrix histogram
    """hist, bins = np.histogram(final_matrix, bins=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.1])
    plt.figure()
    plt.step(bins[:len(bins)-1], hist)"""
    
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
    print(final_matrix.shape)
    vector_distances = squareform(final_matrix, force='tovector', checks=False)
    print(vector_distances.shape)
    linkage_matrix = hac.linkage(vector_distances, method='average', metric='euclidean') #single, average
    print(linkage_matrix)
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
    
    """new_plot = [relevant_thresholds[i] - relevant_thresholds[i-1] for i in range(1, len(relevant_thresholds))]
    new_plot.insert(0,0)
    print(new_plot, relevant_thresholds)
    plt.figure()
    plt.plot(relevant_thresholds, range(0,len(relevant_thresholds)), color='r', linewidth=1, label="New")"""
    truth_fmeasure = 0
    truth_cut = 0
    for x in relevant_thresholds:
        clusters_list = hac.fcluster(linkage_matrix, x, criterion='distance') #'maxclust'
        
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
        concensus_history.append(concensus(1 -final_matrix, catalog, clusters))
        within_history.append(within_clust_sim(1-final_matrix, catalog, clusters))
        between_history.append(between_clust_sim(1-final_matrix, catalog, clusters))
        #print("Within clusters: {0}".format(within_clust_sim(final_matrix, catalog, clusters)))
        precision, recall, f_measure = evaluate(clusters.values(), ground_truth.values())
        #print("Precision: {0}".format(precision))
        #print("Recall: {0}".format(recall))
        if f_measure > truth_fmeasure:
            truth_fmeasure = f_measure
            truth_cut = x
        
        #variance
        total_variance = []
        
        for clust, author_uids in clusters.items():
            clust_variance = []
            clust_zeros =  0
            for uid1 in author_uids:
                for uid2 in author_uids:
                    tmp = final_matrix[ [catalog[uid1]], [catalog[uid2]] ]
                    clust_variance.append(tmp)
                    #if tmp > 0.85:
                    #    clust_zeros += 1
                total_variance.append(np.var(clust_variance))
            #total_variance.append(clust_zeros/(len(author_uids)**2))
        #print("Var", total_variance)
        variance_history.append(np.sum(total_variance))#/num_clusters) ###
        #variance_history.append(np.sum(total_variance))
        
        # entropies with respect to features, for each cluster:
        # - first split the feature of interest (e.g. coauthors) and put them in the "entropy_feature" list, keep repetitions 
        # - then count the ocurrences of each class (e.g. each coauthor)
        # - finally calculate the shannon entropy   
        # - sum up all the entropies 
        """entropy_coauthors = {}
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
        year_entropy_history.append(sum(entropy_years.values()))#/num_clusters)
        coauthor_entropy_history.append(sum(entropy_coauthors.values()))#/num_clusters)
        country_entropy_history.append(sum(entropy_countries.values()))#/num_clusters)
        subject_entropy_history.append(sum(entropy_subjects.values()))#/num_clusters)
        journal_entropy_history.append(sum(entropy_journals.values()))#/num_clusters)
    
    #difference of entropy
    year_entropy_diff = [year_entropy_history[i]-year_entropy_history[i-1] for i in range(1,len(year_entropy_history))]
    coauthor_entropy_diff = [coauthor_entropy_history[i]-coauthor_entropy_history[i-1] for i in range(1,len(coauthor_entropy_history))]
    country_entropy_diff = [country_entropy_history[i]-country_entropy_history[i-1] for i in range(1,len(country_entropy_history))]
    subject_entropy_diff = [subject_entropy_history[i]-subject_entropy_history[i-1] for i in range(1,len(subject_entropy_history))]
    journal_entropy_diff = [journal_entropy_history[i]-journal_entropy_history[i-1] for i in range(1,len(journal_entropy_history))]
    year_entropy_diff.insert(0,0)         
    coauthor_entropy_diff.insert(0,0)
    country_entropy_diff.insert(0,0)
    subject_entropy_diff.insert(0,0)              
    journal_entropy_diff.insert(0,0)"""
    
    #difference of variance
    variance_diff = [variance_history[i]-variance_history[i-1] for i in range(1,len(variance_history))]
    variance_diff.insert(0,0)
    
    #normalization for plotting
    """coauthor_entropy_history_norm = normalize(coauthor_entropy_diff)
    subject_entropy_history_norm = normalize(subject_entropy_diff)
    country_entropy_history_norm = normalize(country_entropy_diff)
    year_entropy_history_norm = normalize(year_entropy_diff)
    journal_entropy_history_norm = normalize(journal_entropy_diff)
    #aggregate_median = np.median([coauthor_entropy_history_norm, subject_entropy_history_norm, country_entropy_history_norm, year_entropy_history_norm, journal_entropy_history_norm], axis=0)
    #aggregate_mean = np.mean([coauthor_entropy_history_norm, subject_entropy_history_norm, country_entropy_history_norm, year_entropy_history_norm, journal_entropy_history_norm], axis=0)
    #aggregate_max = np.max([coauthor_entropy_history_norm, subject_entropy_history_norm, country_entropy_history_norm, year_entropy_history_norm, journal_entropy_history_norm], axis=0)
   
    
    #find elbows
    coauthor_elbow_index = elbow(coauthor_entropy_history_norm) if max(coauthor_entropy_history)!= 0 else 0
    subject_elbow_index = elbow(subject_entropy_history_norm) if max(subject_entropy_history)!= 0 else 0
    country_elbow_index = elbow(country_entropy_history_norm) if max(country_entropy_history)!= 0 else 0
    year_elbow_index = elbow(year_entropy_history_norm) if max(year_entropy_history)!= 0 else 0
    journal_elbow_index = elbow(journal_entropy_history_norm) if max(journal_entropy_history)!= 0 else 0
    #aggregate_median_elbow_index = elbow(aggregate_median) if max(aggregate_median)!= 0 else 0
    #aggregate_mean_elbow_index = elbow(aggregate_mean) if max(aggregate_mean)!= 0 else 0
    #aggregate_max_elbow_index = elbow(aggregate_max) if max(aggregate_max)!= 0 else 0
    """
    
    #best cut
    best_cut_index = concensus_history.index(max(concensus_history))
    best_cut = relevant_thresholds[best_cut_index]
    print("`\nBEst cut found index: {0} Value: {1}".format(best_cut_index,best_cut))
    print("Ground truth best cut value {0} Fmeasure: {1}".format(truth_cut, truth_fmeasure))
    
    #TEST!!!!!!!
    return np.mean(final_matrix[final_matrix>0]), truth_cut
    
    #plot the overall results
    plt.figure("Optimal cut representation")
    """plt.ylabel('Hierarchical clustering threshold')
    plt.ylabel('Entropy')
    plt.plot(relevant_thresholds, year_entropy_history_norm, 'r', label="Year")
    plt.plot(relevant_thresholds, subject_entropy_history_norm, 'b', label="Subjects")
    plt.plot(relevant_thresholds, coauthor_entropy_history_norm, 'g', label="Coauthors")
    plt.plot(relevant_thresholds, country_entropy_history_norm, 'm', label="Countries")
    plt.plot(relevant_thresholds, journal_entropy_history_norm, 'y', label="Journals")
    plt.plot(relevant_thresholds[year_elbow_index], year_entropy_history_norm[year_elbow_index], 'r', marker='o', markersize=10)
    plt.plot(relevant_thresholds[subject_elbow_index], subject_entropy_history_norm[subject_elbow_index], 'b', marker='o', markersize=10)
    plt.plot(relevant_thresholds[coauthor_elbow_index], coauthor_entropy_history_norm[coauthor_elbow_index], 'g', marker='o', markersize=10)
    plt.plot(relevant_thresholds[country_elbow_index], country_entropy_history_norm[country_elbow_index], 'm', marker='o', markersize=10)
    plt.plot(relevant_thresholds[journal_elbow_index], journal_entropy_history_norm[journal_elbow_index], 'y', marker='o', markersize=10)"""
    
    plt.plot(relevant_thresholds, variance_diff, 'r', label="Variance")
    
    #plt.plot(relevant_thresholds, aggregate_median, 'r', label="Median")
    #plt.plot(relevant_thresholds[aggregate_median_elbow_index], aggregate_median[aggregate_median_elbow_index], 'r', marker='o', markersize=10)
    #plt.plot(relevant_thresholds, aggregate_mean, 'g', label="Mean")
    #plt.plot(relevant_thresholds[aggregate_mean_elbow_index], aggregate_mean[aggregate_mean_elbow_index], 'g', marker='o', markersize=10)
    #plt.plot(relevant_thresholds, aggregate_max, 'b', label="Max")
    #plt.plot(relevant_thresholds[aggregate_max_elbow_index], aggregate_max[aggregate_max_elbow_index], 'b', marker='o', markersize=10)
    max_x = max(variance_diff)
    
    max_y = max(variance_diff)
    """year_entropy_history_norm,
    subject_entropy_history_norm,
    coauthor_entropy_history_norm,
    country_entropy_history_norm,
    journal_entropy_history_norm"""

    min_y = min(variance_diff)
    """year_entropy_history_norm,
    subject_entropy_history_norm,
    coauthor_entropy_history_norm,
    country_entropy_history_norm,
    journal_entropy_history_norm"""
        
    #plt.vlines(best_cut, 0, max_y)
    """plt.axis([0, max_x , min_y, max_y])  
    plt.legend(loc=2)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.setGeometry(15,600,800, 400)
    plt.figure()
    plt.plot(relevant_thresholds, concensus_history, color='b', linewidth=1, label="Concensus")
    plt.plot(relevant_thresholds, within_history, color='r', linewidth=1, label="Within")
    plt.plot(relevant_thresholds, between_history, color='g', linewidth=1, label="Between")
    min_x = min(relevant_thresholds)
    max_x = max(relevant_thresholds)
    min_y = min(min(concensus_history,within_history,between_history))
    max_y = max(max(concensus_history,within_history,between_history))
    plt.legend(loc=2)
    plt.vlines(best_cut, min_y, max_y)
    plt.axis([min_x, max_x , min_y, max_y])
    """

    
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
    focus_names = ["Rodriguez F%", "Lefebvre A%", "Ades %", "Abe %", "Zighed %", "Liu W%", "Bassand %", "Boussaid %", "Meyer R%", "Morel M%"]
    best_cuts = []
    means = []
    for x in focus_names:
        m, c = main(x)
        best_cuts.append(c)
        means.append(m)
    x,y,z = zip(*sorted(zip(means, best_cuts,focus_names )))
    
    #print(x, '\n', y,'\n', z)
    plt.figure()
    plt.plot(list(x), list(y), marker='o', linestyle='None')
    for i in range(len(z)):
        plt.annotate(z[i], xy=(x[i], y[i]), xytext=(-10,10), textcoords='offset points', ha='center', va='bottom')