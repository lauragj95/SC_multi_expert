# -*- coding: utf-8 -*-


from sklearn.cluster import DBSCAN
from scipy import spatial
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from einops import rearrange
from sklearn.cluster import DBSCAN
import copy
from sklearn.metrics import silhouette_samples
import math
import torch
from skimage.io import imsave
from utils.distance_utils import calculate_distances_clusters, calculate_distances_diff_cl_clusters,calculate_distances_core_noncore,calculate_distances,calculate_distances_diff_cl
from self_consistency.sc import compute_sc,compute_sc2
from self_consistency.weights import compute_weights


def calculate_nearest_neighbors_non_core(dist,cl):
    """
    Calculates the nearest neighbors for non-core points based on a distance dictionary.

    For each key in the input distance dictionary, selects the two closest neighbors (with the smallest distances)
    and collects their corresponding cluster labels from the provided cluster label dictionary.

    Args:
        dist (dict): A nested dictionary where each key represents a point, and the value is another dictionary
            mapping neighboring points to their distances.
        cl (dict): A dictionary mapping each point to its cluster label.

    Returns:
        dict: A dictionary mapping the selected nearest neighbor points to their cluster labels.
    """
    neighbors = []
    all_neighbors = {}

    for key,value in dist.items():
        val_list = dict(sorted(value.items(), key=lambda x:x[1])[:2])
        for key1 in val_list.keys():
            all_neighbors[key1]=cl[key1]
        neighbors.extend(val_list.keys())    
    return all_neighbors


def get_noisy(labels):
    """
    Counts the number of noisy samples (labeled as -1) in each clustering result.

    Args:
        labels (list of array-like): A list where each element is an array of cluster labels assigned to samples. 
            Noisy samples are expected to have the label -1.

    Returns:
        list: A list containing the count of noisy samples (label -1) for each clustering result in `labels`.
    """
    noisy = []
    for label in labels:
        clusters, count = np.unique(label,return_counts=True)
        dict_aux = dict(zip(clusters,count))
        noisy.append(dict_aux[-1])
    return noisy



def get_core_expert(cls,dists,min=50,max=90):
    """
    Applies DBSCAN clustering to each set of class features and distances, extracting core samples and cluster labels.

    Args:
        cls (list of dict): A list where each element is a dictionary mapping sample identifiers to feature vectors for a class.
        dists (list of array-like): A list where each element contains the pairwise distances for the corresponding class in `cls`.
        min (int,optional): The minimum percentile value to consider when calculating `min_samples`. Default is 50.
        max (int,optional): The maximum percentile value to consider when calculating `min_samples`. Default is 90.
 
    Returns:
        tuple:
            centroids (list of ndarray): List of arrays containing the core samples (cluster centroids) for each class.
            labels (list of ndarray): List of arrays containing the cluster labels assigned to each sample for each class.
            epss (list of float): List of epsilon values (neighborhood radius) used for DBSCAN for each class.
            min_sampless (list of int): List of minimum samples values used for DBSCAN for each class.

    Notes:
        - The epsilon value for DBSCAN is set as the 10th percentile of the distances for each class.
        - If the computed epsilon is zero, it is incremented by one to avoid invalid clustering.
        - The function `get_min_samples` is assumed to compute the minimum number of samples for DBSCAN based on the class and its distances.
    """
    centroids = []
    labels = []
    epss = []
    min_sampless = []
    for i in range(len(cls)):
        eps = np.percentile(dists[i],10)
        if eps == 0:
            eps+=1
        min_samples = get_min_samples(cls[i],dists[i],min,max)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(list(cls[i].values()))
        centroids.append(dbscan.components_)
        labels.append(dbscan.labels_)
        epss.append(eps)
        min_sampless.append(min_samples)

    return centroids,labels,epss,min_sampless



def calculate_nearest_neighbors(dist, dist_others):
    """
    Calculates the most frequent nearest neighbors from a set of distance dictionaries.
    This function takes a primary distance dictionary and a list of additional distance dictionaries,
    merges the neighbor distances for each key, and selects the top 3 nearest neighbors for each key.
    It then aggregates all selected neighbors, extracts their base identifiers (before the underscore),
    and returns the unique neighbor identifiers along with their occurrence counts.
    Args:
        dist (dict): A dictionary where each key maps to another dictionary of neighbor identifiers and their distances.
        dist_others (list of dict): A list of dictionaries, each structured like `dist`, containing additional neighbor distances.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Array of unique neighbor identifiers (as strings).
            - np.ndarray: Array of counts corresponding to each unique neighbor identifier.
    """
    nn = {}
    all_neighbors = []
    for key, value in dist.items():
        
        nn[key] = value
        for dist in dist_others:
            nn[key].update(dist[key])
    
    for key,value in nn.items():
        
        val_list = dict(sorted(value.items(), key=lambda x:x[1])[:3])
        all_neighbors.extend([n.split('_')[0] for n in val_list.keys()])

    return np.unique(all_neighbors,return_counts=True)



def get_metrics(centroids, labels, return_neighbors=True):
    """
    Calculates various clustering metrics for a set of centroids and their corresponding labels.

    This function computes intra- and inter-class statistics, including means and standard deviations of distances, 
    nearest neighbors, silhouette-like coefficients, and other custom metrics for evaluating clustering quality.

    Args:
        centroids (list or np.ndarray): List or array of class centroids.
        labels (list or np.ndarray): List or array of class labels corresponding to the centroids.
        return_neighbors (bool, optional): Whether to compute and return nearest neighbors for each class. Defaults to True.

    Returns:
        tuple: A tuple containing the following elements:
            - scs (list): List of self-consistency (e-distance based) for each class.
            - sc2s (list): List of alternative self-consistency(euclidean distance based) for each class.
            - eucl_distances (list): List of Euclidean distances for each class.
            - min_dists (list): List of minimum inter-class distances for each class.
            - outliers_scs (list): List of outlier silhouette coefficients for each class.
            - e_distss (list): List of extra distance metrics for each class.
            - means (dict): Dictionary of mean intra- and inter-class distances.
            - stds (dict): Dictionary of standard deviations for intra- and inter-class distances.
            - neighbors (list): List of nearest neighbors for each class (if return_neighbors is True).

    Note:
        This function relies on external helper functions:
            - calculate_distances_clusters
            - calculate_distances_diff_cl_clusters
            - calculate_nearest_neighbors
            - compute_sc2
            - compute_sc
        These must be defined elsewhere in the codebase.
    """
    means_intra = []
    means = {}
    stds = {}
    neighbors = [] 
    scs = []
    sc2s = []
    min_dists = []
    outliers_scs = []
    e_distss = []
    eucl_distances = []

    for i, centroid in enumerate(centroids):
        i += 1
        dists = []
        dist_intra, all_intra = calculate_distances_clusters(centroid, i + 1)
        means[f'{i}_{i}'] = sum(all_intra) / (len(centroid) ** 2)
        stds[f'{i}_{i}'] = np.std(all_intra)

        for j, centroid_other in enumerate(centroids):
            j += 1
            if j != i:
                dist_inter, all_inter = calculate_distances_diff_cl_clusters(centroid, centroid_other, j + 1)
                dists.append(dist_inter)
                if f'{i}_{j}' not in means or f'{j}_{i}' not in means:
                    means[f'{i}_{j}'] = np.mean(all_inter)
                    stds[f'{i}_{j}'] = np.std(all_inter)

        if return_neighbors:
            neighbors.append(calculate_nearest_neighbors(dist_intra, dists))

    for i, label in enumerate(labels):
        i += 1
        means_inter = []
        means_intra = []
        means_intra.append(means[f'{i}_{i}'])
        for j in range(len(labels)):
            j += 1
            if j != i:
                means_inter.append(means[f'{i}_{j}']) 
            if f'{j}_{j}' != f'{i}_{i}':
                means_intra.append(means[f'{j}_{j}']) 
        sc2, eucl_distance = compute_sc2(means[f'{i}_{i}'], means_inter, label)
        sc2s.append(sc2)
        eucl_distances.append(eucl_distance)
        sc, min_dist, outliers_sc, e_dists = compute_sc(means_inter, means_intra, label)
        scs.append(sc)
        min_dists.append(min_dist)
        outliers_scs.append(outliers_sc)
        e_distss.append(e_dists)

    return scs, sc2s, eucl_distances, min_dists, outliers_scs, e_distss, means, stds, neighbors




def get_metrics_noDBSCAN(cl):
    """
    Computes various clustering metrics for a given list of clusters without using DBSCAN.
    Parameters:
        cl (list): A list of sample features
    Returns:
        tuple: A tuple containing the following elements:
            - scs (list): List of self-consistency (e-distance based)
            - sc2s (list): List of alternative self-consistency(euclidean distance based)
            - eucl_distances (list): Euclidean distances computed for each class.
            - min_dists (list): Minimum distances between class.
            - e_distss (list): List of distance arrays for each class.
            - means (dict): Dictionary of mean intra- and inter-class distances, keys formatted as '{i}_{j}'.
            - stds (dict): Dictionary of standard deviations of intra- and inter-class distances, keys formatted as '{i}_{j}'.
    Notes:
        - This function assumes the existence of helper functions: `calculate_distances`, `calculate_distances_diff_cl`, `compute_sc2`, and `compute_sc`.
        - The input classes should be structured such that distance calculations between and within class are meaningful.
    """
    scs = []
    sc2s = []
    means = {}
    stds = {}
    eucl_distances = []
    min_dists = []
    e_distss = []

    for i in range(len(cl)):
        dist_intra = calculate_distances(cl[i])
        means[f'{i+1}_{i+1}'] = sum(dist_intra)/(len(cl)**2)
        stds[f'{i+1}_{i+1}'] = np.std(dist_intra)
        for j in range(len(cl)):
            if j!=i:
                dist_inter = calculate_distances_diff_cl(cl[i],cl[j])
                if f'{i+1}_{j+1}' or f'{j+1}_{i+1}' not in means.keys():
                    means[f'{i+1}_{j+1}'] = np.mean(dist_inter)
                    stds[f'{i+1}_{j+1}'] = np.std(dist_inter)

    for i in range(len(cl)):
        
        means_inter = []
        means_intra = []
        means_intra.append(means[f'{i+1}_{i+1}'])
        for j in range(len(cl)):
            if j!=i:
                means_inter.append(means[f'{i+1}_{j+1}']) 
            if f'{j+1}_{j+1}' != f'{i+1}_{i+1}':
                means_intra.append(means[f'{j+1}_{j+1}']) 
        sc2,eucl_distance = compute_sc2(means[f'{i+1}_{i+1}'],means_inter,np.ones(len(cl[i])))
        sc2s.append(sc2)
        eucl_distances.append(eucl_distance)
        sc,min_dist,_,e_dists = compute_sc(means_inter,means_intra,np.ones(len(cl[i])))
        scs.append(sc)
        min_dists.append(min_dist)
        e_distss.append(e_dists)
    

    return scs,sc2s,eucl_distances,min_dists,e_distss,means,stds


def get_outliers(labels,cl):
    """
    Identifies and returns the outlier elements based on DBSCAN clustering labels.

    Parameters:
        labels (list or array-like): Cluster labels assigned by DBSCAN, where outliers are labeled as -1.
        cl (dict): Dictionary whose values correspond to the data points in the same order as the labels.

    Returns:
        list: A list of data points (from cl.values()) that are considered outliers (i.e., have a label of -1).
    """
    outliers = []
    for i in range(len(labels)):
        if labels[i] == -1:
            outliers.append(list(cl.values())[i])
    return outliers



def calculate_nearest_neighbors_all(dists):
    """
    Calculates the nearest neighbors for each key based on provided distance dictionaries.
    Args:
        dists (list of dict): A list where each element is a dictionary mapping keys to dictionaries of neighbor keys and their distances.
    Returns:
        tuple:
            - neighbors (dict): A dictionary mapping each key to its three nearest neighbors (as a dict of neighbor keys and distances).
            - (np.ndarray, np.ndarray): A tuple containing an array of unique neighbor identifiers and their corresponding counts across all keys.
    Notes:
        - The function assumes that each dictionary in `dists` has the same set of keys.
        - Neighbor keys are processed to extract unique identifiers, either by splitting on '_' or concatenating parts.
    """
    nn = {}
    neighbors = {}
    all_neighbors = []
    for key, value in dists[0].items():
        
        nn[key] = value
        for d in dists:
            nn[key].update(d[key])

    for key,value in nn.items():
        
        val_list = dict(sorted(value.items(), key=lambda x:x[1])[:3])
        for n in val_list.keys():
            splits = n.split('_')
            if len(splits) ==2:
                all_neighbors.append(splits[0])
            else:
                all_neighbors.append(splits[0]+splits[1])
        # all_neighbors.extend([n.split('_')[0] for n in val_list.keys()])
        neighbors[key] = val_list
        
    return neighbors,np.unique(all_neighbors,return_counts=True)




def get_nn_noisy(cl,centroids,labels):
    """
    Computes the nearest neighbors for noisy (outlier) points across multiple clusters.

    For each cluster, this function:
        - Identifies outliers within the cluster.
        - Calculates distances from outliers to the cluster's centroids and to outliers in other clusters.
        - Aggregates these distances and computes the nearest neighbors for all outliers.

    Args:
        cl (list): List of clusters, where each element represents a cluster's data points.
        centroids (list): List of centroids for each cluster.
        labels (list): List of label arrays corresponding to each cluster.

    Returns:
        list: A list where each element contains the nearest neighbors for the noisy points in the corresponding cluster.
    """
    nn = []
    for i in range(len(cl)):
        dists_outliers = []
        dists_outliers_out = []
        outliers = get_outliers(labels[i],cl[i])
        dist_outliers_centroids, _ = calculate_distances_diff_cl_clusters(outliers,centroids[i],i+1)
        dist_outliers_out, _ = calculate_distances_clusters(outliers,f'{i+1}_n')
        dists_outliers.append(dist_outliers_centroids)
        dists_outliers_out.append(dist_outliers_out)
        for j in range(len(cl)):
            if j!=i:
                dist_outliersi_j, _ = calculate_distances_diff_cl_clusters(outliers,centroids[j],j+1)
                dist_outliers_outi_j, _ = calculate_distances_diff_cl_clusters(outliers,get_outliers(labels[j],cl[j]),f'{j+1}_n')
                dists_outliers.append(dist_outliersi_j)
                dists_outliers_out.append(dist_outliers_outi_j)
        dists_outliers.extend(dists_outliers_out)

        _, nn_all = calculate_nearest_neighbors_all(dists_outliers)
        nn.append(nn_all)
    return nn



def calculate_percentile_value(cl,min,max):
    """
    Calculates a percentile-based value using a logarithmic scaling between specified minimum and maximum values.

    Parameters:
        cl (float): The cluster label or value to be transformed.
        min (float): The minimum value of the target range.
        max (float): The maximum value of the target range.

    Returns:
        float: The calculated value, rounded if greater than or equal to 1, otherwise returns 1.

    Notes:
        The function uses a logarithmic transformation to scale the input `cl` between `min` and `max`.
        If the computed value is less than 1, the function returns 1.
    """
    a1 = 42
    a2 = min
    b1 = 7063
    b2 = max
    a = (b2-a2)/np.log(b1/a1)
    b = a2 - ((b2-a2)/np.log(b1/a1))*np.log(a1)
    log = a*np.log(cl)+b
    if log>=1:
        return np.round(log)
    else:
        return 1
    

def get_min_samples(cl,dist,min,max):
    """
    Calculates an appropriate value for the `min_samples` parameter used in DBSCAN clustering based on the distribution of pairwise distances between cluster elements.
    Args:
        cl (dict): A dictionary where values represent feature vectors of cluster elements.
        dist (array-like): An array or list of pairwise distances between elements.
        min (int): The minimum percentile value to consider when calculating `min_samples`. Default is 50.
        max (int): The maximum percentile value to consider when calculating `min_samples`. Default is 90.
    Returns:
        int: The computed `min_samples` value, which is typically used as a parameter for DBSCAN clustering.
    """
    dist_matrix = euclidean_distances(list(cl.values()),list(cl.values()))
    samples = []
    percentile = 1
    aux = dist_matrix<np.percentile(dist,10)
    try:
        for i in range(dist_matrix.shape[0]):
            samples.append(np.unique(aux[i],return_counts=True)[1][1]-1)
        if np.percentile(samples,calculate_percentile_value(len(cl),min,max)).astype(int)>=1:
            percentile = np.percentile(samples,calculate_percentile_value(len(cl),min,max)).astype(int)
    except:
        percentile = 1
    return percentile



def get_core_noncore_points(cl_dict1, cl_dict2,labels):
    """
    Identifies core and non-core points based on feature similarity and clustering labels.

    Given two dictionaries of features and a list of cluster labels, this function finds:
    - Core points: patches in `cl_dict1` whose features exactly match any feature in `cl_dict2`.
    - Non-core points: patches in `cl_dict1` that are assigned to a cluster (label != -1) but are not core points.

    Args:
        cl_dict1 (dict): Dictionary mapping patch identifiers to feature vectors.
        cl_dict2 (dict): Dictionary mapping patch identifiers to feature vectors.
        labels (list or array-like): Cluster labels for each patch in `cl_dict1`.

    Returns:
        tuple:
            centroids (dict): Dictionary of core points (patches with matching features in `cl_dict2`).
            non_core (dict): Dictionary of non-core points (clustered patches not in `centroids`).
    """
    centroids = {}
    non_core = {}
    i = 0
    for patch,feat in cl_dict1.items():
        for j,feat1 in enumerate(cl_dict2):
            if spatial.distance.euclidean(feat,feat1) == 0:
                centroids[patch] = feat1
    for i in range(len(labels)):
        if labels[i] != -1 and list(cl_dict1.keys())[i] not in list(centroids.keys()):
            non_core[list(cl_dict1.keys())[i]] = list(cl_dict1.values())[i]
        i=+1
    return centroids,non_core




def get_silhouette_core_noisy(core,noncore, noisy):
    """
    Calculates silhouette scores for core, non-core, and noisy samples.

    This function takes in core, non-core, and noisy sample collections, assigns cluster labels
    (0 for core, 1 for non-core, 2 for noisy), and computes silhouette scores for all samples.
    It then splits the silhouette scores into separate lists for core, non-core, and noisy samples.

    Args:
        core (dict): Dictionary of core samples, where values are sample feature vectors.
        noncore (dict): Dictionary of non-core samples, where values are sample feature vectors.
        noisy (list): List of noisy sample feature vectors.

    Returns:
        list: A list containing:
            - s_cores (np.ndarray): Silhouette scores for all samples (core, non-core, noisy).
            - s_core (np.ndarray): Silhouette scores for core samples.
            - s_noncore (np.ndarray): Silhouette scores for non-core samples.
            - s_noisy (np.ndarray): Silhouette scores for noisy samples.
    """
    cores = copy.deepcopy(list(core.values()))
    labels = list(np.zeros(len(cores)).astype(int))
    cores.extend(list(noncore.values()))
    labels.extend(list(np.ones(len(list(noncore.values()))).astype(int)))
    cores.extend(noisy)
    labels.extend(list(2*np.ones(len(noisy))))

    s_cores = silhouette_samples(cores,labels) 
    s_core = s_cores[:len(core)]
    s_noncore = s_cores[len(core):len(core)+len(noncore)]
    s_noisy = s_cores[len(core)+len(noncore):]

    return [s_cores,s_core,s_noncore,s_noisy]



def get_nn_core_non_core(cl_2,cl_3,cl_4,centroids_2,centroids_3,centroids_4, labels_2,labels_3,labels_4):
    """
    Given clusters, centroids, and labels for three different clusterings, this function identifies core and non-core points,
    calculates distances between them, finds the nearest neighbors for non-core points, and returns updated dictionaries
    containing both core points and their nearest non-core neighbors for each clustering.

    Args:
        cl_2 (dict): Dictionary of points in cluster 2.
        cl_3 (dict): Dictionary of points in cluster 3.
        cl_4 (dict): Dictionary of points in cluster 4.
        centroids_2 (np.ndarray): Centroids for cluster 2.
        centroids_3 (np.ndarray): Centroids for cluster 3.
        centroids_4 (np.ndarray): Centroids for cluster 4.
        labels_2 (np.ndarray): Cluster labels for cluster 2.
        labels_3 (np.ndarray): Cluster labels for cluster 3.
        labels_4 (np.ndarray): Cluster labels for cluster 4.

    Returns:
        tuple: A tuple containing three dictionaries (cl_2_aux, cl_3_aux, cl_4_aux), each representing the union of core points
               and their nearest non-core neighbors for clusters 2, 3, and 4, respectively.
    """
    core2,non_core2 = get_core_noncore_points(cl_2,centroids_2,labels_2)
    dist_core2,allcore2 = calculate_distances_core_noncore(core2,non_core2)
    nn_non_core2 = calculate_nearest_neighbors_non_core(dist_core2,cl_2)
    cl_2_aux = core2.copy()
    cl_2_aux.update(nn_non_core2)
    core3,non_core3 = get_core_noncore_points(cl_3,centroids_3,labels_3)
    dist_core3,allcore3 = calculate_distances_core_noncore(core3,non_core3)
    nn_non_core3 = calculate_nearest_neighbors_non_core(dist_core3,cl_3)
    cl_3_aux = core3.copy()
    cl_3_aux.update(nn_non_core3)
    core4,non_core4 = get_core_noncore_points(cl_4,centroids_4,labels_4)
    dist_core4,allcore4 = calculate_distances_core_noncore(core4,non_core4)
    nn_non_core4 = calculate_nearest_neighbors_non_core(dist_core4,cl_4)
    cl_4_aux = core4.copy()
    cl_4_aux.update(nn_non_core4)
    return cl_2_aux,cl_3_aux,cl_4_aux




def get_weights(scs, outliers_scs, dists, n_classes, dbscan=True, backgroud=True):
    """
    Compute weights and softmax-normalized percentages.

    Args:
        scs: list of scores
        outliers_scs: list of outlier scores
        dists: list of distances
        n_classes: int, number of foreground classes (not counting background)
        dbscan: bool, whether to use DBSCAN logic in compute_weights
        backgroud: bool, if True prepend fixed [10, 0, ..., 0] (background emphasis)
        

    Returns:
        weights (list of lists), percentages (list of arrays)
    """
    weights = []
    soft = torch.nn.Softmax()
    
    # Add first fixed weight for background if requested
    if backgroud:
        vector_len = n_classes + 1  # foreground classes + background
        vec = [0] * vector_len
        vec[0] = 10   # background
        weights.append(vec)

    # Compute dynamic weights
    for i, sc in enumerate(scs):
        outliers = outliers_scs.copy()
        outliers.pop(i)

        # shift index if backgroud=True (to keep same behavior as before)
        idx = i + 1 if backgroud else i

        weights.append(
            compute_weights(outliers_scs[i], outliers, dists[i], sc, idx, dbscan=dbscan)
        )

    # Softmax normalization
    percentages = [
        soft(torch.from_numpy(np.array(w, dtype=float))).cpu().detach().numpy()
        for w in weights
    ]

    return weights, percentages
