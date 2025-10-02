import numpy as np

def compute_sc(means,means_cl, labels):
    """
    Compute the self-consistency (sc) score based on e-distance for clustering results, accounting for outliers.

    Parameters:
        means (list or np.ndarray): List or array of mean values for each cluster.
        means_cl (list or np.ndarray): List or array of mean values for the clusters, where means_cl[0] is a reference value.
        labels (list or np.ndarray): Cluster labels for each data point. Outliers are labeled as -1.

    Returns:
        sc (float): The self-consistency score, adjusted for outliers.
        min_dist (float): The minimum distance computed among clusters.
        outliers_sc (float): The outlier self-consistency adjustment factor.
        dists (list): List of computed distances for each cluster.
    """
    clust,counts = np.unique(labels,return_counts=True)
    clust_dict = dict(zip(clust,counts))
    if -1 in clust_dict.keys():
        outliers_sc = 1 - clust_dict[-1]/len(labels)
    else:
        outliers_sc = 1
    dists = []
    for i,mean in enumerate(means):
        dists.append(2*mean-means_cl[0]-means_cl[i+1])
    sc = outliers_sc * min(dists)
    return sc, min(dists),outliers_sc, dists




def compute_sc2(mean_intra,means, labels):
    """
    Compute a self-consistency (SC) score based on Euclidean distance for clustering results, accounting for outliers.

    Parameters:
        mean_intra (float): The mean intra-cluster distance.
        means (list or np.ndarray): List or array of mean distances for each cluster.
        labels (list or np.ndarray): Cluster labels for each sample. Outliers should be labeled as -1.

    Returns:
        tuple:
            sc (float): The self-consistency score, adjusted for outliers.
            dist_sc (float): The difference between the minimum cluster mean and the intra-cluster mean.

    Notes:
        - Outliers are identified by the label -1.
        - The SC score is penalized by the proportion of outliers in the data.
    """
    clust,counts = np.unique(labels,return_counts=True)
    clust_dict = dict(zip(clust,counts))
    if -1 in clust_dict.keys():
        outliers_sc = 1 - clust_dict[-1]/len(labels)
    else:
        outliers_sc = 1
    dist_sc = min(means) - mean_intra
    sc = outliers_sc * dist_sc
    return sc,dist_sc