import numpy as np

def compute_sc(means,means_cl, labels):
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


def edistance(mean_c, mean_cj,mean_j):
    distance = 2*mean_cj-mean_c-mean_j
    return distance


def compute_sc2(mean_intra,means, labels):
    clust,counts = np.unique(labels,return_counts=True)
    clust_dict = dict(zip(clust,counts))
    if -1 in clust_dict.keys():
        outliers_sc = 1 - clust_dict[-1]/len(labels)
    else:
        outliers_sc = 1
    dist_sc = min(means) - mean_intra
    sc = outliers_sc * dist_sc
    return sc,dist_sc