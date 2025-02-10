# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:29:30 2024

@author: Laura
"""

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
from utils.distance_utils import calculate_distances_clusters, calculate_distances_diff_cl_clusters,calculate_distances_core_noncore
from self_consistency.sc import compute_sc,compute_sc2
from self_consistency.weights import compute_weights


def calculate_nearest_neighbors_non_core(dist,cl):
    neighbors = []
    all_neighbors = {}

    for key,value in dist.items():
        val_list = dict(sorted(value.items(), key=lambda x:x[1])[:2])
        for key1 in val_list.keys():
            all_neighbors[key1]=cl[key1]
        neighbors.extend(val_list.keys())
        
        
    return all_neighbors

def get_noisy(labels):
    noisy = []
    for label in labels:
        clusters, count = np.unique(label,return_counts=True)
        dict_aux = dict(zip(clusters,count))
        noisy.append(dict_aux[-1])
    return noisy

def get_core_expert_OLD(expert,cl2,cl3,cl4,dist2,dist3,dist4):
    if expert ==1:
        dbscan2 = DBSCAN(eps=np.percentile(dist2,10), min_samples=75).fit(list(cl2.values()))
        centroids_2 = dbscan2.components_
        labels_2 = dbscan2.labels_
    
        dbscan3 = DBSCAN(eps=np.percentile(dist3,10), min_samples=55).fit(list(cl3.values()))
        centroids_3 = dbscan3.components_
        labels_3 = dbscan3.labels_
    
    
        dbscan4 = DBSCAN(eps=np.percentile(dist4,10), min_samples=185).fit(list(cl4.values()))
        centroids_4 = dbscan4.components_
        labels_4 = dbscan4.labels_
        
    elif expert == 2:
        dbscan3 = DBSCAN(eps=np.percentile(dist3,10), min_samples=54).fit(list(cl3.values()))
        centroids_3 = dbscan3.components_
        labels_3 = dbscan3.labels_

        dbscan4 = DBSCAN(eps=np.percentile(dist4,10), min_samples=143).fit(list(cl4.values()))
        centroids_4 = dbscan4.components_
        labels_4 = dbscan4.labels_

        dbscan2 = DBSCAN(eps=np.percentile(dist2,10), min_samples=97).fit(list(cl2.values()))
        centroids_2 = dbscan2.components_
        labels_2 = dbscan2.labels_
        
    elif expert == 3 :
        dbscan2 = DBSCAN(eps=np.percentile(dist2,10), min_samples=79).fit(list(cl2.values()))
        centroids_2 = dbscan2.components_
        labels_2 = dbscan2.labels_

        dbscan3 = DBSCAN(eps=np.percentile(dist3,10), min_samples=60).fit(list(cl3.values()))
        centroids_3 = dbscan3.components_
        labels_3 = dbscan3.labels_

        dbscan4 = DBSCAN(eps=np.percentile(dist4,10), min_samples=152).fit(list(cl4.values()))
        centroids_4 = dbscan4.components_
        labels_4 = dbscan4.labels_


def get_core_expert(cls,dists):
    centroids = []
    labels = []
    epss = []
    min_sampless = []
    for i in range(len(cls)):
        eps = np.percentile(dists[i],10)
        min_samples = get_min_samples(cls[i],dists[i])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(list(cls[i].values()))
        centroids.append(dbscan.components_)
        labels.append(dbscan.labels_)
        epss.append(eps)
        min_sampless.append(min_samples)

    return centroids,labels,epss,min_sampless



def calculate_nearest_neighbors(dist, dist_others):
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



def get_metrics(centroids,labels):
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
    for i,centroid in enumerate(centroids):
        i +=1
        dists = []
        dist_intra,all_intra = calculate_distances_clusters(centroid,i+1)
        means[f'{i}_{i}'] = sum(all_intra)/(len(centroid)**2)
        stds[f'{i}_{i}'] = np.std(all_intra)
        for j, centroid_other in enumerate(centroids):
            j +=1
            if j!=i:
                dist_inter,all_inter = calculate_distances_diff_cl_clusters(centroid,centroid_other,j+1)
                dists.append(dist_inter)
                if f'{i}_{j}' or f'{j}_{i}' not in means.keys():
                    means[f'{i}_{j}'] = np.mean(all_inter)
                    stds[f'{i}_{j}'] = np.std(all_inter)
        neighbors.append(calculate_nearest_neighbors(dist_intra,dists))

    for i, label in enumerate(labels):
        i+=1
        means_inter = []
        means_intra = []
        means_intra.append(means[f'{i}_{i}'])
        for j in range(len(labels)):
            j +=1
            if j!=i:
                means_inter.append(means[f'{i}_{j}']) 
            if f'{j}_{j}' != f'{i}_{i}':
                means_intra.append(means[f'{j}_{j}']) 
        sc2,eucl_distance = compute_sc2(means[f'{i}_{i}'],means_inter, label)
        sc2s.append(sc2)
        eucl_distances.append(eucl_distance)
        sc,min_dist,outliers_sc,e_dists = compute_sc(means_inter,means_intra,label)
        scs.append(sc)
        min_dists.append(min_dist)
        outliers_scs.append(outliers_sc)
        e_distss.append(e_dists)

    return scs,sc2s,eucl_distances,min_dists,outliers_scs,e_distss,means,stds,neighbors


def get_outliers(labels,cl):
    outliers = []
    for i in range(len(labels)):
        if labels[i] == -1:
            outliers.append(list(cl.values())[i])
    return outliers

def calculate_nearest_neighbors_all(dists):
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



def calculate_percentile_value(cl):
    a1 = 42
    a2 = 50
    b1 = 7063
    b2 = 90
    a = (b2-a2)/np.log(b1/a1)
    b = a2 - ((b2-a2)/np.log(b1/a1))*np.log(a1)
    log = a*np.log(cl)+b
    return np.round(log)

def calculate_base_min_samples(cl):
    a1 = 74
    a2 = 67
    b1 = 1000
    b2 = 274
    a = (b2-a2)/np.log(b1/a1)
    b = a2 - ((b2-a2)/np.log(b1/a1))*np.log(a1)
    log = a*np.log(cl)+b
    return np.round(log)

def get_min_samples(cl,dist):
    dist_matrix = euclidean_distances(list(cl.values()),list(cl.values()))
    samples = []
    aux = dist_matrix<np.percentile(dist,10)
    for i in range(dist_matrix.shape[0]):
        samples.append(np.unique(aux[i],return_counts=True)[1][1]-1)
    return np.percentile(samples,calculate_percentile_value(len(cl))).astype(int)



def get_core_noncore_points(cl_dict1, cl_dict2,labels):
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

def get_nn_core_non_core(cl_2,cl_3,cl_4,centroids_2,centroids_3,centroids_4, labels_2,labels_3,labels_4):
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



def get_weights(scs,outliers_scs,dists):
    weights = []
    soft = torch.nn.Softmax()
    weights.append([10,0,0,0,0,0])
    for i,sc in enumerate(scs):
        outliers = outliers_scs.copy()
        outliers.pop(i)
        weights.append(compute_weights(outliers_scs[i],outliers,dists[i],sc,i+1))
    
    percentage = []
    for w in weights:
        percentage.append(soft(torch.from_numpy(np.array(w).astype('float'))).cpu().detach().numpy())   

    return weights,percentage