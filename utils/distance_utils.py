# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:29:30 2024

@author: Laura
"""



import numpy as np
import torch
import torch


def calculate_distances(cl):
    dist_matrix = torch.cdist(torch.from_numpy(np.array(list(cl.values()))),torch.from_numpy(np.array(list(cl.values())))).cpu().detach().numpy()
    return dist_matrix.flatten()


def calculate_distances_diff_cl(cl1,cl2):
    dist_matrix = torch.cdist(torch.from_numpy(np.array(list(cl1.values()))),torch.from_numpy(np.array(list(cl2.values())))).cpu().detach().numpy()
    return dist_matrix.flatten()


def calculate_distances_clusters(cl_dict,cluster):
    dist = {}

    all_dist = []
    for i,feat in enumerate(cl_dict):
        dist[i] = {}

        for j,feat1 in enumerate(cl_dict):
            distance = torch.nn.PairwiseDistance(p=2)(torch.from_numpy(feat),torch.from_numpy(feat1)).cpu().detach().numpy()
            all_dist.append(distance)
            if i!=j:
                dist[i][f'cl{cluster}_{j}'] = distance
                    


    return dist,all_dist

def calculate_distances_diff_cl_clusters(cl_dict1, cl_dict2,cluster):
    dist = {}

    all_dist = []
    for i,feat in enumerate(cl_dict1):
        dist[i] = {}

        for j,feat1 in enumerate(cl_dict2):
            distance = torch.nn.PairwiseDistance(p=2)(torch.from_numpy(feat),torch.from_numpy(feat1)).cpu().detach().numpy()
            dist[i][f'cl{cluster}_{j}']= distance#spatial.distance.euclidean(feat,feat1)
            all_dist.append(dist[i][f'cl{cluster}_{j}'])
 
    return dist,all_dist

def calculate_distances_core_noncore(cl_dict1, cl_dict2):
    dist = {}

    all_dist = []
    for key,feat in cl_dict1.items():
        dist[key] = {}

        for key1,feat1 in cl_dict2.items():
            dist[key][key1]= torch.nn.PairwiseDistance(p=2)(torch.from_numpy(feat),torch.from_numpy(feat1)).cpu().detach().numpy()
            all_dist.append(dist[key][key1])
 
    return dist,all_dist

def edistance(mean_c, mean_cj,mean_j):
    distance = 2*mean_cj-mean_c-mean_j
    return distance



