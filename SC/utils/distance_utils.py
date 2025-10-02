# -*- coding: utf-8 -*-



import numpy as np
import torch
import torch


def calculate_distances(cl):
    """
    Calculates the pairwise distances between the values in the input dictionary.
    Args:
        cl (dict): A dictionary where the values are array-like objects representing points or vectors.
    Returns:
        numpy.ndarray: A flattened array containing the pairwise distances between all values in the input dictionary.
    Note:
        The function assumes that the values in the dictionary can be converted to numpy arrays and are suitable for distance computation.
    """
    dist_matrix = torch.cdist(torch.from_numpy(np.array(list(cl.values()))),torch.from_numpy(np.array(list(cl.values())))).cpu().detach().numpy()
    return dist_matrix.flatten()
 


def calculate_distances_diff_cl(cl1,cl2):
    """
    Calculates the pairwise Euclidean distances between the values of two dictionaries and returns the distances as a flattened NumPy array.
    Args:
        cl1 (dict): First dictionary with values that can be converted to a NumPy array.
        cl2 (dict): Second dictionary with values that can be converted to a NumPy array.
    Returns:
        numpy.ndarray: A 1D array containing the flattened pairwise distances between the values of cl1 and cl2.
    Note:
        - Both cl1 and cl2 should have values that are compatible with conversion to NumPy arrays and PyTorch tensors.
        - Requires `torch` and `numpy` to be imported.
    """
    dist_matrix = torch.cdist(torch.from_numpy(np.array(list(cl1.values()))),torch.from_numpy(np.array(list(cl2.values())))).cpu().detach().numpy()
    return dist_matrix.flatten()



def calculate_distances_clusters(cl_dict,cluster):
    """
    Calculates pairwise Euclidean distances between feature vectors in a cluster dictionary.
    Args:
        cl_dict (dict): A dictionary where each key corresponds to a feature vector (e.g., numpy arrays).
        cluster (int or str): The cluster identifier used for labeling distance keys.
    Returns:
        tuple:
            - dist (dict): A nested dictionary where dist[i][f'cl{cluster}_{j}'] contains the distance between feature vectors i and j (excluding self-distances).
            - all_dist (list): A list of all computed pairwise distances (including self-distances).
    """
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
    """
    Calculates pairwise Euclidean distances between features from two different cluster dictionaries.
    Args:
        cl_dict1 (dict): A dictionary where each value is a feature vector (e.g., numpy array) representing a sample from the first cluster.
        cl_dict2 (dict): A dictionary where each value is a feature vector (e.g., numpy array) representing a sample from the second cluster.
        cluster (int or str): The cluster identifier used for labeling the distance keys.
    Returns:
        tuple:
            - dist (dict): A nested dictionary where dist[i][f'cl{cluster}_{j}'] contains the distance between the i-th feature in cl_dict1 and the j-th feature in cl_dict2.
            - all_dist (list): A flat list of all computed distances between features in cl_dict1 and cl_dict2.
    Note:
        This function requires PyTorch and expects the feature vectors to be compatible with torch.from_numpy.
    """
    dist = {}
    all_dist = []
    for i,feat in enumerate(cl_dict1):
        dist[i] = {}

        for j,feat1 in enumerate(cl_dict2):
            distance = torch.nn.PairwiseDistance(p=2)(torch.from_numpy(feat),torch.from_numpy(feat1)).cpu().detach().numpy()
            dist[i][f'cl{cluster}_{j}']= distance
            all_dist.append(dist[i][f'cl{cluster}_{j}'])

    return dist,all_dist



def calculate_distances_core_noncore(cl_dict1, cl_dict2):
    """
    Calculates pairwise Euclidean distances between feature vectors in two dictionaries.
    Given two dictionaries where keys represent identifiers and values are feature vectors (as numpy arrays),
    this function computes the pairwise Euclidean (L2) distance between each feature vector in `cl_dict1` and each in `cl_dict2`.
    Returns a nested dictionary of distances and a flat list of all computed distances.
    Args:
        cl_dict1 (dict): Dictionary mapping keys to feature vectors (numpy arrays).
        cl_dict2 (dict): Dictionary mapping keys to feature vectors (numpy arrays).
    Returns:
        tuple:
            dist (dict): Nested dictionary where dist[key][key1] is the distance between cl_dict1[key] and cl_dict2[key1].
            all_dist (list): List of all computed distances.
    """
    dist = {}

    all_dist = []
    for key,feat in cl_dict1.items():
        dist[key] = {}

        for key1,feat1 in cl_dict2.items():
            dist[key][key1]= torch.nn.PairwiseDistance(p=2)(torch.from_numpy(feat),torch.from_numpy(feat1)).cpu().detach().numpy()
            all_dist.append(dist[key][key1])
 
    return dist,all_dist




def edistance(mean_c, mean_cj,mean_j):
    """
    Calculates a custom distance metric based on three mean values.
    Args:
        mean_c (float): The mean value of the current class or cluster.
        mean_cj (float): The mean value between the current and another class or cluster.
        mean_j (float): The mean value of the other class or cluster.
    Returns:
        float: The computed distance using the formula 2*mean_cj - mean_c - mean_j.
    """
    distance = 2*mean_cj-mean_c-mean_j
    return distance



