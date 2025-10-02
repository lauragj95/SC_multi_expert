# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:29:30 2024

@author: Laura
"""

import numpy as np
import torch
import torch
from utils.maps_utils import uncertainty_map,get_svls_filter_2d,soft_smooth_expert_map,sc_expert_map




        
def calculate_soft(maps,original=None,num_cl=6,size=4096):
    """
    Calculates soft expert maps and an uncertainty map from a list of input maps.

    Args:
        maps (list): List of input maps to be processed.
        original (optional): An optional original map to be processed. Default is None.
        num_cl (int, optional): Number of classes to use in soft smoothing. Default is 6.
        size (int, optional): Size parameter for the uncertainty map. Default is 4096.

    Returns:
        tuple: A tuple containing:
            - soft_maps (list): List of soft-smoothed expert maps.
            - soft_mapOrig: Soft-smoothed original map, or None if not provided.
            - soft_uncertainty_map: Uncertainty map computed from the soft maps.
    """
    soft_maps = [soft_smooth_expert_map(mp,num_cl) for mp in maps]
    if original!=None:
        soft_mapOrig = soft_smooth_expert_map(original)
    else:
        soft_mapOrig = None
    soft_uncertainty_map = uncertainty_map(soft_maps,original=soft_mapOrig,cl=num_cl,size=size)
    return soft_maps,soft_mapOrig,soft_uncertainty_map
    


def calculate_smooth(maps,original=None,num_cl=6,size=4096):
    """
    Applies smoothing and uncertainty estimation to a list of expert maps.

    Args:
        maps (list or iterable): List of expert maps (numpy arrays) to be smoothed.
        original (numpy.ndarray, optional): Original map to be smoothed and used for uncertainty calculation. Defaults to None.
        num_cl (int, optional): Number of classes (channels) in the maps. Defaults to 6.
        size (int, optional): Size of the maps (assumed square). Defaults to 4096.

    Returns:
        tuple:
            svls_labels (list of numpy.ndarray): List of smoothed expert maps after applying the SVLS filter.
            svls_labels_or (numpy.ndarray or None): Smoothed original map after SVLS filter, or None if original is not provided.
            smooth_uncertainty_map (numpy.ndarray): Uncertainty map computed from the smoothed maps.

    Notes:
        - Requires the functions `get_svls_filter_2d`, `soft_smooth_expert_map`, and `uncertainty_map` to be defined elsewhere.
        - Assumes input maps are compatible with the expected preprocessing and torch tensor conversion.
    """
    svls_layer = get_svls_filter_2d(ksize=3, sigma=1, channels=torch.tensor(num_cl))
    smooth_maps = [torch.from_numpy(soft_smooth_expert_map(mp,num_cl,size)).to(torch.int64).unsqueeze(0).contiguous().float() for mp in maps]
    svls_labels = [svls_layer(smooth_map).squeeze(0).cpu().detach().numpy() for smooth_map in smooth_maps]

    if original!=None:
        smooth_or_map = torch.from_numpy(soft_smooth_expert_map(original,size)).to(torch.int64).unsqueeze(0).contiguous().float()
        svls_labels_or = svls_layer(smooth_or_map).squeeze(0).cpu().detach().numpy()
    else:
        svls_labels_or = None
    smooth_uncertainty_map= uncertainty_map(svls_labels,original=svls_labels_or,cl=num_cl,size=size)
    return svls_labels,svls_labels_or,smooth_uncertainty_map



def calculate_sc(maps,weights,original=None,weights_orig=None,num_cl=6,size=4096):
    """
    Calculates self-consistency expert maps, an optional original expert map, and an uncertainty map.

    Args:
        maps (list or np.ndarray): List or array of expert maps to process.
        weights (list or np.ndarray): List or array of weights corresponding to each expert map.
        original (np.ndarray, optional): Original expert map for comparison. Defaults to None.
        weights_orig (np.ndarray, optional): Weights for the original expert map. Required if `original` is provided.
        num_cl (int, optional): Number of classes for the uncertainty map. Defaults to 6.
        size (int, optional): Size of the output maps. Defaults to 4096.

    Returns:
        tuple:
            soft_sc (list): List of self-consistency expert maps.
            soft_sc_or (np.ndarray or None): self-consistency original expert map if `original` is provided, else None.
            sc_uncertainty_map (np.ndarray): Uncertainty map computed from the self-consistency expert maps.
    """
    soft = torch.nn.Softmax()
    sc_maps = [sc_expert_map(maps[i],weights[i],size) for i in range(len(maps))]
    soft_sc = [soft(torch.from_numpy(sc_map)).cpu().detach().numpy() for sc_map in sc_maps]

    if original!=None:
        sc_or_map = sc_expert_map(original,weights_orig,size)
        soft_sc_or = soft(torch.from_numpy(sc_or_map)).cpu().detach().numpy()
    else:
        soft_sc_or = None
    sc_uncertainty_map= uncertainty_map(soft_sc,original=None,cl=num_cl,size=size)
    return soft_sc,soft_sc_or,sc_uncertainty_map




