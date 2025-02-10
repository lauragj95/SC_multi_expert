# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:29:30 2024

@author: Laura
"""

import numpy as np
import torch
import torch
from utils.maps_utils import uncertainty_map,get_svls_filter_2d,soft_smooth_expert_map,sc_expert_map




        
def calculate_soft(maps,original=None):
    soft_maps = [soft_smooth_expert_map(mp) for mp in maps]
    if original!=None:
        soft_mapOrig = soft_smooth_expert_map(original)
    else:
        soft_mapOrig = None
    soft_uncertainty_map = uncertainty_map(soft_maps,original=soft_mapOrig)
    return soft_maps,soft_mapOrig,soft_uncertainty_map
    
def calculate_smooth(maps,original=None):
    svls_layer = get_svls_filter_2d(ksize=3, sigma=1, channels=torch.tensor(6))
    smooth_maps = [torch.from_numpy(soft_smooth_expert_map(mp)).to(torch.int64).unsqueeze(0).contiguous().float() for mp in maps]
    svls_labels = [svls_layer(smooth_map).squeeze(0).cpu().detach().numpy() for smooth_map in smooth_maps]

    if original!=None:
        smooth_or_map = torch.from_numpy(soft_smooth_expert_map(original)).to(torch.int64).unsqueeze(0).contiguous().float()
        svls_labels_or = svls_layer(smooth_or_map).squeeze(0).cpu().detach().numpy()
    else:
        svls_labels_or = None
    smooth_uncertainty_map= uncertainty_map(svls_labels,original=svls_labels_or)
    return svls_labels,svls_labels_or,smooth_uncertainty_map

def calculate_sc(maps,weights,original=None,weights_orig=None):
    soft = torch.nn.Softmax()
    sc_maps = [sc_expert_map(maps[i],weights[i]) for i in range(len(maps))]
    soft_sc = [soft(torch.from_numpy(sc_map)).cpu().detach().numpy() for sc_map in sc_maps]

    if original!=None:
        sc_or_map = sc_expert_map(original,weights_orig)
        soft_sc_or = soft(torch.from_numpy(sc_or_map)).cpu().detach().numpy()
    else:
        soft_sc_or = None
    sc_uncertainty_map= uncertainty_map(soft_sc,original=None)
    return soft_sc,soft_sc_or,sc_uncertainty_map




