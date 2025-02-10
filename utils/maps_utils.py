# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:29:30 2024

@author: Laura
"""

import numpy as np
import torch
import math


def sc_expert_map(expert_mask,weights):
    sc_map = np.zeros((6,4096,4096))
    for cl in range(6):
        for m in range(6):
            sc_map[m,:,:][expert_mask==cl] = weights[cl][m]
    return sc_map

def soft_smooth_expert_map(expert_mask):
    sc_map = np.zeros((6,4096,4096))
    for cl in range(6):
        sc_map[cl,:,:][expert_mask==cl] = 1
    return sc_map


def get_gaussian_kernel_2d(ksize=0, sigma=0):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance + 1e-16)) * torch.exp( 
        -torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance + 1e-16)
        )
    return gaussian_kernel / torch.sum(gaussian_kernel)

class get_svls_filter_2d(torch.nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_svls_filter_2d, self).__init__()
        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)
        neighbors_sum = (1 - gkernel[1,1]) + 1e-16
        gkernel[int(ksize/2), int(ksize/2)] = neighbors_sum
        self.svls_kernel = gkernel / neighbors_sum
        svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
        svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
        padding = int(ksize/2)
        self.svls_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding, padding_mode='replicate')
        self.svls_layer.weight.data = svls_kernel_2d
        self.svls_layer.weight.requires_grad = False
    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()



def uncertainty_map(maps,original=None):
    uncertainty_map = np.zeros((6,4096,4096))
    for i in range(6):
        map_cl = np.zeros((4096,4096))
        for map in maps:
            map_cl += map[i]
        if original!=None:
            map_cl =+ original[i]
        uncertainty_map[i] = (1/len(maps))*map_cl
    return uncertainty_map


def calculate_sc_maps(experts,original,weights,weightsOrig):
    soft = torch.nn.Softmax()
    soft_sc = []
    
    for i,expert in enumerate(experts):
        sc_map = sc_expert_map(expert,weights[i])
        soft_sc.append(soft(torch.from_numpy(sc_map)).cpu().detach().numpy())

    sc_or_map = sc_expert_map(original,weightsOrig)
    soft_sc_or = soft(torch.from_numpy(sc_or_map)).cpu().detach().numpy()

    sc_map = uncertainty_map(soft_sc)

    return sc_map,soft_sc_or

def calculate_soft_maps(maps,original=None):
    if original!=None:
        soft_mapOrig = soft_smooth_expert_map(original)
    else:
        soft_mapOrig = None
    soft_maps = []
    for map in maps:
        soft_maps.append(soft_smooth_expert_map(map))
        soft_uncertainty_map = uncertainty_map(soft_maps,original=soft_mapOrig)
    
    
    return soft_maps,soft_mapOrig,soft_uncertainty_map
    
def calculate_smooth_maps(maps,original=None):
    svls_layer = get_svls_filter_2d(ksize=3, sigma=1, channels=torch.tensor(6))
    if original!=None:
        smooth_or_map = torch.from_numpy(soft_smooth_expert_map(original)).to(torch.int64).unsqueeze(0).contiguous().float()
        svls_labels_or = svls_layer(smooth_or_map).squeeze(0).cpu().detach().numpy()
    else:
        svls_labels_or = None
    
    svls_labels = []
    for map in maps:
        smooth_map = torch.from_numpy(soft_smooth_expert_map(map)).to(torch.int64).unsqueeze(0).contiguous().float()
        svls_labels.append(svls_layer(smooth_map).squeeze(0).cpu().detach().numpy())

    smooth_uncertainty_map= uncertainty_map(svls_labels,original=svls_labels_or)
    return svls_labels,svls_labels_or,smooth_uncertainty_map
