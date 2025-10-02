# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:29:30 2024

@author: Laura
"""

import numpy as np
import torch
import math


def sc_expert_map(expert_mask,weights,size=4096):
    """
    Generates a spatial map assigning weights to each expert based on an SC expert mask.

    Parameters:
        expert_mask (np.ndarray): A 2D array of shape (size, size) where each element indicates the expert class index assigned to that spatial location.
        weights (np.ndarray or list): A 2D array or nested list of shape (num_classes, num_experts) containing the weights to assign for each class-expert pair.
        size (int, optional): The spatial size of the output map (default is 4096).

    Returns:
        np.ndarray: A 3D array of shape (num_experts, size, size) where each slice along the first axis corresponds to the spatial map for an expert, with weights assigned according to the expert_mask and weights matrix.
    """
    sc_map = np.zeros((len(weights),size,size))
    for cl in range(len(weights)):
        for m in range(len(weights)):
            sc_map[m,:,:][expert_mask==cl] = weights[cl][m]
    return sc_map

def soft_smooth_expert_map(expert_mask,num_cl = 6,size = 4096):
    """
    Generates a one-hot encoded spatial map for expert masks.

    Given an expert mask with class labels, this function creates a 3D numpy array
    where each channel corresponds to a class, and the spatial locations belonging
    to that class are set to 1.

    Args:
        expert_mask (np.ndarray): 2D array of shape (size, size) containing integer class labels.
        num_cl (int, optional): Number of classes (channels) to encode. Default is 6.
        size (int, optional): Spatial size of the output map (height and width). Default is 4096.

    Returns:
        np.ndarray: 3D array of shape (num_cl, size, size) with one-hot encoded class maps.
    """
    sc_map = np.zeros((num_cl,size,size))
    for cl in range(num_cl):
        sc_map[cl,:,:][expert_mask==cl] = 1
    return sc_map




def get_gaussian_kernel_2d(ksize=0, sigma=0):
    """
    Generates a 2D Gaussian kernel tensor.

    Args:
        ksize (int): Size of the kernel (both width and height). Must be a positive integer.
        sigma (float): Standard deviation of the Gaussian distribution. Must be a positive number.

    Returns:
        torch.Tensor: A 2D tensor of shape (ksize, ksize) representing the normalized Gaussian kernel.

    Notes:
        - The kernel is normalized so that its sum is 1.
        - Requires the `torch` and `math` modules.
    """
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
    """
    A 2D spatially varying local smoothing (SVLS) filter implemented as a PyTorch module.

    This module creates a depthwise convolutional filter using a modified Gaussian kernel,
    where the center value is replaced by the sum of its neighbors, and normalizes the kernel.
    It is typically used for edge-preserving smoothing or denoising in multi-channel images.

    Args:
        ksize (int): Size of the square kernel. Must be an odd integer. Default is 3.
        sigma (float): Standard deviation for the Gaussian kernel. Default is 0.
        channels (int): Number of input and output channels. The filter is applied independently to each channel.

    Attributes:
        svls_kernel (torch.Tensor): The normalized SVLS kernel used for convolution.
        svls_layer (torch.nn.Conv2d): Depthwise convolutional layer with fixed SVLS kernel weights.

    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Output tensor after applying the SVLS filter, with the same shape as the input.
    """
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





def uncertainty_map(maps,original=None,cl=5,size=4096):
    """
    Computes an uncertainty map by aggregating multiple input maps for each class.

    Args:
        maps (list of np.ndarray): List of 3D numpy arrays of shape (cl, size, size), 
            representing the maps to be aggregated.
        original (np.ndarray, optional): An optional 3D numpy array of shape (cl, size, size) 
            to be added to each class map. Defaults to None.
        cl (int, optional): Number of classes (channels) in the maps. Defaults to 5.
        size (int, optional): Spatial size (height and width) of the maps. Defaults to 4096.

    Returns:
        np.ndarray: Aggregated uncertainty map of shape (cl, size, size).
    """
    uncertainty_map = np.zeros((cl,size,size))
    for i in range(cl):
        map_cl = np.zeros((size,size))
        for map in maps:
            map_cl += map[i]
        if original!=None:
            map_cl =+ original[i]
        uncertainty_map[i] = (1/len(maps))*map_cl
    return uncertainty_map






def calculate_sc_maps(experts,original,weights,weightsOrig,size=4096):
    """
    Calculates self-consistency (SC) maps for a list of expert models and an original model, and computes an uncertainty map.
    Args:
        experts (list): List of expert models or data representations to compute SC maps for.
        original: The original model or data representation for comparison.
        weights (list): List of weights corresponding to each expert.
        weightsOrig: Weights corresponding to the original model.
        size (int, optional): The size parameter for the uncertainty map. Defaults to 4096.
    Returns:
        tuple:
            - sc_map (numpy.ndarray): The computed uncertainty map based on the SC maps of the experts.
            - soft_sc_or (numpy.ndarray): The softmax-normalized SC map of the original model.
    """
    soft = torch.nn.Softmax()
    soft_sc = []
    
    for i,expert in enumerate(experts):
        sc_map = sc_expert_map(expert,weights[i])
        soft_sc.append(soft(torch.from_numpy(sc_map)).cpu().detach().numpy())

    sc_or_map = sc_expert_map(original,weightsOrig)
    soft_sc_or = soft(torch.from_numpy(sc_or_map)).cpu().detach().numpy()

    sc_map = uncertainty_map(soft_sc,size)

    return sc_map,soft_sc_or



def calculate_soft_maps(maps,original=None,size=4096):
    """
    Calculates soft maps and uncertainty map for a list of input maps.
    Args:
        maps (list): List of input maps to be processed.
        original (optional): An optional original map to be processed. Default is None.
        size (int, optional): The size parameter for the uncertainty map calculation. Default is 4096.
    Returns:
        tuple: A tuple containing:
            - soft_maps (list): List of processed soft maps corresponding to the input maps.
            - soft_mapOrig: Processed soft map of the original input, or None if not provided.
            - soft_uncertainty_map: The calculated uncertainty map based on the soft maps.
    Note:
        This function assumes the existence of `soft_smooth_expert_map` and `uncertainty_map` functions.
    """
    if original!=None:
        soft_mapOrig = soft_smooth_expert_map(original)
    else:
        soft_mapOrig = None
    soft_maps = []
    for map in maps:
        soft_maps.append(soft_smooth_expert_map(map))
        soft_uncertainty_map = uncertainty_map(soft_maps,original=soft_mapOrig,size=size)
    
    
    return soft_maps,soft_mapOrig,soft_uncertainty_map
    


def calculate_smooth_maps(maps,original=None,size=4096):
    """
    Applies a smoothing filter to a list of expert maps and optionally an original map, 
    then computes a smooth uncertainty map based on the filtered results.
    Args:
        maps (list or iterable): A list or iterable of expert maps (numpy arrays) to be smoothed.
        original (numpy.ndarray, optional): The original map to be smoothed and used for uncertainty calculation. Defaults to None.
        size (int, optional): The size parameter used in the uncertainty map calculation. Defaults to 4096.
    Returns:
        tuple: A tuple containing:
            - svls_labels (list): List of smoothed expert maps as numpy arrays.
            - svls_labels_or (numpy.ndarray or None): Smoothed original map as a numpy array, or None if not provided.
            - smooth_uncertainty_map (numpy.ndarray): The computed smooth uncertainty map.
    """
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

    smooth_uncertainty_map= uncertainty_map(svls_labels,original=svls_labels_or,size=size)
    return svls_labels,svls_labels_or,smooth_uncertainty_map
