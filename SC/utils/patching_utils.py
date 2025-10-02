
import numpy as np
import torch
from einops import rearrange
import torch

def get_patches(x, vit_patch, vit_region, patch_size=256, region_size=4096, 
                y=None,mode="features", region=True):
    """
    Extracts patches from x for ViT-based processing.
    
    Args:
        x: Input tensor (image or label map).
        vit_patch: Patch-level transformer model.
        vit_region: Region-level transformer model.
        patch_size: Size of each patch.
        region_size: Size of each region.
        mode: features, label or both
        region: If True, applies region-level transformer.
        
    """
    npatch = int(region_size // patch_size)
    num_patches = npatch ** 2
    mask_patch = None
    ps = patch_size

    label_patches = None
    features = None

    # Label patches
    if mode=='label':
        label_patches = x.unfold(0, ps, ps).unfold(1, ps, ps)
        label_patches = rearrange(label_patches, "p1 p2 w h -> (p1 p2) w h")
    elif mode =='both':
        label_patches = y.unfold(0, ps, ps).unfold(1, ps, ps)
        label_patches = rearrange(label_patches, "p1 p2 w h -> (p1 p2) w h")
    if mode in ["features", "both"]:
        x_ = x.unfold(2, ps, ps).unfold(3, ps, ps)
        x_ = rearrange(x_, "b c p1 p2 w h -> (b p1 p2) c w h")

        patch_features = []
        for mini_bs in range(0, x_.shape[0], num_patches):
            minibatch = x_[mini_bs: mini_bs + num_patches]
            sub_mask_mini_patch = None
            f = vit_patch(minibatch, mask=sub_mask_mini_patch).detach()
            patch_features.append(f.unsqueeze(0))

        features = torch.vstack(patch_features)
        if region:
            features = vit_region(
                features.unfold(1, npatch, npatch).transpose(1, 2),
                mask=mask_patch,
            )

    # Return based on flags
    if mode == "features":
        return features
    elif mode == "label":
        return label_patches
    elif mode == "both":
        return features, label_patches



def recontruct_image_from_patches(patches,path_size=256,region_size=4096):
    """
    Reconstructs a 4096x4096 image from a sequence of 256x256 patches.
    Assumes that the patches are provided in row-major order (left to right, top to bottom).
    Each patch is placed sequentially into the reconstructed image.
    Parameters:
        patches (iterable or list of np.ndarray): Sequence of 2D numpy arrays of shape (256, 256),
            representing image patches.
    Returns:
        np.ndarray: The reconstructed 2D image of shape (4096, 4096), with integer type.
    """
    
    img_reconstructed = np.zeros((region_size,region_size)).astype(int)
    posx = 0
    posy = 0
    for patch in patches:
        img_reconstructed[posy:posy+path_size,posx:posx+path_size] = patch
        posx += path_size
        if posx == region_size:
            posx = 0
            posy += path_size
    return img_reconstructed