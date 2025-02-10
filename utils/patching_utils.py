
import numpy as np
import torch
from einops import rearrange
import torch


def get_patches(x,label, vit_patch,vit_region,patch_size,region_size):
    npatch = int(region_size // patch_size)
    num_patches = npatch**2
    mask_patch = None
    ps = 256
    x = x.unfold(2, ps, ps).unfold(
        3, ps, ps
    )
    x = rearrange(
        x, "b c p1 p2 w h -> (b p1 p2) c w h"
    )

    label = label.unfold(1, ps, ps).unfold(
        2, ps, ps
    )
    label = rearrange(
        label, "b p1 p2 w h -> (b p1 p2) w h"
    )
   
    
    patch_features = []
    for mini_bs in range(0, x.shape[0], num_patches):
        minibatch = x[
            mini_bs : mini_bs + num_patches
        ]
        sub_mask_mini_patch = None
        
        f = vit_patch(
            minibatch, mask=sub_mask_mini_patch
        ).detach()
        patch_features.append(f.unsqueeze(0))
    
    x = torch.vstack(patch_features)
    x = vit_region(
        x.unfold(1, npatch, npatch).transpose(1, 2),
        mask=mask_patch,
    ) 
    return x, label

def recontruct_image_from_patches(patches):
    
    img_reconstructed = np.zeros((4096,4096)).astype(int)
    posx = 0
    posy = 0
    for patch in patches:
        img_reconstructed[posy:posy+256,posx:posx+256] = patch
        posx += 256
        # posy += 256 
        if posx == 4096:
            posx = 0
            posy += 256
    return img_reconstructed