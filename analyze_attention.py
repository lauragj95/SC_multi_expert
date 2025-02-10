# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:15:39 2024

@author: Laura
"""

from source import attention_visualization_utils as aux
import numpy as np
import matplotlib
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.io import imread
import os

dir = "D:/datasets/Gleason19/imgs_crop/"
imgs = os.listdir(dir)

if torch.cuda.is_available():
    device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
    print(device)
    

patch_model = aux.get_patch_model(
    pretrained_weights=Path('checkpoints/vit_256_small_dino_fold_4.pt'),
    mask_attn=False,
    device=device,
)
region_model = aux.get_region_model(
    pretrained_weights = Path('checkpoints/vit_4096_xs_dino_fold_4.pt'),
    region_size=4096,
    mask_attn=False,
    device=device,
)

for img in imgs:
    file_name = img.split('.')[0]
    print(file_name)
    region = Image.open(f'D:/datasets/Gleason19/imgs_crop/{img}')
    output_dir = f'attention_maps/indiv_heatmaps/{file_name}_indiv'
    os.makedirs(output_dir, exist_ok=True)
    light_jet = aux.cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.jet)
    aux.create_region_heatmaps_indiv(
        region,
        patch_model,
        region_model,
        output_dir,
        downscale=1,
        threshold=0.5,
        alpha=0.5,
        cmap=light_jet,
        granular=True,
        patch_device=device,
        region_device=device,
    )
    
    # aux.create_region_heatmaps_concat(
    #     region,
    #     patch_model,
    #     region_model,
    #     output_dir,
    #     downscale=1,
    #     alpha=0.5,
    #     cmap=light_jet,
    #     granular=False,
    #     patch_device=device,
    #     region_device=device,
    # )



# ann = imread('D:/datasets/Gleason19/Maps5_T/slide001_core156_classimg_nonconvex.png')
# plt.figure(figsize=(10, 10))
# plt.imshow(ann)

# img = imread('D:/datasets/Gleason19/imgs4096/slide001_core156.jpg')
# plt.figure(figsize=(10, 10))
# plt.imshow(img)