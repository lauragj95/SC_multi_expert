# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:40:19 2024

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
import cv2
import extract_utils
import os 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import spatial


def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img

if torch.cuda.is_available():
    device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
    print(device)

dir = 'D:/datasets/Gleason19/'



patch_model = aux.get_patch_model(
    pretrained_weights=Path('runs/test/epoch_090.pt'),
    mask_attn=False,
    device=device,
)
region_model = aux.get_region_model(
    pretrained_weights = Path('checkpoints/vit_4096_xs_dino_fold_4.pt'),
    region_size=4096,
    mask_attn=False,
    device=device,
)

# patches, patch_attention, region_attention = aux.get_region_attention_scores(
#     region,
#     patch_model,
#     region_model,
#     patch_size = 256,
#     mini_patch_size = 16,
#     downscale = 1
# )

file_name = 'slide001_core156'

region = Image.open(f'{dir}imgs4096/{file_name}.jpg')
scores = aux.get_attention_patch_region(
    region,
    patch_model,
    region_model,
    downscale=1,
    granular=True,
    patch_device=device,
    region_device=device,
)
mask = imread(f'{dir}Maps1_T/{file_name}.png')
mask_crop = center_crop(mask,(4096,4096))




# plt.figure(figsize=(10, 10))
# plt.imshow(att_patches[0][:,:,:3].astype('uint8'))

# plt.figure(figsize=(10, 10))
# plt.imshow(att_patches[-2][:,:,-2], cmap='hot')
# plt.show()


# img = imread('D:/datasets/Gleason19/imgs4096/slide001_core156.jpg')
# plt.figure(figsize=(10, 10))
# plt.imshow(img)



