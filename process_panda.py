# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:08:45 2024

@author: Laura
"""

from skimage.io import imread
import numpy as np
from skimage.color import rgb2hsv
import extract_utils
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil

dir = 'D:/datasets/PANDA/'
a = os.listdir(dir + 'imgs/')
b = imread(dir+ 'imgs/'+a[0])
imgs = [file.split('.')[0] for file in os.listdir(dir + 'train_images')]
masks = [file.split('_')[0] for file in os.listdir(dir + 'train_label_masks')]
masks_copy = masks.copy()
imgs_copy = imgs.copy()
        
df = pd.read_csv(dir + 'train.csv', sep = ',')
radboud = df[df['data_provider']=='radboud']['image_id'].tolist()

final_imgs = []
for img in radboud:
    if img in masks:
        final_imgs.append(img)
        
        # for img in imgs:
#     if img not in final_imgs:
#         # os.remove(dir + 'train_images/' + img + '.tiff')
#         imgs_copy.remove(img)

# for mask in masks:
#     if mask in final_imgs:
#         # os.remove(dir + 'train_images/' + img + '.tiff')
#         shutil.copy(dir + 'train_label_masks/' + mask + '_mask.tiff', dir + 'masks/' + mask + '_mask.tiff')
#         # masks_copy.remove(mask)
        
# mask0 = imread(dir + 'train_label_masks/0018ae58b01bdadc8e347995b69f99aa_mask.tiff')

all = []
for img in final_imgs:
    try:
        mask = imread(dir + 'train_label_masks/' +img+'_mask.tiff')[:,:,0]
        if len(np.unique(mask)) == 5 or 5 in np.unique(mask):
            all.append(img)
    except:
        print('error')
       
classes = {}
img_dict = {'slide_id':[],'slide_path':[]}
for a in final_imgs:
    img_dict['slide_id'].append(a)
    img_dict['slide_path'].append(dir + 'imgs/' + a + '.tiff')
    shutil.copy(dir + 'train_label_masks/' + a + '_mask.tiff', dir + 'masks/' + a + '.tiff')
    shutil.copy(dir + 'train_images/' + a + '.tiff', dir + 'imgs/' + a + '.tiff')
    # mask = imread(dir + 'train_label_masks/' +a+'_mask.tiff')[:,:,0]
    # classes[a] = np.unique(mask,return_counts=True)

img_df = pd.DataFrame.from_dict(img_dict)

img_df.to_csv('D:/multi-expert/hs2p-master/slides.csv',index=False)
    